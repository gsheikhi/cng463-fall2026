import json

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from src.training import DEVICE, finished, fit, schedule

MODEL = "HuggingFaceTB/SmolLM2-135M"
TEMPLATE = "### Instruction:\n{}\n\n### Response:\n"
BETA = 0.1  # strength of the pull towards the reference model in DPO


def load_lm(path=MODEL):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, AutoModelForCausalLM.from_pretrained(path).to(DEVICE)


def to_prompt_completion(row):
    # Convert a row to a prompt-completion pair for SFT. The prompt is the instruction and input, and the completion is the output.
    # TODO
    raise NotImplementedError


def load_sft_data():
    data = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=0).select(range(2200))
    return data.map(to_prompt_completion, remove_columns=data.column_names)


def train_sft(model, tokenizer, train_set, val_set):
    out_dir = "runs/sft"
    if finished(out_dir):
        return
    train_set, val_set = (d.map(lambda r: {"completion": r["completion"] + tokenizer.eos_token})
                          for d in (train_set, val_set))
        # Batches of 2 with 4 accumulation steps act like batches of 8 but use far less memory.
    args = SFTConfig(out_dir, num_train_epochs=1, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                     gradient_accumulation_steps=4, learning_rate=1e-4, max_length=512, fp16=DEVICE == "cuda", **schedule(25))
    fit(SFTTrainer(model=model, args=args, train_dataset=train_set, eval_dataset=val_set,
                   processing_class=tokenizer))


def words(text):
    return len(text.split())


def load_pref_data():
    """UltraFeedback pairs whose chosen answer is short and much shorter than the rejected one."""
    data = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    data = data.filter(lambda r: words(r["prompt"]) <= 60
                       and words(r["chosen"][-1]["content"]) <= 120
                       and words(r["rejected"][-1]["content"]) >= 1.5 * words(r["chosen"][-1]["content"]))
    data = data.shuffle(seed=0).select(range(1100))
    return data.map(lambda r: {"prompt": TEMPLATE.format(r["prompt"]), "chosen": r["chosen"][-1]["content"],
                               "rejected": r["rejected"][-1]["content"]}, remove_columns=data.column_names)


def read_pairs(path):
    """Read a JSON Lines file with the keys prompt, chosen and rejected."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list([{"prompt": TEMPLATE.format(r["prompt"]), "chosen": r["chosen"],
                               "rejected": r["rejected"]} for r in rows])


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected):
    # Compute the DPO loss for a batch of chosen and rejected responses. The arguments are tensors of log-probabilities of the chosen and rejected responses under the policy and reference models.
    raise NotImplementedError


def train_dpo(model, tokenizer, train_set, val_set):
    out_dir = "runs/dpo"
    if finished(out_dir):
        return
    args = DPOConfig(out_dir, beta=BETA, num_train_epochs=2, per_device_train_batch_size=2,
                     per_device_eval_batch_size=2, gradient_accumulation_steps=4, learning_rate=5e-6, max_length=512, max_prompt_length=128,
                     fp16=DEVICE == "cuda", **schedule(25))
    fit(DPOTrainer(model=model, args=args, train_dataset=train_set, eval_dataset=val_set,
                   processing_class=tokenizer))


def generate(model, tokenizer, instructions, max_new_tokens=200):
    """Greedy answers. Returns (answer, stopped) pairs. stopped is True if the model ended the answer itself."""
    model.eval()
    results = []
    for instruction in instructions:
        enc = tokenizer(TEMPLATE.format(instruction), return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
        new = ids[0, enc.input_ids.shape[1]:]
        results.append((tokenizer.decode(new, skip_special_tokens=True).strip(),
                        tokenizer.eos_token_id in new.tolist()))
    return results


def sequence_logprob(model, tokenizer, prompt, response):
    """Sum of the log-probabilities of the response tokens and EOS, given the prompt."""
    prompt_len = len(tokenizer(prompt).input_ids)
    ids = tokenizer(prompt + response + tokenizer.eos_token, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logp = model(ids).logits[0, :-1].log_softmax(-1)
    token_logp = logp.gather(1, ids[0, 1:, None]).squeeze(1)
    return token_logp[prompt_len - 1:].sum().item()
