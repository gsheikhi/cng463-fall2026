import os
import re

import torch
from seqeval.metrics import f1_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)

from src.training import DEVICE, finished, fit, schedule

MODEL = "distilbert-base-cased"
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)})
    return tokenizer, model.to(DEVICE)


def align_labels(word_labels, word_ids):
    """Give each word's label to its first token and -100 to special tokens and later subword tokens."""
    # TODO
    raise NotImplementedError


def train_ner(model, tokenizer, train_set, val_set):
    out_dir = "runs/ner"
    if finished(out_dir):
        return AutoModelForTokenClassification.from_pretrained(out_dir).to(DEVICE)

    def encode(batch):
        enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
        enc["labels"] = [align_labels(tags, enc.word_ids(i)) for i, tags in enumerate(batch["ner_tags"])]
        return enc

    train_data, val_data = (d.map(encode, batched=True, remove_columns=d.column_names) for d in (train_set, val_set))
    args = TrainingArguments(out_dir, num_train_epochs=2, per_device_train_batch_size=16,
                             learning_rate=5e-5, fp16=DEVICE == "cuda", **schedule(200))
    fit(Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
                data_collator=DataCollatorForTokenClassification(tokenizer)))
    return model


def predict_ner(model, tokenizer, sentences):
    """Return one label per word for each sentence. A sentence is a list of words."""
    model.eval()
    results = []
    for words in sentences:
        enc = tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model(**enc).logits[0].argmax(-1).tolist()
        labels = {}
        for pos, w in enumerate(enc.word_ids()):
            if w is not None and w not in labels:
                labels[w] = LABELS[ids[pos]]
        results.append([labels.get(i, "O") for i in range(len(words))])
    return results


def entity_f1(gold, pred):
    """Entity-level F1. Both arguments are lists of label lists."""
    return f1_score(gold, pred)


def blank_annotation(sentences, path):
    """Write one word per line with the label O and a blank line after each sentence."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists. Delete it first if you want to start again.")
    with open(path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for word in re.findall(r"\w+(?:['’-]\w+)*|[^\w\s]", sentence):
                f.write(f"{word}\tO\n")
            f.write("\n")


def read_annotation(path):
    """Return the words and labels of a file written by blank_annotation."""
    words, labels = [[]], [[]]
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                if words[-1]:
                    words.append([])
                    labels.append([])
                continue
            if len(parts) != 2 or parts[1] not in LABELS:
                raise ValueError(f"{path}, line {n}: expected a word and one of {LABELS}")
            words[-1].append(parts[0])
            labels[-1].append(parts[1])
    if not words[-1]:
        words.pop()
        labels.pop()
    return words, labels
