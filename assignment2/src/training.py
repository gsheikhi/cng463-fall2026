"""Checkpoints, a live loss plot and early stopping, used by train_ner, train_sft and train_dpo.

Trainer calls every callback method with (args, state, control, **kwargs), so the methods below keep
these arguments even when they do not use them."""
import gc
import os
import shutil
from glob import glob

import matplotlib.pyplot as plt
import torch
from IPython.display import Image, display
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
PATIENCE = 3  # evaluations in a row without a new lowest validation loss before training stops


def schedule(eval_steps):
    """Log and evaluate every eval_steps steps. Save a checkpoint every 2 * eval_steps steps.
    save_total_limit=1 keeps only the latest checkpoint, so each new checkpoint replaces the last one."""
    return dict(eval_strategy="steps", eval_steps=eval_steps, logging_steps=eval_steps,
                save_strategy="steps", save_steps=2 * eval_steps, save_total_limit=1, report_to="none")


class EarlyStopping(TrainerCallback):
    """Stops training when none of the last PATIENCE evaluations beat the lowest validation loss so far."""

    def on_evaluate(self, args, state, control, **kwargs):
        losses = [log["eval_loss"] for log in state.log_history if "eval_loss" in log]
        evals_since_best = len(losses) - 1 - losses.index(min(losses))
        if evals_since_best >= PATIENCE:
            print(f"Early stopping at step {state.global_step}: no lower validation loss in {PATIENCE} evaluations.")
            control.should_training_stop = True


class LossPlot(TrainerCallback):
    """Redraws the training and validation loss every time the trainer logs."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        plt.close(self.fig)  # the figure is shown only through the display handle below
        self.handle = display(self.fig, display_id=True)

    def on_log(self, args, state, control, **kwargs):
        self.ax.clear()
        for key, name in [("loss", "training"), ("eval_loss", "validation")]:
            points = [(log["step"], log[key]) for log in state.log_history if key in log]
            if points:
                self.ax.plot(*zip(*points), marker="o", label=name)
        self.ax.set(xlabel="step", ylabel="loss")
        self.ax.legend()
        self.handle.update(self.fig)


def free_memory():
    """Give GPU memory that no variable uses any more back to the system."""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def finished(out_dir):
    """True if fit already saved a final model to out_dir. Then it also shows the loss plot saved with it."""
    if os.path.exists(os.path.join(out_dir, "config.json")):
        print(f"Using the model already trained in {out_dir}. Delete that folder to train again.")
        if os.path.exists(os.path.join(out_dir, "loss.png")):
            display(Image(os.path.join(out_dir, "loss.png")))
        return True
    return False


def fit(trainer):
    """Train with a live loss plot and early stopping, resuming from the latest checkpoint in the output folder
    if an earlier run was interrupted. Saves the final model and its loss plot there and deletes the checkpoint.
    A resumed run keeps the settings it started with, such as the batch size, so after changing a
    setting, delete the output folder to train from scratch."""
    out_dir = trainer.args.output_dir
    free_memory()
    plot = LossPlot()
    trainer.add_callback(EarlyStopping())
    trainer.add_callback(plot)
    trainer.add_callback(LossPlot())
    checkpoint = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    if checkpoint:
        print("Resuming from", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)
    plot.fig.savefig(os.path.join(out_dir, "loss.png"))
    trainer.save_model(out_dir)
    for folder in glob(os.path.join(out_dir, "checkpoint-*")):
        shutil.rmtree(folder)
