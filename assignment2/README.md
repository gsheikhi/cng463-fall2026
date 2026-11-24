# Assignment 2: Encoders, and decoders

CNG463 Introduction to NLP, Fall 2026. Due 12/11/2026 11:59 p.m. Cyprus Time.

The tasks, marks, rules and submission steps are in `assignment2.ipynb`. This file explains the setup only and it is in a similar format to Assignment 1.

## Files

`assignment2.ipynb` is the notebook you run. `src/` holds the given training code and the functions you implement. `training.py` holds the checkpoint, loss plot and early stopping code shared by all parts. `ner.py` covers Part A (DistilBERT, named entity recognition), and `preference.py` covers Part B (SmolLM2-135M, SFT and DPO). `data/` holds the files you label and write. Training writes to `runs/`. Do not submit `runs/`.

## GPU

You need a GPU. A free Colab T4 is enough. So is a Mac with Apple silicon. All training together takes about 60 minutes. Free Colab limits GPU time per day, so do not retrain without a reason. Every model is saved to `runs/`. During training, a checkpoint is saved regularly and replaces the previous one, so if a session ends, running the cell again continues from the latest checkpoint. A finished model is loaded from `runs/` instead of being trained again. Delete its folder in `runs/` to train it from scratch. Files deleted in Drive go to its bin, so empty the bin if Drive runs out of space.

## Running in Colab

Colab deletes its files when a session ends, so keep your copy in Google Drive. Once, in a new Colab notebook, run the following.

```
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive
!git clone https://github.com/<organisation>/cng463-assignments.git
```

Then open `cng463-assignments/assignment2/assignment2.ipynb` from Drive and choose Runtime > Change runtime type > T4 GPU. Its setup cell mounts Drive and installs the requirements in each session. Open the files in `src/` and `data/` from the Colab file browser to edit them. Your changes are saved to Drive.

## Running locally in VS Code

Use this only if your computer has an NVIDIA GPU or is a Mac with Apple silicon (M1 or later). The code uses the Mac's GPU automatically, but training takes about three to four times as long as on a T4. On a computer without either, use Colab. Install Python 3.10 or later and Git, then run these commands in a terminal. Use only the packages in `requirements.txt`.

```
git clone https://github.com/<organisation>/cng463-assignments.git
cd cng463-assignments/assignment2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open `assignment2.ipynb` in VS Code and select the `.venv` interpreter as the kernel. The notebook must run from the `assignment2` folder.
