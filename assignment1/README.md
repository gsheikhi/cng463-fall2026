# Assignment 1: A small language pipeline

Due 31 Oct 2026 before midnight, Cyprus time.

You turn a book PDF into a corpus, then build a tokeniser, byte-pair encoding and n-gram language models. 

## Files

`assignment1.ipynb` is the notebook you run. `src/` holds the functions you implement: `text.py` covers the corpus, tokenisation and data split, `bpe.py` covers byte-pair encoding and `lm.py` covers the language models. Put your PDF in `data/`. The notebook writes your sentences to `data/corpus.txt`.

## Running locally in VS Code

Install Python 3.10 or later and Git, then run these commands in a terminal.

```
git clone https://github.com/gsheikhi/cng463-fall2026.git
cd cng463-assignments/assignment1
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open `assignment1.ipynb` in VS Code and select the `.venv` interpreter as the kernel. The notebook must run from the `assignment1` folder.

## Running in Colab

Colab deletes its files when a session ends, so keep your copy in Google Drive. Once, in a new Colab notebook, run the following.

```
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive
!git clone https://github.com/gsheikhi/cng463-fall2026.git
```

Then open `cng463-assignments/assignment1/assignment1.ipynb` from Drive. Its setup cell mounts Drive and installs the requirements in each session. Open the files in `src/` from the Colab file browser to edit them. Your changes are saved to Drive.


## Rules

Use only the Python standard library, NumPy, pandas, Matplotlib and `pypdf`. Part II of Duo Quiz 1 asks about your own submission and multiplies your mark for this assignment.

## Submission

Restart the kernel, run all cells and save the notebook with its outputs. Zip the `assignment1` folder with `src/`, your PDF and `data/corpus.txt`, leaving out `.venv`. Upload the ZIP to the course platform. Late submissions follow the syllabus.
