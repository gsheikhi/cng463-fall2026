# Assignment 1: A small language pipeline

CNG463 Introduction to NLP, Fall 2026. Due 31/10/2026 11:59 p.m. Cyprus Time.

You turn a book PDF into a corpus, then build a tokeniser, byte-pair encoding and n-gram language models. The tasks, marks, rules and submission steps are in `assignment1.ipynb`. This file explains the setup only.

## Files

`assignment1.ipynb` is the notebook you run. `src/` holds the functions you implement. `text.py` covers the corpus, tokenisation and data split, `bpe.py` covers byte-pair encoding and `lm.py` covers the language models. Put your PDF in `data/`. The notebook writes your sentences to `data/corpus.txt`.

## Running locally in VS Code

Install Python 3.10 or later and Git, then run these commands in a terminal.

```
git clone https://github.com/<organisation>/cng463-assignments.git
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
!git clone https://github.com/<organisation>/cng463-assignments.git
```

Then open `cng463-assignments/assignment1/assignment1.ipynb` from Drive. Its setup cell mounts Drive and installs the requirements in each session. Open the files in `src/` from the Colab file browser to edit them. Your changes are saved to Drive.

## Choosing a book

Use a PDF with a text layer. If you cannot select its text in a PDF viewer, it will not work. Public-domain or openly licensed books are the safest choice, for example from Project Gutenberg, Wikisource or the Internet Archive. If a site offers only HTML or EPUB, print the book to PDF. You may use a book under copyright that you own, but never push it to a public repository. Choose a page range with 5,000 to 50,000 words and at least 200 sentences.
