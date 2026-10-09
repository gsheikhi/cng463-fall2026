import re
import random
import unicodedata
from collections import Counter
from pypdf import PdfReader


def extract_pages(path, first, last):
    """Return the text of pages first..last (1-based, inclusive)."""
    reader = PdfReader(path)
    return [reader.pages[i].extract_text() for i in range(first - 1, last)]


def clean_pages(pages):
    # TODO [Task 1]: Remove page numbers, running headers and headings from each page.
    raise NotImplementedError


def normalise_text(pages):
    # TODO [Task 1]: Join the pages into one string. Rejoin words split by a hyphen at a line end
    # and replace ligatures such as "ﬁ".
    raise NotImplementedError


def split_into_sentences(text):
    # TODO [Task 1]: Return a list of sentences. Do not split after abbreviations such as "Mr.".
    raise NotImplementedError

def save_sentences(sentences, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences) + "\n")
        
def tokenise(text):
    # TODO [Task 2]: Return a list of lowercase tokens.
    raise NotImplementedError


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        return [tokenise(line) for line in f if line.strip()]


def deduplicate(sentences):
    # TODO [Task 3]: Keep the first copy of each tokenised sentence.
    raise NotImplementedError


def split_blocks(sentences, block_size=10, seed=42):
    # TODO [Task 3]: Shuffle blocks of consecutive sentences with random.Random(seed).
    # Return train, dev and test sentence lists made from 80/10/10 of the blocks.
    raise NotImplementedError


def corpus_stats(sentences):
    # TODO [Task 4]: Return the token Counter, N, V and V/N.
    raise NotImplementedError
