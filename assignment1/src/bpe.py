from collections import Counter

END = "</w>"


def merge_symbols(symbols, pair):
    # TODO [Task 5]: Return a tuple in which each occurrence of pair is merged, left to right, without overlaps.
    raise NotImplementedError


def learn_bpe(word_counts, num_merges):
    # TODO [Task 5]: Return a list of ((left, right), count) in the order the merges were learned.
    raise NotImplementedError


def encode_word(word, merges):
    # TODO [Task 6]: Return the tuple of pieces after applying the merges in order.
    raise NotImplementedError
