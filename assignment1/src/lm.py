import math
import random
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NGramLM:
    def __init__(self, n, alpha=1.0, min_count=2):
        self.n, self.alpha, self.min_count = n, alpha, min_count

    def fit(self, sentences):
        # TODO [Task 7]: Set self.vocab (a sorted list of prediction tokens) and count
        # each context and its next token. Return self.
        raise NotImplementedError

    def probability(self, token, context=()):
        # TODO [Task 7]: Return the add-alpha estimate of P(token | context).
        raise NotImplementedError

    def perplexity(self, sentences):
        # TODO [Task 8]: Return exp of the mean negative log probability, including one EOS per sentence.
        raise NotImplementedError

    def generate(self, seed, max_tokens=40):
        # TODO [Task 8]: Sample with random.Random(seed) until EOS or max_tokens. Return the tokens.
        raise NotImplementedError
