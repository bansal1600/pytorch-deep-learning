# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # V2-03: Byte Pair Encoding — Solution

# %%
from collections import Counter, defaultdict
from torch.utils.data import Dataset


class DummyDataGenerator:
    def __init__(self):
        self.corpus = ["low", "lower", "newest", "widest"]
    def text(self):
        return self.corpus


class CorpusDataset(Dataset):
    def __init__(self, words):
        self.words = words
    def __len__(self): return len(self.words)
    def __getitem__(self, i): return self.words[i]


def get_vocab(corpus):
    vocab = Counter()
    for word in corpus:
        vocab[tuple(list(word) + ["</w>"])] += 1
    return vocab


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        w = " ".join(word).replace(bigram, replacement)
        new_vocab[tuple(w.split())] = freq
    return new_vocab


class BPETokenizer:
    def __init__(self):
        self.merges = []
    def train(self, corpus, num_merges=10):
        vocab = get_vocab(corpus)
        for _ in range(num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)
            self.merges.append(best)
        return self
    def tokenize(self, word):
        tokens = list(word) + ["</w>"]
        for a, b in self.merges:
            merged = a + b
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i + 1] == b:
                    tokens = tokens[:i] + [merged] + tokens[i + 2:]
                else:
                    i += 1
        return tokens


corpus = DummyDataGenerator().text()
tok = BPETokenizer().train(corpus, num_merges=5)
print(f"merges: {tok.merges}")
print(f"tokenize 'lower': {tok.tokenize('lower')}")
print("✓ BPE trained and tokenizing")
