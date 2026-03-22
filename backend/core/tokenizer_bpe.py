import collections
import json
import os
import re
from backend.core import config

# =========================
# SPECIAL TOKENS
# =========================
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|unk|>"
]

PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"


class BPETokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = []
        self.stoi = {}
        self.itos = {}
        self.merges = []
        self.bpe_ranks = {}
        self.cache = {}

    # =========================
    # TOKENIZATION (BETTER)
    # =========================
    def tokenize(self, text):
        if not text:
            return []
        special = "|".join(re.escape(tok) for tok in SPECIAL_TOKENS)
        pattern = rf"{special}|\w+|[^\w\s]"
        return re.findall(pattern, text.lower())

    # =========================
    # GET STATS
    # =========================
    def get_stats(self, tokens):
        pairs = collections.Counter()
        for word in tokens:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    # =========================
    # MERGE
    # =========================
    def merge_vocab(self, pair, tokens):
        new_tokens = []
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word in tokens:
            new_word = word.replace(bigram, replacement)
            new_tokens.append(new_word)

        return new_tokens

    # =========================
    # TRAIN
    # =========================
    def train(self, text):
        words = self.tokenize(text)
        tokens = [" ".join(list(word)) + " </w>" for word in words]

        merges = []
        for _ in range(self.vocab_size):
            pairs = self.get_stats(tokens)
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            tokens = self.merge_vocab(best, tokens)
            merges.append(best)

        vocab = set(" ".join(tokens).split())

        # ADD SPECIAL TOKENS
        vocab.update(SPECIAL_TOKENS)

        self.vocab = sorted(vocab)
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.merges = merges
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    # =========================
    # BPE (WITH CACHE)
    # =========================
    def bpe(self, word):
        if word in self.cache:
            return self.cache[word]

        tokens = list(word) + ["</w>"]

        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            candidates = [p for p in pairs if p in self.bpe_ranks]

            if not candidates:
                break

            best = min(candidates, key=lambda p: self.bpe_ranks[p])

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        self.cache[word] = tokens
        return tokens

    # =========================
    # ENCODE (FIXED)
    # =========================
    def encode(self, text):
        tokens = []

        for word in self.tokenize(text):
            if word in self.stoi:
                tokens.append(self.stoi[word])
            else:
                for sub in self.bpe(word):
                    # Use get() with safe default fallback
                    unk_id = self.stoi.get(UNK_TOKEN, 0)  # Default to 0 if UNK not found
                    tokens.append(self.stoi.get(sub, unk_id))

        return tokens

    # =========================
    # DECODE (CLEAN)
    # =========================
    def decode(self, token_ids):
        text = ""

        for i in token_ids:
            token = self.itos.get(int(i), UNK_TOKEN)
            if token in SPECIAL_TOKENS:
                continue
            if token.endswith("</w>"):
                text += token[:-4] + " "
            else:
                text += token

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text.strip()

    # =========================
    # SAVE / LOAD
    # =========================
    def save(self, path):
        data = {
            "vocab": self.vocab,
            "merges": [list(p) for p in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.merges = [tuple(p) for p in data.get("merges", [])]
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(self.merges)}


# =========================
# LOAD TEXT
# =========================
def _load_text(max_chars=None):
    for path in (config.PROCESSED_DATA_PATH, config.RAW_DATA_PATH, "data.txt"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read(max_chars).lower() if max_chars else f.read().lower()
    return ""


# =========================
# INIT TOKENIZER
# =========================
tokenizer = BPETokenizer(vocab_size=config.BPE_VOCAB_SIZE)

if os.path.exists(config.TOKENIZER_PATH):
    tokenizer.load(config.TOKENIZER_PATH)
else:
    text = _load_text(config.TOKENIZER_TRAIN_CHARS)
    tokenizer.train(text)

    os.makedirs(os.path.dirname(config.TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(config.TOKENIZER_PATH)


vocab_size = len(tokenizer.vocab)


# =========================
# API
# =========================
def encode(text):
    return tokenizer.encode(text)


def decode(tokens):
    return tokenizer.decode(tokens)


# =========================
# DIAGNOSTIC
# =========================
def _diagnose_sample(text):
    ids = encode(text)
    roundtrip = decode(ids)
    print("input:", text)
    print("ids:", ids[:60], "..." if len(ids) > 60 else "")
    print("output:", roundtrip)


if __name__ == "__main__":
    _diagnose_sample("short term operations! business finance.")
