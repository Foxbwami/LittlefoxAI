import re


def looks_gibberish(text):
    if not text:
        return True
    if len(text) < 10:
        return True
    # too many repeated chars
    if re.search(r"(.)\\1{6,}", text):
        return True
    # low alpha ratio
    letters = sum(1 for c in text if c.isalpha())
    ratio = letters / max(len(text), 1)
    if ratio < 0.4:
        return True
    # repeated bigrams
    words = text.lower().split()
    if len(words) > 8:
        pairs = list(zip(words, words[1:]))
        if pairs:
            repeat = max(pairs.count(p) for p in set(pairs))
            if repeat / max(len(pairs), 1) > 0.35:
                return True
    return False


def compress_repetition(text):
    if not text:
        return text
    # collapse repeating tokens
    out = []
    last = None
    rep = 0
    for tok in text.split():
        if tok == last:
            rep += 1
            if rep >= 2:
                continue
        else:
            rep = 0
        out.append(tok)
        last = tok
    return " ".join(out)
