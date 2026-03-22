import random

_STARTERS = [
    "Alright,",
    "So,",
    "Here's the thing,",
    "Well,",
]


def add_human_touch(text, tone="neutral"):
    if not text:
        return text
    if tone == "academic":
        return text
    lower = text.lower()
    if any(lower.startswith(greet) for greet in ["hi", "hello", "hey"]):
        return text
    starter = random.choice(_STARTERS)
    if text[0].isupper():
        return f"{starter} {text}"
    return f"{starter} {text}"
