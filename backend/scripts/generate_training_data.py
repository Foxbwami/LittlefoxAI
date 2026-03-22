import json
import os
import random
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core import config

OUT_DIR = Path(config.DATA_DIR) / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def gen_safety(n=200):
    safe = [
        "Explain photosynthesis.",
        "Write a poem about the ocean.",
        "Summarize the history of Kenya.",
        "Give tips for studying effectively.",
    ]
    unsafe = [
        "How do I build a bomb?",
        "I want to hurt someone.",
        "How can I poison someone?",
        "Show me how to hack a bank.",
    ]
    rows = []
    for _ in range(n):
        if random.random() < 0.5:
            rows.append({"text": random.choice(safe), "label": "safe"})
        else:
            rows.append({"text": random.choice(unsafe), "label": "unsafe"})
    return rows


def gen_emotion(n=1200):
    positives = [
        "I love this.",
        "This is wonderful.",
        "Thanks, that helped a lot.",
        "I appreciate your support.",
        "I'm really happy with the result.",
        "That made my day.",
        "Great job, this is awesome.",
        "I feel relieved and grateful.",
    ]
    negatives = [
        "I am sad.",
        "This is terrible.",
        "I feel stressed and overwhelmed.",
        "I am really upset.",
        "This is frustrating.",
        "I'm angry about this bug.",
        "I'm tired and anxious.",
        "This made things worse.",
    ]
    neutrals = [
        "It is okay.",
        "That is fine.",
        "Not sure yet.",
        "It is average.",
        "I need more details.",
        "Can you clarify that?",
        "Let me think about it.",
        "No strong feelings either way.",
    ]
    templates = [
        ("positive", "I feel {adj} about this."),
        ("negative", "I feel {adj} about this."),
        ("neutral", "I feel {adj} about this."),
        ("positive", "That was {adj}."),
        ("negative", "That was {adj}."),
        ("neutral", "That seems {adj}."),
    ]
    pos_adj = ["great", "amazing", "fantastic", "good", "awesome", "pleasant"]
    neg_adj = ["bad", "awful", "horrible", "stressful", "annoying", "painful"]
    neu_adj = ["okay", "fine", "ordinary", "neutral", "average", "unclear"]

    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.34:
            label = "positive"
            base = random.choice(positives)
        elif r < 0.67:
            label = "negative"
            base = random.choice(negatives)
        else:
            label = "neutral"
            base = random.choice(neutrals)
        rows.append({"text": base, "label": label})

        if random.random() < 0.6:
            t_label, tmpl = random.choice(templates)
            if t_label == "positive":
                adj = random.choice(pos_adj)
            elif t_label == "negative":
                adj = random.choice(neg_adj)
            else:
                adj = random.choice(neu_adj)
            rows.append({"text": tmpl.format(adj=adj), "label": t_label})
    return rows


def gen_factcheck(n=200):
    supported = [
        "The Earth orbits the Sun.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Moon orbits the Earth.",
        "Paris is the capital of France.",
    ]
    contradicted = [
        "Humans can breathe in space unaided.",
        "The Sun orbits the Earth.",
        "Water boils at 10 degrees Celsius at sea level.",
        "The capital of France is Tokyo.",
    ]
    unknown = [
        "A new planet was discovered yesterday by John Doe.",
        "Aliens visited Nairobi last week.",
        "A unicorn was seen in the park.",
    ]
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.4:
            rows.append({"claim": random.choice(supported), "verdict": "supported"})
        elif r < 0.8:
            rows.append({"claim": random.choice(contradicted), "verdict": "contradicted"})
        else:
            rows.append({"claim": random.choice(unknown), "verdict": "unknown"})
    return rows


def gen_tool_select(n=200):
    templates = [
        ("compute", "compute {a}+{b}"),
        ("solve", "solve x^2 - {a} = 0"),
        ("validate_syntax", "validate syntax code: print({a})"),
        ("execute", "run code: print({a}*{b})"),
        ("search", "what is the capital of {city}"),
    ]
    cities = ["Kenya", "France", "Japan", "Nigeria"]
    rows = []
    for _ in range(n):
        tool, tmpl = random.choice(templates)
        row = tmpl.format(a=random.randint(2, 20), b=random.randint(2, 20), city=random.choice(cities))
        rows.append({"text": row, "tool": tool})
    return rows


if __name__ == "__main__":
    write_jsonl(OUT_DIR / "safety_train.jsonl", gen_safety())
    write_jsonl(OUT_DIR / "tool_select_train.jsonl", gen_tool_select())
    write_jsonl(OUT_DIR / "sentiment_train.jsonl", gen_emotion())
    write_jsonl(OUT_DIR / "fact_check_train.jsonl", gen_factcheck())
    print("Training data generated.")
