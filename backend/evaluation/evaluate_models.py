from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json
from backend.core import config
from backend.services.ner import extract_entities
from backend.services.moderation import check_safety
from backend.tools.tool_selector import select_tool
from backend.services.fact_check import fact_check_claim
from backend.services.emotion import detect_tone


DATA = Path(config.DATA_DIR) / "training"


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def eval_ner():
    rows = load_jsonl(DATA / "ner_train.jsonl")
    if not rows:
        return 0.0
    hits = 0
    total = 0
    for row in rows:
        ents = extract_entities(row["text"])
        gold = {(e["text"], e["label"]) for e in row["entities"]}
        pred = {(e["text"], e["label"]) for e in ents}
        total += len(gold)
        hits += len(gold & pred)
    return hits / total if total else 0.0


def eval_safety():
    rows = load_jsonl(DATA / "safety_train.jsonl")
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        verdict = check_safety(row["text"])
        allowed = verdict.get("allowed", True)
        if row["label"] == "safe" and allowed:
            correct += 1
        if row["label"] == "unsafe" and not allowed:
            correct += 1
    return correct / max(len(rows), 1)


def eval_tool_select():
    rows = load_jsonl(DATA / "tool_select_train.jsonl")
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        if select_tool(row["text"]) == row["tool"]:
            correct += 1
    return correct / max(len(rows), 1)


def eval_fact_check():
    rows = load_jsonl(DATA / "fact_check_train.jsonl")
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        verdict = fact_check_claim(row["claim"], sources=[], allow_web=False)
        if verdict.get("verdict") == row["verdict"]:
            correct += 1
    return correct / max(len(rows), 1)


def eval_emotion():
    samples = [
        ("I am stressed and tired.", "empathetic"),
        ("There is an error in my code.", "helpful"),
        ("Thanks for your help!", "warm"),
        ("I feel great about this.", "warm"),
        ("This is awful and frustrating.", "empathetic"),
        ("Not sure yet, I need more details.", "neutral"),
    ]
    correct = 0
    for text, label in samples:
        if detect_tone(text) == label:
            correct += 1
    return correct / len(samples)


if __name__ == "__main__":
    scores = {
        "NER": eval_ner(),
        "Safety": eval_safety(),
        "ToolSelection": eval_tool_select(),
        "FactCheck": eval_fact_check(),
        "Emotion": eval_emotion(),
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("Model Scores:")
    for name, score in ranked:
        print(f"{name}: {score:.2f}")
