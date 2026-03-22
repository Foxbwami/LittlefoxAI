import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core import config

DATA = Path(config.DATA_DIR) / "training" / "ner_train.jsonl"
OUT_DIR = Path(config.BASE_DIR) / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows():
    rows = []
    with open(DATA, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_rows()
    gazetteer = {}
    for row in rows:
        for ent in row.get("entities", []):
            label = ent["label"]
            gazetteer.setdefault(label, set()).add(ent["text"])
    output = {label: sorted(list(values)) for label, values in gazetteer.items()}
    out_path = OUT_DIR / "ner_gazetteer.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("NER gazetteer trained.")


if __name__ == "__main__":
    main()
