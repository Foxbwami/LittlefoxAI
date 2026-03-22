import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.core import config

DATA = Path(config.DATA_DIR) / "training" / "fact_check_train.jsonl"
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
    texts = [r["claim"] for r in rows]
    labels = [r["verdict"] for r in rows]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(X, labels)
    joblib.dump(vec, OUT_DIR / "factcheck_vectorizer.joblib")
    joblib.dump(clf, OUT_DIR / "factcheck_model.joblib")
    print("Fact-check model trained.")


if __name__ == "__main__":
    main()
