import csv
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core import config


def clean_text(text):
    if not text:
        return ""
    # Normalize whitespace and remove control chars
    text = text.replace("\u0000", " ")
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    # Collapse repeated characters (e.g., "loooool" -> "loool")
    text = re.sub(r"(.)\1{5,}", r"\1\1\1", text)
    # Split letter/number boundaries
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    # Split camel-like concatenations
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Normalize punctuation spacing
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", text)
    # Remove non-ASCII noise but keep basic punctuation
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_usable(text, min_words=3):
    if not text:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(len(text), 1) < 0.5:
        return False
    return True


def _find_source():
    candidates = [
        os.path.join(config.DATA_DIR, "raw", "dataset.csv"),
        config.RAW_DATA_PATH,
        os.path.join(config.DATA_DIR, "raw", "data.txt"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        try:
            reader = csv.DictReader(f)
            if reader.fieldnames and "prompt" in reader.fieldnames and "response" in reader.fieldnames:
                for row in reader:
                    yield row.get("prompt", ""), row.get("response", "")
                return
        except Exception:
            pass

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                yield row[0], row[1]


def main():
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    source = _find_source()
    if source is None:
        raise FileNotFoundError("No data source found for preprocessing.")

    lines = []
    seen = set()
    if source.lower().endswith(".csv"):
        for prompt, response in _csv_rows(source):
            prompt = clean_text(prompt)
            response = clean_text(response)
            if not _looks_usable(prompt) or not _looks_usable(response, min_words=4):
                continue
            key = (prompt, response)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"<|user|> {prompt} <|assistant|> {response} <|end|>")
    else:
        with open(source, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = clean_text(raw)
        if cleaned:
            lines.append(f"<|user|> Continue. <|assistant|> {cleaned} <|end|>")

    with open(config.PROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Wrote {len(lines)} training lines to {config.PROCESSED_DATA_PATH}")
    print(f"Source: {source}")


if __name__ == "__main__":
    main()
