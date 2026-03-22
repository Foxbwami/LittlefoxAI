import argparse
import json
import os
import pickle
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.core import config
from backend.retrieval.embeddings import embed


def _read_text(path, max_chars=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read(max_chars)
        return data
    except Exception:
        return ""


def _load_jsonl(path):
    docs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                content = obj.get("content") or obj.get("text") or ""
                url = obj.get("url") or obj.get("id") or "local://jsonl"
                if content:
                    docs.append({"url": url, "content": content})
    except Exception:
        return []
    return docs


def _chunk_text(text, chunk_size=800, overlap=200, min_chunk=200):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        window = text[start:end]
        last_dot = window.rfind(".")
        if last_dot > 200:
            end = start + last_dot + 1
            window = text[start:end]
        if len(window) >= min_chunk:
            chunks.append(window)
        if end >= len(text):
            break
        start = max(end - overlap, 0)
        if start == end:
            break
    return chunks


def _collect_docs(input_path, max_docs=None, max_chars_per_file=200000):
    docs = []

    # Local curated pages
    local_pages = os.path.join(config.DATA_DIR, "local_pages.jsonl")
    if os.path.exists(local_pages):
        docs.extend(_load_jsonl(local_pages))

    # Processed / raw text
    if os.path.exists(config.PROCESSED_DATA_PATH):
        text = _read_text(config.PROCESSED_DATA_PATH, max_chars_per_file)
        if text:
            docs.append({"url": "local://processed-data", "content": text})

    if os.path.exists(config.RAW_DATA_PATH):
        text = _read_text(config.RAW_DATA_PATH, max_chars_per_file)
        if text:
            docs.append({"url": "local://raw-data", "content": text})

    if input_path and os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for name in files:
                if max_docs and len(docs) >= max_docs:
                    return docs
                path = os.path.join(root, name)
                lower = name.lower()
                if lower.endswith(".jsonl"):
                    docs.extend(_load_jsonl(path))
                elif lower.endswith((".txt", ".md", ".csv")):
                    content = _read_text(path, max_chars_per_file)
                    if content:
                        docs.append({"url": f"local://{name}", "content": content})
    return docs[:max_docs] if max_docs else docs


def main():
    parser = argparse.ArgumentParser(description="Build local search index for LittlefoxAI.")
    parser.add_argument("--input", default=os.path.join(config.DATA_DIR, "raw"))
    parser.add_argument("--out", default=config.SEARCH_INDEX_DIR)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--min-chunk", type=int, default=200)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--max-chars", type=int, default=200000)
    args = parser.parse_args()

    docs = _collect_docs(args.input, max_docs=args.max_docs, max_chars_per_file=args.max_chars)
    if not docs:
        raise RuntimeError("No documents found to index.")

    pages = []
    for doc in docs:
        base = doc.get("url") or "local://doc"
        chunks = _chunk_text(
            doc.get("content", ""),
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            min_chunk=args.min_chunk,
        )
        for i, chunk in enumerate(chunks):
            pages.append({"url": f"{base}#chunk-{i}", "content": chunk})

    texts = [p["content"] for p in pages]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    matrix = vectorizer.fit_transform(texts)

    embeddings = embed(texts)
    if embeddings is None:
        embeddings = None
    else:
        embeddings = np.array(embeddings, dtype="float32")

    pagerank = np.ones(len(pages), dtype="float32")
    if len(pages) > 0:
        pagerank = pagerank / len(pages)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pages.pkl"), "wb") as f:
        pickle.dump(pages, f)
    with open(os.path.join(args.out, "tfidf.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(args.out, "matrix.pkl"), "wb") as f:
        pickle.dump(matrix, f)
    with open(os.path.join(args.out, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)
    with open(os.path.join(args.out, "pagerank.pkl"), "wb") as f:
        pickle.dump(pagerank, f)

    print(f"Indexed {len(pages)} chunks from {len(docs)} docs.")
    print(f"Output written to {args.out}")


if __name__ == "__main__":
    main()
