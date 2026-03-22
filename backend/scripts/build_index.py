import os
import pickle
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from backend.crawler.scheduler import crawl_distributed
from backend.retrieval.pagerank import compute_pagerank
from backend.retrieval.tfidf import build_tfidf
from backend.retrieval.embeddings import embed
from backend.core import config


def load_seeds():
    seed_path = os.path.join(config.DATA_DIR, "seeds.txt")
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Neural_network",
        "https://en.wikipedia.org/wiki/Information_retrieval",
        "https://en.wikipedia.org/wiki/Search_engine",
    ]


def main():
    os.makedirs(config.SEARCH_INDEX_DIR, exist_ok=True)

    local_path = os.path.join(config.DATA_DIR, "local_pages.jsonl")
    pages = []
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json
                    pages.append(json.loads(line))
                except Exception:
                    continue
    else:
        seeds = load_seeds()[: config.CRAWL_MAX_PAGES]
        pages = crawl_distributed(seeds)
    if not pages:
        print("No pages crawled.")
        return

    texts = [p["content"] for p in pages]
    pagerank = compute_pagerank(pages)
    tfidf, matrix = build_tfidf(texts)
    embeddings = embed(texts)

    with open(os.path.join(config.SEARCH_INDEX_DIR, "pages.pkl"), "wb") as f:
        pickle.dump(pages, f)
    with open(os.path.join(config.SEARCH_INDEX_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(config.SEARCH_INDEX_DIR, "matrix.pkl"), "wb") as f:
        pickle.dump(matrix, f)
    with open(os.path.join(config.SEARCH_INDEX_DIR, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)
    with open(os.path.join(config.SEARCH_INDEX_DIR, "pagerank.pkl"), "wb") as f:
        pickle.dump(pagerank, f)

    print(f"Indexed {len(pages)} pages")


if __name__ == "__main__":
    main()
