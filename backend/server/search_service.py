import os
import pickle
from backend.core import config
from backend.retrieval.search import hybrid_search
from backend.retrieval.reranker import rerank_indices

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


class SearchIndex:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.pages = None
        self.tfidf = None
        self.matrix = None
        self.embeddings = None
        self.pagerank = None
        self._bm25 = None
        self._bm25_docs = None

    def load(self):
        try:
            with open(os.path.join(self.index_dir, "pages.pkl"), "rb") as f:
                self.pages = pickle.load(f)
            with open(os.path.join(self.index_dir, "tfidf.pkl"), "rb") as f:
                self.tfidf = pickle.load(f)
            with open(os.path.join(self.index_dir, "matrix.pkl"), "rb") as f:
                self.matrix = pickle.load(f)
            with open(os.path.join(self.index_dir, "embeddings.pkl"), "rb") as f:
                self.embeddings = pickle.load(f)
            with open(os.path.join(self.index_dir, "pagerank.pkl"), "rb") as f:
                self.pagerank = pickle.load(f)
        except Exception:
            return False
        self._build_bm25()
        return True

    def _build_bm25(self):
        if self.pages is None or BM25Okapi is None:
            self._bm25 = None
            self._bm25_docs = None
            return
        docs = []
        for p in self.pages:
            text = (p.get("content") or "").lower()
            docs.append(text.split())
        if docs:
            self._bm25_docs = docs
            self._bm25 = BM25Okapi(docs)

    def search(self, query, top_k=5):
        if self.pages is None:
            if not self.load():
                return []
        base_k = min(max(config.RERANK_CANDIDATES, top_k), len(self.pages))
        idxs = list(hybrid_search(query, self.tfidf, self.matrix, self.embeddings, self.pagerank, top_k=base_k))
        scored = self._rerank(query, idxs, top_k=top_k)
        results = []
        for score, i in scored:
            p = self.pages[int(i)]
            snippet = p["content"][:300]
            snippet = (
                snippet.replace("â€™", "'")
                .replace("â€”", " - ")
                .replace("—", " - ")
            )
            snippet = snippet.replace("[", "").replace("]", "")
            results.append(
                {
                    "title": p["url"].split("/")[-1].replace("_", " ") or p["url"],
                    "url": p["url"],
                    "snippet": snippet,
                    "source": "local",
                    "score": float(score),
                }
            )
        return results

    def _rerank(self, query, idxs, top_k=5):
        if idxs is None or len(idxs) == 0:
            return []
        return rerank_indices(
            query,
            idxs,
            pages=self.pages,
            tfidf=self.tfidf,
            matrix=self.matrix,
            embeddings=self.embeddings,
            pagerank=self.pagerank,
            bm25=self._bm25,
            bm25_docs=self._bm25_docs,
            top_k=top_k,
        )
