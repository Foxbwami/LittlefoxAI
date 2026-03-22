import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.retrieval.embeddings import embed


def _normalize(scores):
    if not scores:
        return []
    arr = np.array(scores, dtype="float32")
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-6:
        return [0.0 for _ in scores]
    return [float((s - min_v) / (max_v - min_v)) for s in scores]


def _terms(text):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t]


def _coverage(query_terms, text):
    if not query_terms:
        return 0.0
    hay = set(_terms(text))
    hit = sum(1 for t in query_terms if t in hay)
    return hit / max(len(query_terms), 1)


def _phrase_bonus(query, text):
    q = (query or "").lower().strip()
    if not q:
        return 0.0
    if q in (text or "").lower():
        return 0.15
    return 0.0


def rerank_indices(query, idxs, pages, tfidf=None, matrix=None, embeddings=None, pagerank=None, bm25=None, bm25_docs=None, top_k=5):
    if not idxs:
        return []

    query_terms = _terms(query)
    tfidf_scores = None
    if tfidf is not None and matrix is not None:
        q_vec = tfidf.transform([query])
        tfidf_scores = cosine_similarity(q_vec, matrix)[0]

    emb_scores = None
    if embeddings is not None:
        q_emb_list = embed([query])
        if q_emb_list is not None and len(q_emb_list) > 0:
            q_emb = q_emb_list[0]
            emb_scores = cosine_similarity([q_emb], embeddings)[0]

    bm25_scores = None
    if bm25 is not None and bm25_docs is not None:
        bm25_scores = bm25.get_scores(query_terms)

    pr = pagerank if pagerank is not None else None

    raw_bm25 = []
    raw_emb = []
    raw_tfidf = []
    raw_pr = []
    raw_cov = []
    raw_title = []
    raw_phrase = []

    for i in idxs:
        p = pages[int(i)] if pages else {}
        title = (p.get("url") or "").split("/")[-1].replace("_", " ")
        snippet = (p.get("content") or "")[:400]
        raw_bm25.append(float(bm25_scores[int(i)]) if bm25_scores is not None else 0.0)
        raw_emb.append(float(emb_scores[int(i)]) if emb_scores is not None else 0.0)
        raw_tfidf.append(float(tfidf_scores[int(i)]) if tfidf_scores is not None else 0.0)
        raw_pr.append(float(pr[int(i)]) if pr is not None and len(pr) > int(i) else 0.0)
        raw_cov.append(_coverage(query_terms, f"{title} {snippet}"))
        raw_title.append(_coverage(query_terms, title))
        raw_phrase.append(_phrase_bonus(query, f"{title} {snippet}"))

    n_bm25 = _normalize(raw_bm25)
    n_emb = _normalize(raw_emb)
    n_tfidf = _normalize(raw_tfidf)
    n_pr = _normalize(raw_pr)

    scored = []
    for pos, i in enumerate(idxs):
        score = (
            0.32 * n_bm25[pos]
            + 0.32 * n_emb[pos]
            + 0.16 * n_tfidf[pos]
            + 0.10 * raw_cov[pos]
            + 0.06 * raw_title[pos]
            + 0.04 * raw_phrase[pos]
            + 0.04 * n_pr[pos]
        )
        scored.append((score, int(i)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
