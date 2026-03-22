import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.retrieval.embeddings import embed


def hybrid_search(query, tfidf, matrix, embeddings, pagerank, top_k=5):
    if matrix is None or tfidf is None:
        return []

    q_vec = tfidf.transform([query])
    tfidf_scores = cosine_similarity(q_vec, matrix)[0]

    emb_scores = np.zeros_like(tfidf_scores)
    if embeddings is not None:
        q_emb_list = embed([query])
        if q_emb_list is not None and len(q_emb_list) > 0:
            q_emb = q_emb_list[0]
            emb_scores = cosine_similarity([q_emb], embeddings)[0]

    pr = pagerank if pagerank is not None and len(pagerank) == len(tfidf_scores) else np.zeros_like(tfidf_scores)

    final_score = tfidf_scores * 0.4 + emb_scores * 0.4 + pr * 0.2
    ranked = np.argsort(final_score)[::-1]
    return ranked[:top_k]
