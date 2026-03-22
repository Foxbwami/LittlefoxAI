import os
import time
import pickle
import faiss
import numpy as np
from backend.core import config
from backend.retrieval.embeddings import get_embedding


class VectorStore:
    def __init__(self, dim, index_path, meta_path):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.entries = []
        self._add_count = 0

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.entries = pickle.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.entries, f)

    def add(self, user_id, role, message):
        emb = get_embedding(message)
        if emb is None:
            return
        vector = np.array([emb], dtype="float32")
        self.index.add(vector)
        entry = {
            "user_id": user_id,
            "role": role,
            "message": message,
            "embedding": emb,
            "timestamp": time.time(),
            "importance": min(len(message.split()) / 30.0, 1.0),
        }
        self.entries.append(entry)
        self._add_count += 1
        if self._add_count % config.MEMORY_PRUNE_EVERY == 0:
            self.prune_user(user_id)
        save_every = getattr(config, "VECTOR_SAVE_EVERY", 20)
        if save_every and self._add_count % save_every == 0:
            self.save()

    def search(self, user_id, query, k=5):
        if self.index.ntotal == 0:
            return []
        query_emb = get_embedding(query)
        if query_emb is None:
            return []
        vector = np.array([query_emb], dtype="float32")
        oversample = max(k * 5, k)
        distances, indices = self.index.search(vector, oversample)
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.entries):
                continue
            entry = self.entries[idx]
            if entry["user_id"] != user_id:
                continue
            results.append(entry)
            if len(results) >= k:
                break
        return results

    def prune_user(self, user_id):
        user_indices = [i for i, e in enumerate(self.entries) if e["user_id"] == user_id]
        if len(user_indices) <= config.MEMORY_MAX_PER_USER:
            return

        now = time.time()
        scored = []
        for i in user_indices:
            e = self.entries[i]
            age_hours = max((now - e["timestamp"]) / 3600.0, 0.0)
            recency = 1.0 / (1.0 + age_hours)
            importance = max(e.get("importance", 0.0), config.MEMORY_MIN_IMPORTANCE)
            score = 0.6 * importance + 0.4 * recency
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = set(i for _, i in scored[: config.MEMORY_PRUNE_TARGET])
        self._rebuild(keep)

    def _rebuild(self, keep_indices):
        kept_entries = []
        vectors = []
        for i, e in enumerate(self.entries):
            if i in keep_indices:
                kept_entries.append(e)
                vectors.append(e["embedding"])
        self.entries = kept_entries
        self.index = faiss.IndexFlatL2(self.dim)
        if vectors:
            self.index.add(np.array(vectors, dtype="float32"))
