from sentence_transformers import SentenceTransformer
from backend.core import config
import numpy as np
import hashlib

_model = None
_failed = False


def _get_model():
    global _model, _failed
    if _failed:
        return None
    if getattr(config, "EMBEDDINGS_DISABLED", False):
        _failed = True
        return None
    if _model is None:
        try:
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
        except Exception:
            if config.ALLOW_EMBEDDINGS_FALLBACK:
                _failed = True
                return None
            raise
    return _model


def embed(texts):
    model = _get_model()
    if model is None:
        if getattr(config, "ALLOW_HASH_EMBEDDINGS", False):
            return _hash_embed(texts)
        return None
    return model.encode(texts)


def embed_one(text):
    vectors = embed([text])
    if vectors is None:
        return None
    return vectors[0] if len(vectors) > 0 else None


def get_embedding(text):
    vec = embed_one(text)
    if vec is None:
        return None
    return vec.tolist() if hasattr(vec, "tolist") else vec


def _hash_embed(texts):
    dim = config.EMBEDDING_DIM
    vectors = []
    for text in texts:
        vec = np.zeros(dim, dtype="float32")
        for token in (text or "").lower().split():
            h = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vectors.append(vec)
    return vectors
