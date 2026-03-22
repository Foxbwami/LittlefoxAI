from backend.core import config

_mod = None
_mod_failed = False
_clf = None
_vec = None


def _get_moderator():
    global _mod, _mod_failed
    if _mod_failed or not config.SAFETY_USE_MODEL:
        return None
    if _mod is None:
        try:
            from transformers import pipeline
            model_name = getattr(config, "SAFETY_MODEL_NAME", "unitary/toxic-bert")
            _mod = pipeline("text-classification", model=model_name)
        except Exception:
            _mod_failed = True
            return None
    return _mod


def check_safety(text):
    text = (text or "").strip()
    if not text:
        return {"allowed": True, "labels": [], "score": 0.0, "reason": "empty"}

    clf, vec = _load_safety_classifier()
    if clf is not None and vec is not None:
        try:
            X = vec.transform([text])
            score = float(clf.predict_proba(X)[0][1])
            allowed = score < config.SAFETY_BLOCK_THRESHOLD
            return {
                "allowed": allowed,
                "labels": ["unsafe" if not allowed else "safe"],
                "score": score,
                "reason": "trained",
            }
        except Exception:
            pass

    mod = _get_moderator()
    if mod is not None:
        try:
            result = mod(text[:1000])
            if isinstance(result, list) and result:
                label = result[0].get("label", "")
                score = float(result[0].get("score", 0))
                allowed = score < config.SAFETY_BLOCK_THRESHOLD
                return {
                    "allowed": allowed,
                    "labels": [label],
                    "score": score,
                    "reason": "model",
                }
        except Exception:
            pass

    # fallback keyword filter
    lowered = text.lower()
    blocked = [w for w in config.SAFETY_BLOCKLIST if w in lowered]
    if blocked:
        return {
            "allowed": False,
            "labels": blocked[:3],
            "score": 1.0,
            "reason": "keyword",
        }
    return {"allowed": True, "labels": [], "score": 0.0, "reason": "keyword"}


def _load_safety_classifier():
    global _clf, _vec
    if _clf is not None and _vec is not None:
        return _clf, _vec
    try:
        import joblib
        import os
        base = os.path.join(config.BASE_DIR, "models")
        vec_path = os.path.join(base, "safety_vectorizer.joblib")
        clf_path = os.path.join(base, "safety_model.joblib")
        if os.path.exists(vec_path) and os.path.exists(clf_path):
            _vec = joblib.load(vec_path)
            _clf = joblib.load(clf_path)
            return _clf, _vec
    except Exception:
        _clf, _vec = None, None
    return _clf, _vec
