from backend.core import config

_clf = None
_vec = None


def detect_tone(user_input):
    text = (user_input or "").strip()
    model = _load_emotion_classifier()
    if model is not None:
        clf, vec = model
        try:
            X = vec.transform([text])
            pred = clf.predict(X)[0]
            if pred == "positive":
                return "warm"
            if pred == "negative":
                return "empathetic"
            return "neutral"
        except Exception:
            pass

    lowered = text.lower()
    if any(word in lowered for word in ["sad", "tired", "stressed", "overwhelmed", "anxious", "lonely", "upset"]):
        return "empathetic"
    if any(word in lowered for word in ["error", "bug", "issue", "problem", "fail", "help"]):
        return "helpful"
    if any(word in lowered for word in ["thanks", "thank you", "appreciate"]):
        return "warm"
    return "neutral"


def _load_emotion_classifier():
    global _clf, _vec
    if _clf is not None and _vec is not None:
        return _clf, _vec
    try:
        import joblib
        import os
        base = os.path.join(config.BASE_DIR, "models")
        vec_path = os.path.join(base, "emotion_vectorizer.joblib")
        clf_path = os.path.join(base, "emotion_model.joblib")
        if os.path.exists(vec_path) and os.path.exists(clf_path):
            _vec = joblib.load(vec_path)
            _clf = joblib.load(clf_path)
            return _clf, _vec
    except Exception:
        _clf, _vec = None, None
    return None
