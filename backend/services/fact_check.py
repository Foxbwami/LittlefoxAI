import re
from backend.core import config
from backend.tools.browser import search_web

_clf = None
_vec = None


def fact_check_claim(claim, sources=None, allow_web=True):
    claim = (claim or "").strip()
    if not claim:
        return {"verdict": "unknown", "confidence": 0.0, "evidence": []}

    evidence = []
    if sources:
        evidence.extend([s for s in sources if s.get("snippet")])

    if allow_web and config.BROWSE_ON_QUESTION:
        web = search_web(claim, top_k=3)
        evidence.extend(web)

    verdict, confidence = _simple_verdict(claim, evidence)
    if verdict == "unknown":
        model = _load_factcheck_classifier()
        if model is not None:
            clf, vec = model
            try:
                X = vec.transform([claim])
                pred = clf.predict(X)[0]
                proba = max(clf.predict_proba(X)[0])
                return {
                    "verdict": pred,
                    "confidence": float(proba),
                    "evidence": evidence[:3],
                }
            except Exception:
                pass
    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence[:3],
    }


def _simple_verdict(claim, evidence):
    if not evidence:
        return "unknown", 0.0
    claim_lower = claim.lower()
    negations = ["not", "never", "no", "false", "incorrect"]
    negated = any(n in claim_lower for n in negations)
    hits = 0
    for e in evidence:
        snippet = (e.get("snippet") or "").lower()
        if not snippet:
            continue
        if _overlap(claim_lower, snippet) > 0.3:
            hits += 1
            if any(n in snippet for n in negations) and not negated:
                return "contradicted", 0.55
    if hits:
        return "supported", min(0.5 + 0.1 * hits, 0.8)
    return "unknown", 0.2


def _overlap(a, b):
    a_terms = set(re.findall(r"[a-z0-9]+", a))
    b_terms = set(re.findall(r"[a-z0-9]+", b))
    if not a_terms:
        return 0.0
    return len(a_terms & b_terms) / max(len(a_terms), 1)


def _load_factcheck_classifier():
    global _clf, _vec
    if _clf is not None and _vec is not None:
        return _clf, _vec
    try:
        import joblib
        import os
        base = os.path.join(config.BASE_DIR, "models")
        vec_path = os.path.join(base, "factcheck_vectorizer.joblib")
        clf_path = os.path.join(base, "factcheck_model.joblib")
        if os.path.exists(vec_path) and os.path.exists(clf_path):
            _vec = joblib.load(vec_path)
            _clf = joblib.load(clf_path)
            return _clf, _vec
    except Exception:
        _clf, _vec = None, None
    return None
