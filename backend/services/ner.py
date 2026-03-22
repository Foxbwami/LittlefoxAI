import re
from backend.core import config

_nlp = None
_nlp_failed = False
_gazetteer = None


def _get_nlp():
    global _nlp, _nlp_failed
    if _nlp_failed or not config.NER_USE_SPACY:
        return None
    if _nlp is None:
        try:
            import spacy
            model = getattr(config, "NER_SPACY_MODEL", "en_core_web_sm")
            _nlp = spacy.load(model)
        except Exception:
            _nlp_failed = True
            return None
    return _nlp


def extract_entities(text):
    text = text or ""
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return _dedupe_entities(ents)
    gaz = _load_gazetteer()
    if gaz:
        ents = []
        for label, values in gaz.items():
            for value in values:
                if value and value in text:
                    ents.append({"text": value, "label": label})
        if ents:
            return _dedupe_entities(ents)
    return _regex_entities(text)


def _load_gazetteer():
    global _gazetteer
    if _gazetteer is not None:
        return _gazetteer
    try:
        import json
        import os
        path = os.path.join(config.BASE_DIR, "models", "ner_gazetteer.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _gazetteer = json.load(f)
                return _gazetteer
    except Exception:
        _gazetteer = None
    return _gazetteer


def _regex_entities(text):
    ents = []
    # dates
    for m in re.findall(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b", text):
        ents.append({"text": m, "label": "DATE"})
    # emails
    for m in re.findall(r"[\\w\\.-]+@[\\w\\.-]+", text):
        ents.append({"text": m, "label": "EMAIL"})
    # urls
    for m in re.findall(r"https?://[^\\s]+", text):
        ents.append({"text": m, "label": "URL"})
    # simple proper nouns (capitalized sequences)
    for m in re.findall(r"\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\b", text):
        ents.append({"text": m, "label": "PROPN"})
    return _dedupe_entities(ents)


def _dedupe_entities(ents):
    seen = set()
    deduped = []
    for ent in ents:
        key = (ent["text"], ent["label"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ent)
    return deduped
