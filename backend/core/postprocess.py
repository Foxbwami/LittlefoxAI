import re
from backend.core import config
from backend.core.quality import compress_repetition
from backend.services.pii import redact_pii

_summarizer = None


def _load_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            from transformers import pipeline
            _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            _summarizer = None
    return _summarizer


def clean_text(text):
    # Remove multiple spaces and normalize
    text = re.sub(r"\s+", " ", text).strip()
    # Remove broken token sequences (letters+digits+letters), but keep pure numbers intact
    text = re.sub(r"([A-Za-z]+)(\d+)([A-Za-z]+)", r"\1 \3", text)  # Fix "loans2grants" -> "loans grants"
    # Insert spaces between letters and numbers
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    # Insert space after punctuation when missing
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", text)
    # Split camel-like concatenations
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text


def _sentences(text):
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        return nltk.sent_tokenize(text)
    except Exception:
        return [text]


def summarize(text):
    if not config.POSTPROCESS_SUMMARIZE:
        return text
    summarizer = _load_summarizer()
    if summarizer is None:
        return text
    try:
        result = summarizer(text, max_length=120, min_length=40, do_sample=False)
        return result[0]["summary_text"]
    except Exception:
        return text


def format_response(text):
    if _looks_structured(text):
        return _format_structured(text)
    text = clean_text(text)
    text = compress_repetition(text)
    text = text.replace("<unk>", "").strip()
    lowered = text.lower()
    for marker in ["ai:", "assistant:", "answer:"]:
        if marker in lowered:
            idx = lowered.rfind(marker)
            text = text[idx + len(marker):].strip()
            lowered = text.lower()

    prefix = config.PROMPT_PREFIX.lower()
    if lowered.startswith(prefix):
        text = text[len(prefix):].strip()

    for junk in ["user name", "personality", "context:", "question:"]:
        text = text.replace(junk, "")
    for marker in ["user:", "assistant:", "ai:"]:
        text = text.replace(marker, "")
    # drop tokens that look like prompt echoes
    parts = []
    last = None
    for tok in text.split():
        low = tok.lower()
        if low.startswith("user") or low.startswith("assistant") or low.startswith("ai"):
            continue
        if tok == last:
            continue
        parts.append(tok)
        last = tok
    text = " ".join(parts)
    sentences = _sentences(text)
    joined = " ".join(sentences)
    if getattr(config, "ENABLE_PII_REDACTION", True):
        joined = redact_pii(joined)
    return joined[: config.POSTPROCESS_MAX_CHARS]


def _looks_structured(text):
    if not text:
        return False
    lowered = text.lower()
    return "\n" in text or "references:" in lowered or "key points:" in lowered


def _format_structured(text):
    if not text:
        return text
    text = text.replace("<unk>", "").strip()
    lines = []
    for line in text.splitlines():
        line = clean_text(line)
        if not line:
            continue
        lines.append(line)
    return "\n".join(lines)[: config.POSTPROCESS_MAX_CHARS]
