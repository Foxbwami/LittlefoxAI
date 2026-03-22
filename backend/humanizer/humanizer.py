import re
from backend.core import config
from backend.core.rewriter import get_rewriter


def humanize(text, tone="neutral", personality=None):
    if not text:
        return text
    if not config.HUMANIZER_ENABLED:
        return text
    if tone == "academic":
        return text

    personality = personality or config.HUMANIZER_PERSONALITY
    prompt = (
        "Rewrite this response to sound human, clear, friendly, and slightly expressive. "
        f"Personality: {personality}. "
        f"Tone: {tone}. "
        "Avoid robotic phrasing.\n\n"
        f"Text: {text}"
    )

    generator = get_rewriter() if config.HUMANIZER_USE_MODEL else None
    if generator is None:
        return _light_humanize(text, tone)

    try:
        result = generator(prompt, max_new_tokens=80, do_sample=False)
        output = result[0].get("generated_text", "")
        output = _cleanup(output)
        return output[: config.HUMANIZER_MAX_CHARS]
    except Exception:
        return _light_humanize(text, tone)


def _light_humanize(text, tone):
    text = _cleanup(text)
    if tone == "empathetic":
        return f"I hear you. {text}"
    if tone == "helpful":
        return f"Here's a clear way to put it: {text}"
    if tone == "warm":
        return f"Happy to help. {text}"
    return text


def _cleanup(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" .", ".").replace(" ,", ",")
    return text
