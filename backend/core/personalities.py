from backend.core import config
from backend.core.rewriter import get_rewriter

PERSONALITIES = {
    "mentor": "Confident, calm, wise, slightly witty, supportive.",
    "friendly": "Warm, casual, easy-going, encouraging.",
    "formal": "Professional, structured, serious.",
    "genz": "Casual, fun, expressive, playful.",
    "strict": "Direct, concise, no-nonsense, critical but fair.",
    "funny": "Light, witty, playful.",
}


def apply_personality(text, personality):
    if not text or not personality:
        return text
    style = PERSONALITIES.get(personality, personality)
    prompt = (
        "Respond in this personality:\n"
        f"{style}\n\n"
        f"Text: {text}"
    )
    generator = get_rewriter() if config.HUMANIZER_USE_MODEL else None
    if generator is None:
        return text
    try:
        result = generator(prompt, max_new_tokens=80, do_sample=False)
        output = result[0].get("generated_text", "")
        return output[: config.HUMANIZER_MAX_CHARS]
    except Exception:
        return text
