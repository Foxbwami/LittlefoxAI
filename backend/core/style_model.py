from backend.core import config
from backend.core.rewriter import get_rewriter


def apply_style(text, style):
    if not text or not style:
        return text
    if style in ["default", "friendly"]:
        return text
    prompt = (
        f"Rewrite this in a {style} tone. Keep it concise and natural.\n\n"
        f"Text: {text}"
    )
    generator = get_rewriter()
    if generator is None:
        return text
    try:
        result = generator(prompt, max_new_tokens=60, do_sample=False)
        output = result[0].get("generated_text", "")
        return output[: config.HUMANIZER_MAX_CHARS]
    except Exception:
        return text
