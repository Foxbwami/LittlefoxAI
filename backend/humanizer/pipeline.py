from backend.services.emotion import detect_tone
from backend.humanizer.humanizer import humanize
from backend.core.enhancer import add_human_touch
from backend.core.style_model import apply_style
from backend.core.personalities import apply_personality
from backend.memory.user_profile import update_user_style
from backend.agents.rlhf import reward_score
from backend.core import config


def process_response(user_input, response, user_id=None, personality_hint=None):
    if not response:
        return response
    if not getattr(config, "HUMANIZER_ENABLED", False):
        return response
    if _looks_like_tool_response(response):
        return response
    if _is_coding_query(user_input) or _looks_like_code(response) or _is_reasoning_query(user_input):
        return response
    tone = detect_tone(user_input)
    style = update_user_style(user_id, user_input) if user_input else "friendly"

    text = humanize(response, tone=tone)
    text = apply_style(text, style)

    if personality_hint:
        text = apply_personality(text, personality_hint)

    if reward_score(text) < 0:
        text = humanize(text, tone=tone)

    text = add_human_touch(text, tone=tone)
    return text


def _is_coding_query(text):
    lower = (text or "").lower()
    keywords = ["python", "code", "function", "algorithm", "bug", "stack trace", "compile", "error", "exception"]
    return any(k in lower for k in keywords)


def _looks_like_code(text):
    if "```" in text:
        return True
    return "def " in text or "class " in text or "{" in text and "}" in text


def _looks_like_tool_response(text):
    lower = (text or "").lower()
    return (
        lower.startswith("result:") or
        lower.startswith("solution") or
        "latex:" in lower or
        "syntax ok" in lower or
        "execution error" in lower or
        "execution finished" in lower or
        "output:" in lower or
        "fact-check:" in lower
    )


def _is_reasoning_query(text):
    lower = (text or "").lower()
    keywords = [
        "mislabeled", "wrongly labeled", "wrongly labelled", "sequence", "next number", "maximize", "minimize", "fencing",
        "ethical", "dilemma", "framework", "plausibility", "speed of sound", "vacuum",
        "counterfactual", "world without the internet", "pattern recognition",
    ]
    return any(k in lower for k in keywords)
