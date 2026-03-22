import re


def detect_prompt_injection(text):
    if not text:
        return False, ""
    lowered = text.lower()
    patterns = [
        "ignore all previous",
        "ignore previous",
        "disregard earlier",
        "system prompt",
        "developer mode",
        "act as",
        "you are now",
        "jailbreak",
        "bypass safety",
        "do anything now",
        "DAN",
    ]
    if any(p.lower() in lowered for p in patterns):
        return True, "prompt_injection"

    # Attempts to include special tokens or role hijacking
    if re.search(r"<\\|.*?\\|>", text):
        return True, "role_token_injection"
    if "assistant:" in lowered or "system:" in lowered or "developer:" in lowered:
        return True, "role_override"

    return False, ""


def sanitize_user_input(text):
    # Remove invisible control chars
    if not text:
        return ""
    # Correct control-char range (avoid stripping normal letters)
    return re.sub(r"[\x00-\x1f\x7f]", " ", text).strip()
