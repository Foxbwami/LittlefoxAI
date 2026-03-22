_user_styles = {}


def update_user_style(user_id, message):
    text = (message or "").lower()
    if any(word in text for word in ["bro", "yo", "lmao", "lit", "vibe", "fr"]):
        style = "genz"
    elif any(word in text for word in ["please", "kindly", "thank you", "regards"]):
        style = "formal"
    elif any(word in text for word in ["strict", "serious", "critical"]):
        style = "strict"
    elif any(word in text for word in ["mentor", "coach", "guide"]):
        style = "mentor"
    else:
        style = "friendly"
    if user_id:
        _user_styles[user_id] = style
    return style


def get_user_style(user_id):
    return _user_styles.get(user_id)
