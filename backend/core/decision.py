from backend.core import config


def is_knowledge_query(text):
    if not text:
        return False
    lowered = text.lower()
    if any(phrase in lowered for phrase in ["next number", "sequence", "mislabeled", "three boxes", "maximize", "minimize", "fencing", "rectangular pen", "logic puzzle", "pattern recognition"]):
        return False
    if any(phrase in lowered for phrase in ["write a noir", "write a story", "dialogue between", "creative generation"]):
        return False
    tokens = lowered.split()
    small_talk = [
        "how are you",
        "how's it going",
        "whats up",
        "what's up",
        "hello",
        "hi",
        "hey",
        "what did i just say",
        "what did i say",
        "thank you",
        "thanks",
    ]
    if any(phrase in lowered for phrase in small_talk):
        return False
    if len(tokens) <= 3 and any(tok in ["hi", "hey", "hello", "yo"] for tok in tokens):
        return False
    if "?" in text:
        return True
    starters = ("what", "who", "when", "where", "why", "how", "define", "explain", "compare")
    if lowered.startswith(starters):
        return True
    for phrase in ["what is", "who is", "how to", "how do", "define", "explain", "difference between", "compare"]:
        if phrase in lowered:
            return True
    if any(word in lowered for word in ["history of", "meaning of", "examples of", "benefits of", "risks of"]):
        return True
    if any(word in lowered for word in ["i feel", "i am", "my day", "my life"]) and "?" not in text:
        return False
    return False


def needs_search(text, tone="default", force_web=False, knowledge_query=None):
    if not config.BROWSE_ON_QUESTION:
        force_web = False
    lowered = (text or "").lower()
    knowledge_query = is_knowledge_query(text) if knowledge_query is None else knowledge_query
    triggers = [
        "latest",
        "news",
        "current",
        "today",
        "yesterday",
        "this week",
        "this month",
        "this year",
        "breaking",
        "search",
        "find",
        "source",
        "citation",
        "who is",
        "what is",
        "where is",
        "when is",
        "how to",
        "define",
        "explain",
        "compare",
        "research",
        "paper",
        "study",
        "evidence",
        "statistics",
    ]
    if force_web or config.COGNITIVE_FORCE_WEB:
        return knowledge_query or tone == "academic" or any(t in lowered for t in triggers)
    if tone == "academic":
        return True
    if knowledge_query:
        return True
    return any(t in lowered for t in triggers)
