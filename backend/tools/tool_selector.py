from backend.core import config
from backend.core.router import route_intent


def select_tool(user_input):
    route = route_intent(user_input)
    intent = route.get("intent", "chat")
    if any(k in (user_input or "").lower() for k in ["calculate", "compute", "evaluate"]):
        return "compute"
    if any(k in (user_input or "").lower() for k in ["solve", "equation"]):
        return "solve"
    if any(k in (user_input or "").lower() for k in ["validate syntax", "check syntax"]):
        return "validate_syntax"
    if any(k in (user_input or "").lower() for k in ["run", "execute"]):
        return "execute"
    if intent == "knowledge":
        return "search"
    if intent == "academic":
        return "search"
    return "chat"
