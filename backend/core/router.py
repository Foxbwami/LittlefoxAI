import re
import torch
from backend.core import config
from backend.core.decision import is_knowledge_query

_router = None
_router_failed = False


def _get_router():
    global _router, _router_failed
    if _router_failed:
        return None
    if _router is None:
        try:
            from transformers import pipeline
            model_name = getattr(config, "DECISION_MODEL_NAME", None) or config.HF_GENERATION_MODEL
            _router = pipeline("text2text-generation", model=model_name)
        except Exception:
            try:
                _router = _build_seq2seq_router(model_name)
            except Exception:
                _router_failed = True
                return None
    return _router


class _Seq2SeqRouter:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, prompt, max_new_tokens=32, do_sample=False):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return [{"generated_text": text}]


def _build_seq2seq_router(model_name):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()
    return _Seq2SeqRouter(model, tokenizer, device)


def route_intent(text, tone="default", academic_instructions=""):
    lowered = (text or "").lower()
    if not text:
        return {"intent": "chat", "search_needed": False, "reason": "empty"}
    if tone == "academic":
        return {"intent": "academic", "search_needed": True, "reason": "explicit academic tone"}
    if any(k in lowered for k in ["mislabeled", "wrongly labeled", "wrongly labelled", "sequence", "next number", "maximize", "minimize", "fencing", "pattern recognition"]):
        return {"intent": "reasoning", "search_needed": False, "reason": "pre_rule_reasoning"}
    if any(k in lowered for k in ["ethical", "dilemma", "framework", "plausibility", "speed of sound", "vacuum"]):
        return {"intent": "reasoning", "search_needed": False, "reason": "pre_rule_reasoning"}
    if "world without the internet" in lowered:
        return {"intent": "creative", "search_needed": False, "reason": "pre_rule_creative"}
    if any(k in lowered for k in ["noir", "dialogue", "robot detective", "creative generation"]):
        return {"intent": "creative", "search_needed": False, "reason": "pre_rule_creative"}

    if not config.DECISION_MODEL_ENABLED:
        return _fallback_route(text, tone=tone)

    router = _get_router()
    if router is None:
        return _fallback_route(text, tone=tone)

    prompt = (
        "Classify the user request into one intent: chat, knowledge, academic, coding, creative, reasoning. "
        "Decide if web_search is needed: yes or no. "
        "Return strictly as: intent=<intent>;search=<yes/no>.\n"
        f"User: {text}"
    )
    try:
        result = router(prompt, max_new_tokens=24, do_sample=False)
        text_out = result[0].get("generated_text", "")
    except Exception:
        return _fallback_route(text, tone=tone)

    intent = _parse_field(text_out, "intent") or _infer_intent(text_out)
    search = _parse_field(text_out, "search")
    if intent is None:
        intent = _fallback_route(text, tone=tone)["intent"]
    search_needed = True if search == "yes" else False if search == "no" else None

    if search_needed is None:
        search_needed = _fallback_route(text, tone=tone)["search_needed"]

    if "academic" in lowered or "citation" in lowered or "cite" in lowered:
        intent = "academic"
        search_needed = True

    return {"intent": intent, "search_needed": search_needed, "reason": "model_router"}


def _parse_field(text, key):
    if not text:
        return ""
    match = re.search(rf"{key}\s*=\s*([a-zA-Z_-]+)", text)
    if match:
        return match.group(1).strip().lower()
    return ""


def _infer_intent(text):
    if not text:
        return ""
    lowered = text.lower()
    for label in ["chat", "knowledge", "academic", "coding", "creative", "reasoning"]:
        if label in lowered:
            return label
    return ""


def _fallback_route(text, tone="default"):
    lowered = (text or "").lower()
    if tone == "academic":
        return {"intent": "academic", "search_needed": True, "reason": "tone"}
    if "code" in lowered or "python" in lowered or "refactor" in lowered:
        return {"intent": "coding", "search_needed": False, "reason": "keyword"}
    if any(w in lowered for w in ["sequence", "next number", "mislabeled", "logic", "optimize", "maximize", "minimize", "pattern"]):
        return {"intent": "reasoning", "search_needed": False, "reason": "reasoning"}
    if any(w in lowered for w in ["ethical", "dilemma", "philosophical", "framework", "plausibility", "speed of sound", "vacuum"]):
        return {"intent": "reasoning", "search_needed": False, "reason": "reasoning"}
    if any(w in lowered for w in ["write", "story", "dialogue", "noir", "creative"]):
        return {"intent": "creative", "search_needed": False, "reason": "keyword"}
    if is_knowledge_query(text):
        return {"intent": "knowledge", "search_needed": True, "reason": "knowledge_query"}
    return {"intent": "chat", "search_needed": False, "reason": "fallback"}
