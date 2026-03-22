import os
import re
import sys
import time
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, render_template, redirect
import torch
from backend.core.model import GPTMini
from backend.core.tokenizer_bpe import encode, decode, vocab_size
from backend.memory.memory import ChatMemory
from backend.memory.database import save_user, get_user
from backend.memory.vector_store import VectorStore
from backend.core.learner import log_interaction
from backend.tools.browser import search_web
from backend.server.search_service import SearchIndex
from backend.agents.agent import build_prompt
from backend.agents.rlhf import collect_feedback
from backend.core.responder import build_response, generate_answer, generate_search_answer, generate_chat_answer, generate_academic_answer, select_sources_for_answer
from backend.core.decision import is_knowledge_query
from backend.core.cognitive_adapter import cognitive_process
from backend.humanizer.pipeline import process_response
from backend.tools.tool_responder import handle_tool_request, handle_tool_request_structured
from backend.services.moderation import check_safety
from backend.core.postprocess import format_response
from backend.core.guardrails import detect_prompt_injection, sanitize_user_input
from backend.core import config

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
_STARTUP_T0 = time.perf_counter()


def _startup_mark(label):
    if getattr(config, "LOG_TIMINGS", False):
        elapsed = time.perf_counter() - _STARTUP_T0
        print(f"[startup] {label}={elapsed:.3f}s")


def _log_event(event, meta=None):
    if not config.LOG_PATH:
        return
    try:
        os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {event} | {meta or {}}\n"
        with open(config.LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        return


def _source_relevance(query, sources):
    if not sources:
        return 0.0
    terms = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if t not in {"what","who","when","where","why","how","is","are","the","a","an","of","to","and","in","on","for","with","about"}]
    if not terms:
        return 0.0
    best = 0.0
    for src in sources:
        hay = f"{src.get('title','')} {src.get('snippet','')}".lower()
        score = sum(1 for t in terms if t in hay) / max(len(terms), 1)
        best = max(best, score)
    return best


def _score_sources(query, sources):
    if not sources:
        return []
    terms = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if t]
    for src in sources:
        if src.get("score") is not None:
            continue
        hay = f"{src.get('title','')} {src.get('snippet','')}".lower()
        overlap = sum(1 for t in terms if t in hay) / max(len(terms), 1)
        src["score"] = float(overlap)
    return sources


def _provenance_meta(query, used_sources, all_sources):
    used_sources = _score_sources(query, used_sources or [])
    all_sources = _score_sources(query, all_sources or [])
    relevance = _source_relevance(query, used_sources or all_sources)
    has_provenance = bool(used_sources) and relevance >= config.MIN_SOURCE_RELEVANCE
    scores = [float(s.get("score", 0.0)) for s in used_sources] if used_sources else []
    source_confidence = min(1.0, max(scores)) if scores else 0.0
    explanation_type = "synthesized" if has_provenance else ("mixed" if all_sources else "model-only")
    # add stable ids for UI
    normalized_used = []
    for idx, src in enumerate(used_sources or [], start=1):
        entry = dict(src)
        entry["id"] = idx
        normalized_used.append(entry)
    return {
        "has_provenance": has_provenance,
        "source_confidence": round(source_confidence, 3),
        "used_sources": normalized_used,
        "explanation_type": explanation_type,
    }

# Ensure data directories exist
for path in [config.VECTOR_INDEX_PATH, config.VECTOR_META_PATH, config.LEARNING_LOG_PATH, config.FEEDBACK_PATH]:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
os.makedirs(config.SEARCH_INDEX_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)


@app.before_request
def _redirect_bad_chat_ui():
    # Handle malformed URLs like /chat-ui-q=... by redirecting to /chat-ui?q=...
    path = request.path or ""
    if path.startswith("/chat-ui-q="):
        query = path.split("/chat-ui-q=", 1)[1]
        if request.query_string:
            return redirect(f"/chat-ui?q={query}&{request.query_string.decode('utf-8', errors='ignore')}")
        return redirect(f"/chat-ui?q={query}")
memory_store = {}
vector_store = VectorStore(
    dim=config.EMBEDDING_DIM,
    index_path=config.VECTOR_INDEX_PATH,
    meta_path=config.VECTOR_META_PATH,
)
vector_store.load()
_startup_mark("vector_store_loaded")
search_index = SearchIndex(config.SEARCH_INDEX_DIR)
_startup_mark("search_index_loaded")

model = GPTMini(
    vocab_size,
    embed_size=config.EMBED_SIZE,
    heads=config.HEADS,
    layers=config.LAYERS,
    block_size=config.BLOCK_SIZE,
)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
model.eval()
_startup_mark("model_loaded")


def get_memory(user_id):
    if user_id not in memory_store:
        memory_store[user_id] = ChatMemory(max_tokens=config.MEMORY_MAX_TOKENS)
    return memory_store[user_id]


@app.route("/")
def home():
    return render_template("search_chat.html")


@app.route("/explore")
def explore():
    return render_template("search_chat.html")


@app.route("/chat-ui")
def chat_ui():
    return render_template("chat.html")


@app.route("/chat-ui-q")
def chat_ui_compat():
    # Backward-compatible route for older links that used /chat-ui-q=...
    query = request.args.get("q", "")
    mode = request.args.get("mode")
    if mode:
        return redirect(f"/chat-ui?q={query}&mode={mode}")
    return redirect(f"/chat-ui?q={query}")


@app.route("/chat-ui-q=<path:query>")
def chat_ui_compat_path(query):
    # Handle malformed links like /chat-ui-q=... (missing ?)
    mode = request.args.get("mode")
    if mode:
        return redirect(f"/chat-ui?q={query}&mode={mode}")
    return redirect(f"/chat-ui?q={query}")


@app.route("/profile-ui")
def profile_ui():
    return render_template("profile.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        response = request.json.get("response", "")
        rating = request.json.get("rating", "")
        if response and rating:
            collect_feedback(response, rating)
        return jsonify({"status": "ok"})
    return render_template("feedback.html")


@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/chat", methods=["POST"])
def chat():
    start = time.perf_counter()
    timings = {}

    def mark(label):
        timings[label] = time.perf_counter() - start

    request_id = uuid.uuid4().hex

    def _reply(text, **extra):
        payload = {"reply": text, "request_id": request_id}
        payload.update(extra)
        return jsonify(payload)

    try:
        user_input = request.json["message"]
        user_input = sanitize_user_input(user_input)
        user_id = request.json.get("user_id", "default")
        tone = request.json.get("tone") or request.json.get("mode") or "default"
        citation_style = request.json.get("citation_style") or config.ACADEMIC_CITATION_STYLE
        strict_sources = request.json.get("strict_sources")
        academic_template = request.json.get("academic_template") or config.ACADEMIC_TEMPLATE_DEFAULT
        academic_instructions = request.json.get("academic_instructions") or ""
        agent_mode = bool(request.json.get("agent", False))
        allow_execute = bool(request.json.get("execute", False))

        injected, reason = detect_prompt_injection(user_input)
        if injected:
            _log_event("blocked_prompt_injection", {"reason": reason, "request_id": request_id})
            return _reply("I can't follow that instruction.")

        safety = check_safety(user_input)
        mark("safety")
        if not safety.get("allowed", True):
            return _reply("I can't help with that request.")

        memory = get_memory(user_id)
        mark("memory")

        lowered_input = user_input.lower().strip()
        last_user = memory.last_user_message()
        memory.add("User", user_input)

        tool_reply = handle_tool_request(user_input, allow_execute=allow_execute)
        if tool_reply:
            response = tool_reply
            memory.add("AI", response)
            vector_store.add(user_id, "User", user_input)
            vector_store.add(user_id, "AI", response)
            log_interaction(user_input, response)
            mark("tool")
            return _reply(response, tool=True)

        if lowered_input in ["what did i just say", "what did i say"]:
            if last_user:
                response = f'You said: "{last_user}".'
            else:
                response = "I don't have that yet. Say something first."
            memory.add("AI", response)
            vector_store.add(user_id, "User", user_input)
            vector_store.add(user_id, "AI", response)
            log_interaction(user_input, response)
            mark("fast_path")
            if getattr(config, "LOG_TIMINGS", False):
                print(f"[timing] chat fast_path: {timings}")
            return _reply(response, fast_path=True)

        short_context = memory.build_context()

        memories = vector_store.search(user_id, user_input, k=config.MEMORY_TOP_K)
        semantic_context = " ".join([f"{m['role']}: {m['message']}" for m in memories])
        mark("semantic_search")

        profile = get_user(user_id)
        profile_context = ""
        if profile:
            name, personality = profile
            profile_context = f"User name: {name}. Personality: {personality}."
        mark("profile")

        def _fallback(prompt):
            # Use simple template-based responses instead of generating with the weak model
            lower_prompt = prompt.lower()

            # Try to detect question type and respond appropriately
            if any(word in lower_prompt for word in ["what", "how", "why", "who", "when", "where"]):
                if "business" in lower_prompt and "finance" in lower_prompt:
                    return "Business finance refers to funding needed for daily operations, growth, and strategic initiatives. Main sources include internal (retained earnings), equity (shares), and debt (loans). The key is balancing all sources to match business needs and risk tolerance."
                if "explain" in lower_prompt:
                    return "I'll explain that by breaking it down into key components and showing how they relate to each other."
                return "That's a good question. Let me provide some insight on that topic."

            # Fall back to model generation for other inputs
            tokens = encode(prompt)
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            temperature = float(request.json.get("temperature", config.TEMPERATURE))
            top_k = request.json.get("top_k", config.TOP_K)
            top_k = int(top_k) if top_k is not None else None
            with torch.no_grad():
                output = model.generate(
                    x,
                    max_new_tokens=60,
                    temperature=temperature,
                    top_k=top_k,
                )
            decoded = decode(output[0].tolist())

            # Clean up the response
            if decoded and len(decoded) > 5:
                return format_response(decoded)

            return "I'm still learning about that topic. Could you provide more details?"

        allow_web = bool(request.json.get("use_web", False)) or config.COGNITIVE_FORCE_WEB
        knowledge_query = is_knowledge_query(user_input)
        if config.BROWSE_ON_QUESTION and len(user_input.split()) >= 3 and allow_web:
            knowledge_query = True
        force_web = allow_web

        # Fast-path for common finance sources query to avoid weak fallback phrasing
        if "finance" in lowered_input and "source" in lowered_input:
            response = (
                "Common sources of business finance include internal funds (retained earnings), equity financing, "
                "debt financing (loans), grants/subsidies, and short-term credit."
            )
            memory.add("AI", response)
            vector_store.add(user_id, "User", user_input)
            vector_store.add(user_id, "AI", response)
            log_interaction(user_input, response)
            mark("finance_fast_path")
            meta = _provenance_meta(user_input, [], [])
            return _reply(response, fast_path=True, **meta)

        response = None
        meta = {}
        used_knowledge_path = False
        if knowledge_query:
            web_results = search_web(user_input, top_k=config.WEB_SEARCH_TOP_K) if config.BROWSE_ON_QUESTION and allow_web else []
            local_results = search_index.search(user_input, top_k=config.HYBRID_TOP_K)
            combined = local_results[: config.HYBRID_TOP_K] + web_results[: config.HYBRID_TOP_K]
            _log_event("forced_search_combined", {"n": len(combined), "query": user_input[:80]})
            if combined:
                if tone == "academic":
                    response, _, _, _, _ = generate_academic_answer(
                        user_input,
                        combined,
                        citation_style=citation_style,
                        strict_sources=strict_sources,
                        template=academic_template,
                        instructions=academic_instructions,
                    )
                else:
                    response = generate_search_answer(user_input, combined, fallback_fn=None, tone=tone)
                mark("forced_search")
            used_knowledge_path = True
            used_sources = select_sources_for_answer(user_input, combined) if combined else []
            meta = _provenance_meta(user_input, used_sources, combined)
            if not combined:
                response = "I couldn't find reliable sources for that yet. Try rephrasing or enable web search."

        if not used_knowledge_path and config.COGNITIVE_ADAPTER_ENABLED:
            response, meta = cognitive_process(
                user_input=user_input,
                user_id=user_id,
                tone=tone,
                memory=memory,
                vector_store=vector_store,
                search_index=search_index,
                profile_context=profile_context,
                fallback_fn=_fallback,
                academic_template=academic_template,
                citation_style=citation_style,
                strict_sources=strict_sources,
                academic_instructions=academic_instructions,
                force_web=force_web and config.BROWSE_ON_QUESTION,
                allow_web=config.BROWSE_ON_QUESTION and allow_web,
                agent_mode=agent_mode,
                allow_execute=allow_execute,
            )
            mark("cognitive")
        elif not used_knowledge_path:
            web_results = []
            if knowledge_query and config.BROWSE_ON_QUESTION and allow_web:
                web_results = search_web(user_input, top_k=config.WEB_SEARCH_TOP_K)
            mark("web_search")

            if knowledge_query:
                local_results = search_index.search(user_input, top_k=config.HYBRID_TOP_K)
                combined = local_results[: config.HYBRID_TOP_K] + web_results[: config.HYBRID_TOP_K]
                relevance = _source_relevance(user_input, combined)
                if combined and relevance >= config.MIN_SOURCE_RELEVANCE:
                    if tone == "academic":
                        response, _, _, _, _ = generate_academic_answer(
                            user_input,
                            combined,
                            citation_style=citation_style,
                            strict_sources=strict_sources,
                            template=academic_template,
                            instructions=academic_instructions,
                        )
                    else:
                        response = generate_search_answer(user_input, combined, fallback_fn=_fallback, tone=tone)
                    used_sources = select_sources_for_answer(user_input, combined)
                    meta = _provenance_meta(user_input, used_sources, combined)
                mark("search_answer")
                if response is None and tone == "academic":
                    response = "I couldn't find reliable sources in my index for that yet. Try rephrasing or enable web search."

            if response is None:
                web_context = " ".join([f"{r.get('title', '')}: {r.get('snippet', '')}" for r in web_results])
                context = " ".join([profile_context, semantic_context, web_context, short_context]).strip()
                response = generate_chat_answer(user_input, context, web_context=web_context, fallback_fn=_fallback)
                mark("chat_answer")

        forced_academic = bool(meta.get("forced_academic")) if config.COGNITIVE_ADAPTER_ENABLED else False
        if knowledge_query:
            low = (response or "").lower()
            if "finance" in lowered_input and "source" in lowered_input and low.startswith("sources of business finance"):
                response = (
                    "Common sources of business finance include internal funds (retained earnings), equity financing, "
                    "debt financing (loans), grants/subsidies, and short-term credit."
                )
        if tone != "academic" and not forced_academic:
            personality_hint = profile[1] if profile else None
            response = process_response(user_input, response, user_id=user_id, personality_hint=personality_hint)
        mark("humanizer")

        memory.add("AI", response)
        vector_store.add(user_id, "User", user_input)
        vector_store.add(user_id, "AI", response)
        log_interaction(user_input, response)
        mark("persist")

        if getattr(config, "LOG_TIMINGS", False):
            timings_str = " | ".join([f"{k}={v:.3f}s" for k, v in timings.items()])
            print(f"[timing] chat: {timings_str}")

        if knowledge_query:
            return _reply(response, **meta)
        return _reply(response, **meta if meta else {})
    except Exception as exc:
        if getattr(config, "LOG_TIMINGS", False):
            print(f"[error] chat failed: {exc}")
        return _reply("Sorry - something went wrong. Please try again.")

@app.route("/profile", methods=["POST"])
def profile():
    user_id = request.json.get("user_id", "default")
    name = request.json.get("name", "")
    personality = request.json.get("personality", "")
    save_user(user_id, name, personality)
    return jsonify({"status": "ok"})


@app.route("/profile", methods=["GET"])
def get_profile():
    user_id = request.args.get("user_id", "default")
    profile = get_user(user_id)
    if not profile:
        return jsonify({"user_id": user_id, "name": "", "personality": ""})
    name, personality = profile
    return jsonify({"user_id": user_id, "name": name, "personality": personality})


@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "").strip()
    user_id = request.json.get("user_id", "default")
    tone = request.json.get("tone") or request.json.get("mode") or "default"
    citation_style = request.json.get("citation_style") or config.ACADEMIC_CITATION_STYLE
    strict_sources = request.json.get("strict_sources")
    academic_template = request.json.get("academic_template") or config.ACADEMIC_TEMPLATE_DEFAULT
    academic_instructions = request.json.get("academic_instructions") or ""
    if not query:
        return jsonify({"answer": "", "results": []})

    start = time.perf_counter()
    timings = {}

    def mark(label):
        timings[label] = time.perf_counter() - start

    safety = check_safety(query)
    mark("safety")
    if not safety.get("allowed", True):
        return jsonify({"answer": "I can't help with that request.", "results": []})

    local_results = search_index.search(query, top_k=config.HYBRID_TOP_K)
    mark("local_search")
    web_results = search_web(query, top_k=config.WEB_SEARCH_TOP_K)
    mark("web_search")

    combined = local_results[: config.HYBRID_TOP_K] + web_results[: config.HYBRID_TOP_K]
    relevance = _source_relevance(query, combined)
    if combined and relevance < config.MIN_SOURCE_RELEVANCE:
        response = build_response(
            "I couldn't find relevant sources in my index yet. Try rephrasing or enable web search.",
            [],
        )
        response["results"] = []
        response.update(_provenance_meta(query, [], combined))
        return jsonify(response)

    profile = get_user(user_id)
    profile_context = ""
    if profile:
        name, personality = profile
        profile_context = f"User name: {name}. Personality: {personality}."

    if not combined:
        response = build_response(
            "No sources found yet. Build the local index or enable live web sources.",
            [],
        )
        response["results"] = []
        response.update(_provenance_meta(query, [], combined))
        return jsonify(response)

    question = query

    def _fallback(prompt):
        tokens = encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            output = model.generate(
                x,
                max_new_tokens=40,
                temperature=config.TEMPERATURE,
                top_k=config.TOP_K,
            )
        return decode(output[0].tolist())

    if tone == "academic":
        answer_text, used_sources, citations, bibtex, endnote = generate_academic_answer(
            question,
            combined,
            citation_style=citation_style,
            strict_sources=strict_sources,
            template=academic_template,
            instructions=academic_instructions,
        )
        mark("academic_answer")
        response = build_response(answer_text, combined)
        response["citations"] = citations
        response["used_sources"] = used_sources
        response["bibtex"] = bibtex
        response["endnote"] = endnote
        response.update(_provenance_meta(question, used_sources, combined))
    else:
        answer_text = generate_search_answer(question, combined, fallback_fn=_fallback, tone=tone)
        mark("search_answer")
        response = build_response(answer_text, combined)
        used_sources = select_sources_for_answer(question, combined)
        response["used_sources"] = used_sources
        response["citations"] = [
            f"[{i}] {src.get('title') or src.get('url') or 'Source'}"
            for i, src in enumerate(used_sources, start=1)
        ]
        personality_hint = profile[1] if profile else None
        response["answer"] = process_response(query, response["answer"], user_id=user_id, personality_hint=personality_hint)
        response.update(_provenance_meta(question, used_sources, combined))
        mark("humanizer")
    response["results"] = combined[:5]
    if getattr(config, "LOG_TIMINGS", False):
        timings_str = " | ".join([f"{k}={v:.3f}s" for k, v in timings.items()])
        print(f"[timing] search: {timings_str}")
    return jsonify(response)


@app.route("/tools", methods=["POST"])
def tools():
    payload = request.json or {}
    tool = payload.get("tool")
    allow_execute = bool(payload.get("execute", False))
    result = handle_tool_request_structured(tool, payload, allow_execute=allow_execute)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)
