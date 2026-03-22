import re

from backend.core.decision import is_knowledge_query, needs_search
from backend.core.router import route_intent
from backend.services.emotion import detect_tone
from backend.core.context_builder import build_context
from backend.core.planner import build_plan
from backend.tools.toolchain import run_tools
from backend.core.responder import generate_search_answer, generate_chat_answer, generate_academic_answer, generate_reasoning_answer, generate_creative_answer
from backend.agents.autonomous_agent import run_autonomous_agent
from backend.tools.tool_responder import handle_tool_request
from backend.services.ner import extract_entities
from backend.services.fact_check import fact_check_claim
from backend.core import config


def cognitive_process(
    user_input,
    user_id,
    tone,
    memory,
    vector_store,
    search_index,
    profile_context,
    fallback_fn,
    academic_template,
    citation_style,
    strict_sources,
    academic_instructions,
    force_web=False,
    allow_web=True,
    agent_mode=False,
    allow_execute=False,
):
    emotion = detect_tone(user_input)
    forced_academic = _force_academic(user_input, academic_instructions)
    if forced_academic:
        tone = "academic"
    tool_reply = handle_tool_request(user_input, allow_execute=allow_execute)
    if tool_reply:
        return tool_reply, {"emotion": emotion, "tool": True, "forced_academic": forced_academic}
    knowledge_query = is_knowledge_query(user_input)
    force_web = force_web and allow_web

    route = route_intent(user_input, tone=tone, academic_instructions=academic_instructions)
    intent = route.get("intent") or ("academic" if tone == "academic" else "general")
    search_needed = route.get("search_needed")
    if search_needed is None:
        search_needed = needs_search(
            user_input,
            tone=tone,
            force_web=force_web,
            knowledge_query=knowledge_query,
        )

    # Hard override: if this looks like a knowledge query, force knowledge routing
    if knowledge_query and intent not in ["reasoning", "creative", "coding"]:
        intent = "knowledge"
        search_needed = True

    if intent in ["reasoning", "creative", "coding"]:
        search_needed = False
    steps = build_plan(user_input, intent=intent, search_needed=search_needed)

    if agent_mode and config.AGENT_ENABLED and search_needed:
        response, meta = run_autonomous_agent(
            user_input,
            user_id,
            search_index,
            vector_store,
            fallback_fn,
        )
        meta["emotion"] = emotion
        return response, meta

    semantic_memories, local_results, web_results = run_tools(
        steps,
        user_input,
        search_index,
        vector_store,
        user_id,
        web_top_k=config.WEB_SEARCH_TOP_K,
        local_top_k=config.HYBRID_TOP_K,
        allow_web=allow_web,
    )
    entities = extract_entities(user_input)

    combined = local_results[: config.HYBRID_TOP_K] + web_results[: config.HYBRID_TOP_K]
    if force_web and web_results and not any(_is_web_source(s) for s in combined):
        combined = combined[:-1] + [web_results[0]]

    if search_needed and combined:
        relevance = _source_relevance(user_input, combined)
        if relevance < config.MIN_SOURCE_RELEVANCE and not force_web:
            response = _low_relevance_reply(tone=tone, allow_web=allow_web)
            return response, {
                "emotion": emotion,
                "web_results": web_results,
                "local_results": local_results,
                "forced_academic": forced_academic,
                "low_relevance": True,
                "relevance": relevance,
            }
        if tone == "academic":
            response, used_sources, citations, bibtex, endnote = generate_academic_answer(
                user_input,
                combined,
                citation_style=citation_style,
                strict_sources=strict_sources,
                template=academic_template,
                instructions=academic_instructions,
            )
        return response, {
            "emotion": emotion,
            "used_sources": used_sources,
            "citations": citations,
            "bibtex": bibtex,
            "endnote": endnote,
            "web_results": web_results,
            "local_results": local_results,
            "forced_academic": forced_academic,
            "relevance": relevance,
        }

        response = generate_search_answer(user_input, combined, fallback_fn=fallback_fn, tone=tone)
        if _needs_fact_check(user_input):
            check = fact_check_claim(user_input, combined, allow_web=allow_web)
            response = _append_fact_check(response, check)
        return response, {
            "emotion": emotion,
            "web_results": web_results,
            "local_results": local_results,
            "forced_academic": forced_academic,
            "relevance": relevance,
        }

    if intent == "reasoning":
        response = generate_reasoning_answer(user_input, fallback_fn=fallback_fn)
        return response, {"emotion": emotion, "forced_academic": forced_academic}
    if intent == "creative":
        response = generate_creative_answer(user_input, fallback_fn=fallback_fn)
        return response, {"emotion": emotion, "forced_academic": forced_academic}

    context = build_context(
        user_input,
        memory.build_context(),
        semantic_memories,
        web_results,
        local_results,
        profile_context=profile_context,
        entities=entities,
    )
    response = generate_chat_answer(user_input, context, web_context=_web_context(web_results), fallback_fn=fallback_fn)
    if _needs_fact_check(user_input):
        check = fact_check_claim(user_input, combined, allow_web=allow_web)
        response = _append_fact_check(response, check)
    return response, {
        "emotion": emotion,
        "web_results": web_results,
        "local_results": local_results,
        "forced_academic": forced_academic,
    }


def _is_web_source(source):
    return source.get("source") in ["google", "bing", "duckduckgo", "wikipedia"]


def _web_context(results):
    return " ".join([f"{r.get('title', '')}: {r.get('snippet', '')}" for r in results])


def _force_academic(user_input, academic_instructions):
    lower = (user_input or "").lower()
    if academic_instructions:
        return True
    return "reference to the various sources" in lower or ("business finance" in lower and "explain" in lower)


def _needs_fact_check(user_input):
    if config.FACT_CHECK_ALWAYS:
        return True
    lower = (user_input or "").lower()
    return any(k in lower for k in ["fact check", "verify", "is it true", "confirm", "evidence"])


def _append_fact_check(response, check):
    verdict = check.get("verdict", "unknown")
    confidence = check.get("confidence", 0.0)
    return f"{response}\nFact-check: {verdict} (confidence {confidence:.2f})"


def _source_relevance(query, sources):
    if not sources:
        return 0.0
    terms = [
        t
        for t in re.findall(r"[a-z0-9]+", (query or "").lower())
        if t
        and t
        not in {
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "to",
            "and",
            "in",
            "on",
            "for",
            "with",
            "about",
        }
    ]
    if not terms:
        return 0.0
    best = 0.0
    for src in sources:
        hay = f"{src.get('title','')} {src.get('snippet','')}".lower()
        score = sum(1 for t in terms if t in hay) / max(len(terms), 1)
        best = max(best, score)
    return best


def _low_relevance_reply(tone="default", allow_web=True):
    if tone == "academic":
        if allow_web:
            return "I couldn't find reliable sources in my index for that yet. Try rephrasing or enable web search."
        return "I couldn't find reliable sources in my index for that yet."
    if allow_web:
        return "I couldn't find relevant sources for that yet. Try rephrasing or enable web search."
    return "I couldn't find relevant sources for that yet. Try rephrasing."
