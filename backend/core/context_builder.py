from backend.core import config


def _limit_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def _summarize_sources(sources, max_items):
    snippets = []
    for src in sources[:max_items]:
        snippet = src.get("snippet", "")
        if snippet:
            snippets.append(snippet)
    return " ".join(snippets)


def build_context(user_input, memory_context, semantic_memories, web_results, local_results, profile_context="", entities=None):
    semantic_text = " ".join(
        [f"{m.get('role', 'User')}: {m.get('message', '')}" for m in semantic_memories]
    )
    entity_text = ""
    if entities:
        labels = []
        for ent in entities[:6]:
            labels.append(f"{ent.get('text')}({ent.get('label')})")
        entity_text = "Entities: " + ", ".join(labels)
    local_text = _summarize_sources(local_results, config.SEARCH_CONTEXT_TOP_K)
    web_text = _summarize_sources(web_results, config.SEARCH_CONTEXT_TOP_K)

    parts = []
    if profile_context:
        parts.append(profile_context)
    if semantic_text:
        parts.append(f"Memory: {semantic_text}")
    if entity_text:
        parts.append(entity_text)
    if memory_context:
        parts.append(f"Recent: {memory_context}")
    if local_text:
        parts.append(f"Local data: {local_text}")
    if web_text:
        parts.append(f"Web data: {web_text}")

    context = " ".join(parts).strip()
    context = _limit_words(context, config.PROMPT_MAX_WORDS)
    if context:
        return f"{context}\nUser: {user_input}"
    return f"User: {user_input}"
