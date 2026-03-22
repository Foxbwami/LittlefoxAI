from backend.tools.browser import search_web


def _simple_rerank(query, results):
    if not results:
        return results
    terms = [t for t in query.lower().split() if t]
    scored = []
    for r in results:
        hay = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
        score = sum(1 for t in terms if t in hay)
        if r.get("source") in ["google", "bing", "duckduckgo", "wikipedia"]:
            score += 0.5
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]


def _verify_results(results):
    seen = set()
    verified = []
    for r in results:
        url = r.get("url") or ""
        if not r.get("snippet"):
            continue
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        verified.append(r)
    return verified


def run_tools(steps, user_input, search_index, vector_store, user_id, web_top_k, local_top_k, allow_web=True):
    local_results = []
    web_results = []
    memories = []
    combined = []

    for step in steps:
        if step.name == "retrieve_memory":
            memories = vector_store.search(user_id, user_input, k=5)
        elif step.name == "search_local":
            local_results = search_index.search(user_input, top_k=local_top_k)
        elif step.name == "search_web" and allow_web:
            web_results = search_web(user_input, top_k=web_top_k)
        elif step.name == "rerank":
            combined = _simple_rerank(user_input, local_results + web_results)
            local_results = [r for r in combined if r.get("source") == "local"]
            web_results = [r for r in combined if r.get("source") != "local"]
        elif step.name == "verify":
            combined = _verify_results(local_results + web_results)
            local_results = [r for r in combined if r.get("source") == "local"]
            web_results = [r for r in combined if r.get("source") != "local"]

    return memories, local_results, web_results
