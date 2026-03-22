def build_prompt(question, search_results):
    context = " ".join([r.get("snippet", "")[:200] for r in search_results])
    return (
        "Use the provided sources to answer. If sources are weak or missing, say so.\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer (concise, factual):"
    )
