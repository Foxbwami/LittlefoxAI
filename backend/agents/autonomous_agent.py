from backend.core.planner import build_plan
from backend.tools.toolchain import run_tools
from backend.core.responder import generate_search_answer
from backend.core import config


def run_autonomous_agent(user_input, user_id, search_index, vector_store, fallback_fn):
    plan = build_plan(user_input, intent="knowledge", search_needed=True)
    memories, local_results, web_results = run_tools(
        plan,
        user_input,
        search_index,
        vector_store,
        user_id,
        web_top_k=config.WEB_SEARCH_TOP_K,
        local_top_k=config.HYBRID_TOP_K,
    )
    combined = local_results[: config.HYBRID_TOP_K] + web_results[: config.HYBRID_TOP_K]
    response = generate_search_answer(user_input, combined, fallback_fn=fallback_fn)
    meta = {
        "plan": [step.name for step in plan],
        "memories": memories,
        "local_results": local_results,
        "web_results": web_results,
    }
    return response, meta

