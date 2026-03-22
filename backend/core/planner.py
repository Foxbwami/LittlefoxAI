from dataclasses import dataclass
from backend.core import config


@dataclass
class PlanStep:
    name: str
    detail: str = ""


def build_plan(user_input, intent="general", search_needed=False):
    steps = []
    if intent == "academic":
        steps.append(PlanStep("retrieve_memory", "Pull user-specific context."))
        steps.append(PlanStep("search_local", "Use indexed sources."))
        if search_needed:
            steps.append(PlanStep("search_web", "Fetch authoritative web sources."))
        steps.append(PlanStep("rerank", "Re-rank sources for relevance."))
        steps.append(PlanStep("verify", "Validate sources and remove weak ones."))
        steps.append(PlanStep("synthesize", "Compose academic response with citations."))
    elif intent == "knowledge":
        steps.append(PlanStep("retrieve_memory", "Find relevant memory snippets."))
        if search_needed:
            steps.append(PlanStep("search_web", "Get current facts."))
        steps.append(PlanStep("search_local", "Use local indexed data."))
        steps.append(PlanStep("rerank", "Re-rank sources for relevance."))
        steps.append(PlanStep("verify", "Validate sources and remove weak ones."))
        steps.append(PlanStep("synthesize", "Merge sources into concise answer."))
    else:
        steps.append(PlanStep("retrieve_memory", "Pull short-term context."))
        if search_needed:
            steps.append(PlanStep("search_web", "Pull quick factual context."))
        steps.append(PlanStep("respond", "Generate natural response."))

    if config.PLANNER_MAX_STEPS and len(steps) > config.PLANNER_MAX_STEPS:
        return steps[: config.PLANNER_MAX_STEPS]
    return steps
