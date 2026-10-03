from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import CriticAgent, WriterAgent
from .models import EpisodePlan, StoryState


class EpisodeGraphState(TypedDict, total=False):
    state: StoryState
    plan: EpisodePlan
    text: str
    quality: dict[str, Any]
    attempts: int


def build_episode_graph(
    writer: WriterAgent,
    critic: CriticAgent,
    max_attempts: int,
    on_critic: Callable[[StoryState, int, dict[str, Any]], None] | None = None,
    checkpointer: Any | None = None,
):
    """Build the bounded writer -> critic loop used for each episode."""

    def draft(data: EpisodeGraphState) -> dict[str, Any]:
        attempt = data.get("attempts", 0) + 1
        return {
            "text": writer.run(data["state"], data["plan"]),
            "attempts": attempt,
        }

    def judge(data: EpisodeGraphState) -> dict[str, Any]:
        quality = critic.run(
            data["text"],
            data["plan"],
            [episode.text for episode in data["state"].episodes],
        )
        if on_critic is not None:
            on_critic(data["state"], data["attempts"], quality)
        return {"quality": quality}

    def choose_next(data: EpisodeGraphState) -> str:
        if data["quality"]["approved_by_critic"]:
            return END
        if data["attempts"] >= max_attempts:
            raise ValueError(f"Critic rejected episode {data['plan'].number}: {data['quality']}")
        return "draft"

    graph = StateGraph(EpisodeGraphState)
    graph.add_node("draft", draft)
    graph.add_node("critic", judge)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "critic")
    graph.add_conditional_edges("critic", choose_next, {"draft": "draft", END: END})
    return graph.compile(checkpointer=checkpointer)
