from __future__ import annotations

import time
import uuid
from typing import Any

from .agents import CriticAgent, PlannerAgent, WriterAgent
from .models import Canon, Episode, StoryState, arc_from_dict
from .provider import LocalProvider, StoryProvider
from .store import StoryStore
from .workflow import build_episode_graph


class StoryOrchestrator:
    """Coordinates the planner, writer, critic, and human review checkpoints."""

    def __init__(
        self,
        store: StoryStore,
        provider: StoryProvider | None = None,
        max_writer_attempts: int = 2,
        max_episode_cost_usd: float = 0.02,
    ):
        provider = provider or LocalProvider()
        self.store = store
        self.planner = PlannerAgent(provider)
        self.writer = WriterAgent(provider)
        self.critic = CriticAgent()
        self.max_writer_attempts = max_writer_attempts
        self.max_episode_cost_usd = max_episode_cost_usd
        self.episode_graph = build_episode_graph(
            self.writer,
            self.critic,
            max_writer_attempts,
            on_critic=self._trace_critic,
            checkpointer=self.store.checkpointer,
        )

    def create(self, premise: str, story_id: str | None = None) -> StoryState:
        if not premise.strip():
            raise ValueError("A premise is required")
        state = StoryState(story_id=story_id or uuid.uuid4().hex[:8], premise=premise.strip())
        self.store.save(state)
        self.store.trace(state.story_id, "story_created", premise=state.premise)
        return state

    def create_arc(self, state: StoryState) -> StoryState:
        started = time.perf_counter()
        self._trace_agent(state, self.planner.name, "started")
        arc = arc_from_dict(self.planner.run(state.premise))
        self._validate_arc(arc)
        state.arc = arc
        state.status = "awaiting_arc_approval"
        self.store.save(state)
        self._trace_agent(
            state,
            self.planner.name,
            "completed",
            episodes=len(arc.episodes),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        return state

    def edit_arc(self, state: StoryState, arc_data: dict[str, Any]) -> StoryState:
        arc = arc_from_dict(arc_data)
        self._validate_arc(arc)
        state.arc = arc
        state.status = "awaiting_arc_approval"
        self.store.save(state)
        self.store.trace(state.story_id, "arc_edited", episodes=len(arc.episodes))
        return state

    def approve_arc(self, state: StoryState) -> StoryState:
        if state.arc is None:
            raise ValueError("Create the arc before approving it")
        self._validate_arc(state.arc)
        state.status = "writing"
        self.store.save(state)
        self.store.trace(state.story_id, "arc_approved")
        return state

    def add_directive(self, state: StoryState, feedback: str) -> StoryState:
        feedback = feedback.strip()
        if not feedback:
            raise ValueError("Feedback cannot be empty")
        state.directives.append(feedback)
        self.store.save(state)
        self.store.trace(state.story_id, "directive_added", feedback=feedback)
        return state

    def write_next(self, state: StoryState) -> Episode:
        if state.arc is None or state.status != "writing":
            raise ValueError("Approve the arc and resolve the current review before writing")
        if state.next_episode > 200:
            raise ValueError("The 200-episode plan is complete")

        plan = state.plan(state.next_episode)
        self._trace_agent(state, self.writer.name, "started", episode=plan.number)
        started = time.perf_counter()
        result = self.episode_graph.invoke(
            {"state": state, "plan": plan},
            config={
                "configurable": {
                    "thread_id": f"{state.story_id}:episode:{plan.number}",
                }
            },
        )
        text = result["text"]
        last_quality = result["quality"]
        attempts = result.get("attempts", 1)
        estimated_tokens = round(last_quality["word_count"] * 1.3)
        estimated_cost = round(estimated_tokens * 0.00001 * attempts, 6)
        if estimated_cost > self.max_episode_cost_usd:
            self.store.trace(
                state.story_id,
                "episode_budget_exceeded",
                episode=plan.number,
                attempts=attempts,
                estimated_cost_usd=estimated_cost,
                budget_usd=self.max_episode_cost_usd,
            )
            raise ValueError(
                f"Episode {plan.number} exceeded the estimated cost cap of "
                f"${self.max_episode_cost_usd:.4f}"
            )

        episode = Episode(number=plan.number, title=plan.title, text=text, quality=last_quality)
        state.episodes.append(episode)
        state.next_episode += 1
        state.status = "awaiting_episode_review"
        self.store.save(state)
        self._trace_agent(
            state,
            self.writer.name,
            "completed",
            episode=episode.number,
            latency_ms=round((time.perf_counter() - started) * 1000),
            word_count=last_quality["word_count"],
            attempts=attempts,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
            budget_usd=self.max_episode_cost_usd,
        )
        self.store.trace(state.story_id, "episode_drafted", episode=episode.number, quality=last_quality)
        return episode

    def review_episode(self, state: StoryState, number: int, decision: str, feedback: str = "") -> StoryState:
        episode = state.episode(number)
        if episode is None:
            raise ValueError(f"Episode {number} does not exist")
        if decision not in {"approve", "reject", "edit"}:
            raise ValueError("Decision must be approve, reject, or edit")

        if decision == "reject":
            if feedback:
                episode.feedback.append(feedback)
                self.add_directive(state, feedback)
            state.episodes = [item for item in state.episodes if item.number != number]
            state.next_episode = number
        elif decision == "edit":
            if not feedback:
                raise ValueError("An edit needs replacement text")
            quality = self.critic.run(
                feedback,
                state.plan(number),
                [item.text for item in state.episodes if item.number != number],
            )
            if not quality["approved_by_critic"]:
                raise ValueError("Edited episode did not pass the critic")
            episode.text = feedback
            episode.quality = quality
            episode.revision += 1
            episode.status = "approved"
            self._rebuild_canon(state)
        else:
            episode.status = "approved"
            self._rebuild_canon(state)

        state.status = "writing"
        self.store.save(state)
        self.store.trace(state.story_id, "episode_reviewed", episode=number, decision=decision, feedback=feedback)
        return state

    def edit_history(self, state: StoryState, number: int, replacement: str) -> StoryState:
        episode = state.episode(number)
        if episode is None:
            raise ValueError(f"Episode {number} does not exist")
        quality = self.critic.run(
            replacement,
            state.plan(number),
            [item.text for item in state.episodes if item.number != number],
        )
        if not quality["approved_by_critic"]:
            raise ValueError("Edited episode did not pass the critic")
        episode.text = replacement
        episode.quality = quality
        episode.revision += 1
        episode.status = "approved"
        state.episodes = [item for item in state.episodes if item.number <= number]
        state.next_episode = number + 1
        state.status = "writing"
        self._rebuild_canon(state)
        self.store.save(state)
        self.store.trace(state.story_id, "history_edited", episode=number, rebuilt_from=number + 1)
        return state

    def _rebuild_canon(self, state: StoryState) -> None:
        state.canon = Canon()
        for episode in state.episodes:
            if episode.status == "approved":
                self._update_canon(state, episode)

    @staticmethod
    def _update_canon(state: StoryState, episode: Episode) -> None:
        plan = state.plan(episode.number)
        state.canon.facts.append(f"Episode {episode.number} completed: {plan.purpose}")
        state.canon.timeline.append(f"Episode {episode.number}: {plan.title}")
        for thread in plan.threads_advanced:
            if thread not in state.canon.open_threads:
                state.canon.open_threads.append(thread)
        for name in plan.characters:
            state.canon.character_states[name] = f"Present in episode {episode.number}"

    def _trace_agent(self, state: StoryState, agent: str, event: str, **details: Any) -> None:
        self.store.trace(state.story_id, "agent", agent=agent, phase=event, **details)

    def _trace_critic(self, state: StoryState, attempt: int, quality: dict[str, Any]) -> None:
        self._trace_agent(
            state,
            self.critic.name,
            "completed" if quality["approved_by_critic"] else "rejected",
            episode=state.next_episode,
            attempt=attempt,
            quality=quality,
        )

    @staticmethod
    def _validate_arc(arc: Any) -> None:
        numbers = [episode.number for episode in arc.episodes]
        if len(numbers) != 200 or numbers != list(range(1, 201)):
            raise ValueError("The arc must contain exactly episodes 1 through 200")
