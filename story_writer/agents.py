from __future__ import annotations

import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Protocol
from typing import Any

from .models import EpisodePlan, StoryState
from .provider import StoryProvider, build_context


class Agent(Protocol):
    name: str


@dataclass
class PlannerAgent:
    provider: StoryProvider
    name: str = "planner"

    def run(self, premise: str) -> dict[str, Any]:
        return self.provider.plan(premise)


@dataclass
class WriterAgent:
    provider: StoryProvider
    name: str = "writer"

    def run(self, state: StoryState, plan: EpisodePlan) -> str:
        return self.provider.write_episode(state, plan, build_context(state, plan))


class CriticAgent:
    name: str = "critic"
    minimum_words: int = 400
    maximum_words: int = 700
    repetition_threshold: float = 0.985

    def run(
        self,
        text: str,
        plan: EpisodePlan | None = None,
        previous_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        words = text.split()
        sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
        has_hook = bool(
            re.search(
                r"(tomorrow|future|message|door|constellation|chart|archive|voice|scorebook|selection|innings|comeback|trial|match|season)[^.!?]*[.!?]$",
                text,
                re.IGNORECASE,
            )
        )
        if plan and plan.hook.strip().lower() in text.lower():
            has_hook = True
        normalized = " ".join(text.lower().split())
        repetition_score = max(
            (
                SequenceMatcher(None, normalized, " ".join(previous.lower().split())).ratio()
                for previous in (previous_texts or [])
            ),
            default=0.0,
        )
        repeated_episode = repetition_score >= self.repetition_threshold
        required_characters = plan.characters if plan else []
        missing_characters = [
            name for name in required_characters if name.lower() not in text.lower()
        ]
        return {
            "word_count": len(words),
            "within_target": self.minimum_words <= len(words) <= self.maximum_words,
            "has_hook": has_hook,
            "repetition_score": round(repetition_score, 3),
            "repeated_episode": repeated_episode,
            "missing_characters": missing_characters,
            "consistency_pass": not missing_characters,
            "sentence_count": len(sentences),
            "repeated_plan_terms": sum(
                text.lower().count(term.lower()) for term in (plan.threads_advanced if plan else [])
            ),
            "approved_by_critic": (
                self.minimum_words <= len(words) <= self.maximum_words
                and has_hook
                and not repeated_episode
                and not missing_characters
            ),
        }
