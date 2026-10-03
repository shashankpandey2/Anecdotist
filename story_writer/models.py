from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EpisodePlan:
    number: int
    title: str
    purpose: str
    hook: str
    characters: list[str] = field(default_factory=list)
    threads_advanced: list[str] = field(default_factory=list)


@dataclass
class ArcPlan:
    premise: str
    logline: str
    themes: list[str]
    characters: list[dict[str, Any]]
    phases: list[dict[str, Any]]
    episodes: list[EpisodePlan]


@dataclass
class Canon:
    facts: list[str] = field(default_factory=list)
    character_states: dict[str, str] = field(default_factory=dict)
    timeline: list[str] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)


@dataclass
class Episode:
    number: int
    title: str
    text: str
    status: str = "draft"
    feedback: list[str] = field(default_factory=list)
    revision: int = 1
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoryState:
    story_id: str
    premise: str
    arc: ArcPlan | None = None
    canon: Canon = field(default_factory=Canon)
    directives: list[str] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    next_episode: int = 1
    status: str = "planning"

    def episode(self, number: int) -> Episode | None:
        return next((item for item in self.episodes if item.number == number), None)

    def plan(self, number: int) -> EpisodePlan:
        if self.arc is None:
            raise ValueError("The story has no arc plan")
        return self.arc.episodes[number - 1]


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value


def episode_plan_from_dict(data: dict[str, Any]) -> EpisodePlan:
    return EpisodePlan(**data)


def arc_from_dict(data: dict[str, Any]) -> ArcPlan:
    return ArcPlan(
        premise=data["premise"],
        logline=data["logline"],
        themes=data["themes"],
        characters=data["characters"],
        phases=data["phases"],
        episodes=[episode_plan_from_dict(item) for item in data["episodes"]],
    )


def state_from_dict(data: dict[str, Any]) -> StoryState:
    return StoryState(
        story_id=data["story_id"],
        premise=data["premise"],
        arc=arc_from_dict(data["arc"]) if data.get("arc") else None,
        canon=Canon(**data.get("canon", {})),
        directives=data.get("directives", []),
        episodes=[Episode(**item) for item in data.get("episodes", [])],
        next_episode=data.get("next_episode", 1),
        status=data.get("status", "planning"),
    )
