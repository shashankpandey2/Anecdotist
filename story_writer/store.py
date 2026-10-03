from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .models import StoryState, state_from_dict, to_dict


class StoryStore:
    """Persists one story as readable JSON and append-only trace events."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(
            self.root / "checkpoints.sqlite3",
            check_same_thread=False,
        )
        self.checkpointer = SqliteSaver(
            self._checkpoint_connection,
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("story_writer.models", "ArcPlan"),
                    ("story_writer.models", "Canon"),
                    ("story_writer.models", "Episode"),
                    ("story_writer.models", "EpisodePlan"),
                    ("story_writer.models", "StoryState"),
                ]
            ),
        )
        self.checkpointer.setup()

    def close(self) -> None:
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None

    def __del__(self) -> None:
        self.close()

    def state_path(self, story_id: str) -> Path:
        return self.root / f"{story_id}.json"

    def trace_path(self, story_id: str) -> Path:
        return self.root / f"{story_id}.trace.jsonl"

    def save(self, state: StoryState) -> None:
        path = self.state_path(state.story_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(to_dict(state), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def exists(self, story_id: str) -> bool:
        return self.state_path(story_id).exists()

    def load(self, story_id: str) -> StoryState:
        return state_from_dict(json.loads(self.state_path(story_id).read_text(encoding="utf-8")))

    def trace(self, story_id: str, event: str, **details: object) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **details,
        }
        with self.trace_path(story_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
