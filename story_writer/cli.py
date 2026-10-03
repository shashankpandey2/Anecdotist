from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .engine import StoryOrchestrator
from .provider import GeminiProvider, LocalProvider, OllamaProvider
from .store import StoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write and review a resumable 200-episode serial.")
    parser.add_argument("--data-dir", type=Path, default=Path(".story-data"))
    parser.add_argument("--provider", choices=("gemini", "ollama", "local"), default="gemini")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"))
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="Create a story and its 200-episode plan")
    new.add_argument("premise")
    new.add_argument("--id")
    for name in ("approve-plan", "write", "show", "plan"):
        command = commands.add_parser(name)
        command.add_argument("story_id")
    edit_arc = commands.add_parser("edit-plan", help="Replace the plan with a reviewed JSON file")
    edit_arc.add_argument("story_id")
    edit_arc.add_argument("json_file", type=Path)
    history = commands.add_parser("edit-history", help="Rewrite an episode and rebuild later state")
    history.add_argument("story_id")
    history.add_argument("number", type=int)
    history.add_argument("replacement_file", type=Path)
    feedback = commands.add_parser("feedback")
    feedback.add_argument("story_id")
    feedback.add_argument("text")
    review = commands.add_parser("review")
    review.add_argument("story_id")
    review.add_argument("number", type=int)
    review.add_argument("decision", choices=("approve", "reject", "edit"))
    review.add_argument("text", nargs="?", default="")
    return parser


def build_engine(args: argparse.Namespace) -> StoryOrchestrator:
    cost_cap = float(os.getenv("STORY_EPISODE_COST_CAP_USD", "0.02"))
    if args.provider == "ollama":
        provider = OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    elif args.provider == "local":
        provider = LocalProvider()
    else:
        provider = GeminiProvider(model=args.model)
    return StoryOrchestrator(StoryStore(args.data_dir), provider=provider, max_episode_cost_usd=cost_cap)


def execute(args: argparse.Namespace, engine: StoryOrchestrator) -> None:
    if args.command == "new":
        state = engine.create(args.premise, args.id)
        engine.create_arc(state)
        print(f"Story {state.story_id} created. Review the arc, then run approve-plan {state.story_id}.")
        return

    state = engine.store.load(args.story_id)
    if args.command == "approve-plan":
        engine.approve_arc(state)
        print("Plan approved. Run write to draft the next episode.")
    elif args.command == "plan":
        print(json.dumps(state.arc, default=lambda value: value.__dict__, indent=2))
    elif args.command == "edit-plan":
        data = json.loads(args.json_file.read_text(encoding="utf-8"))
        engine.edit_arc(state, data)
        print("Plan updated. Review it, then run approve-plan.")
    elif args.command == "write":
        episode = engine.write_next(state)
        print(f"Episode {episode.number}: {episode.title}\n\n{episode.text}")
    elif args.command == "feedback":
        engine.add_directive(state, args.text)
        print("Directive saved and will be carried into future episodes.")
    elif args.command == "review":
        review_text = args.text
        if args.decision == "edit":
            review_text = Path(args.text).read_text(encoding="utf-8").strip()
        engine.review_episode(state, args.number, args.decision, review_text)
        print(f"Episode {args.number} marked {args.decision}.")
    elif args.command == "edit-history":
        replacement = args.replacement_file.read_text(encoding="utf-8").strip()
        engine.edit_history(state, args.number, replacement)
        print(f"History edited at episode {args.number}; later episodes will be regenerated.")
    else:
        print(f"status={state.status} next_episode={state.next_episode} episodes={len(state.episodes)}")


def main() -> None:
    args = build_parser().parse_args()
    execute(args, build_engine(args))


if __name__ == "__main__":
    main()
