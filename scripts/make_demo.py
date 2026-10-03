import json
from pathlib import Path
import shutil

from story_writer.engine import StoryOrchestrator
from story_writer.models import to_dict
from story_writer.store import StoryStore


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
DATA = DEMO / "state"


def main() -> None:
    if DEMO.exists():
        shutil.rmtree(DEMO)
    DATA.mkdir(parents=True)

    engine = StoryOrchestrator(StoryStore(DATA))
    state = engine.create(
        "A talented young batter from a neighborhood club earns a place at a state cricket academy and must rebuild her game after a serious injury.",
        story_id="demo",
    )
    engine.create_arc(state)
    (DEMO / "arc.json").write_text(
        json.dumps(to_dict(state.arc), indent=2) + "\n", encoding="utf-8"
    )
    engine.approve_arc(state)

    episodes_dir = DEMO / "episodes"
    episodes_dir.mkdir()
    for number in range(1, 16):
        episode = engine.write_next(state)
        (episodes_dir / f"episode-{episode.number:03d}.txt").write_text(episode.text + "\n", encoding="utf-8")
        if number == 3:
            engine.add_directive(state, "Let Mira and Kabir earn their partnership through cricket, not instant trust.")
        if number == 7:
            engine.review_episode(state, number, "reject", "The recovery is too easy. Let Mira sit with the setback before the comeback.")
            episode = engine.write_next(state)
            (episodes_dir / f"episode-{episode.number:03d}.txt").write_text(episode.text + "\n", encoding="utf-8")
        engine.review_episode(state, episode.number, "approve")

    (DEMO / "state.json").write_text(
        json.dumps(to_dict(state), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {len(state.episodes)} approved episodes in {DEMO}")


if __name__ == "__main__":
    main()
