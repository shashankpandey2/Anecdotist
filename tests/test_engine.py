from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path

from story_writer.engine import StoryOrchestrator
from story_writer.store import StoryStore


class StoryEngineTests(unittest.TestCase):
    @contextmanager
    def story(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = StoryOrchestrator(StoryStore(Path(directory)))
            state = engine.create("A young batter earns a place at a cricket academy after a difficult trial.")
            engine.create_arc(state)
            engine.approve_arc(state)
            yield engine, state

    def test_writes_target_length_and_persists_trace(self):
        with self.story() as (engine, state):
            episode = engine.write_next(state)
            self.assertTrue(400 <= len(episode.text.split()) <= 700)
            self.assertTrue(episode.quality["has_hook"])
            self.assertTrue(engine.store.trace_path(state.story_id).exists())
            trace = engine.store.trace_path(state.story_id).read_text(encoding="utf-8")
            self.assertIn('"agent": "writer"', trace)
            self.assertIn('"agent": "critic"', trace)

            checkpoints = list(
                engine.store.checkpointer.list(
                    {"configurable": {"thread_id": f"{state.story_id}:episode:1"}}
                )
            )
            self.assertGreaterEqual(len(checkpoints), 2)

    def test_rejection_regenerates_same_episode(self):
        with self.story() as (engine, state):
            engine.write_next(state)
            engine.review_episode(state, 1, "reject", "Slow down the friendship.")
            self.assertEqual(state.next_episode, 1)
            self.assertIsNone(state.episode(1))
            rewritten = engine.write_next(state)
            self.assertEqual(rewritten.number, 1)
            self.assertIn("Slow down the friendship.", state.directives)
            self.assertIn("not to rush the next innings", rewritten.text)

    def test_retroactive_edit_invalidates_future_episodes(self):
        with self.story() as (engine, state):
            engine.write_next(state)
            engine.review_episode(state, 1, "approve")
            engine.write_next(state)
            replacement = state.episode(1).text.replace(
                "The scorebook recorded one more decision:",
                "A revised line. The scorebook recorded one more decision:",
                1,
            )
            engine.edit_history(state, 1, replacement)
            self.assertEqual(state.next_episode, 2)
            self.assertIsNone(state.episode(2))
            self.assertEqual(state.episode(1).revision, 2)

    def test_arc_has_200_sequential_plans(self):
        with self.story() as (_, state):
            self.assertEqual(len(state.arc.episodes), 200)
            self.assertEqual([item.number for item in state.arc.episodes], list(range(1, 201)))

    def test_episode_budget_is_enforced_before_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StoryStore(Path(directory))
            engine = StoryOrchestrator(store, max_episode_cost_usd=0.000001)
            state = engine.create("A young batter earns a place at a cricket academy.")
            engine.create_arc(state)
            engine.approve_arc(state)

            with self.assertRaises(ValueError, msg="episode budget should stop an oversized estimate"):
                engine.write_next(state)

            self.assertEqual(state.next_episode, 1)
            self.assertIsNone(state.episode(1))
            trace = store.trace_path(state.story_id).read_text(encoding="utf-8")
            self.assertIn('"event": "episode_budget_exceeded"', trace)


if __name__ == "__main__":
    unittest.main()
