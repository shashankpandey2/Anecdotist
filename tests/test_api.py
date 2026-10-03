import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from story_writer.api import app


class StoryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.previous_data_dir = os.environ.get("STORY_DATA_DIR")
        self.previous_provider = os.environ.get("STORY_PROVIDER")
        os.environ["STORY_DATA_DIR"] = self.directory.name
        os.environ["STORY_PROVIDER"] = "local"
        self.client = TestClient(app)

    def tearDown(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("STORY_DATA_DIR", None)
        else:
            os.environ["STORY_DATA_DIR"] = self.previous_data_dir
        if self.previous_provider is None:
            os.environ.pop("STORY_PROVIDER", None)
        else:
            os.environ["STORY_PROVIDER"] = self.previous_provider
        self.directory.cleanup()

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_story_review_flow(self) -> None:
        created = self.client.post(
            "/stories",
            json={"story_id": "api-test", "premise": "A batter earns a place in a state cricket academy."},
        )
        self.assertEqual(created.status_code, 200)
        self.assertEqual(len(created.json()["arc"]["episodes"]), 200)

        self.assertEqual(self.client.post("/stories/api-test/arc/approve").status_code, 200)
        drafted = self.client.post("/stories/api-test/episodes/next")
        self.assertEqual(drafted.status_code, 200)
        self.assertEqual(drafted.json()["next_episode"], 2)

        feedback = self.client.post(
            "/stories/api-test/feedback",
            json={"feedback": "Let the partnership develop through action."},
        )
        self.assertEqual(feedback.status_code, 200)

        reviewed = self.client.post(
            "/stories/api-test/episodes/1/review",
            json={"decision": "approve"},
        )
        self.assertEqual(reviewed.status_code, 200)
        self.assertEqual(reviewed.json()["episodes"][0]["status"], "approved")

    def test_missing_story_returns_not_found(self) -> None:
        response = self.client.get("/stories/missing")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
