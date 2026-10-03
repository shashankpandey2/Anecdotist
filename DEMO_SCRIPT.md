# Screen Recording Script

Keep the recording under five minutes. Run these commands from the repository root after running `uv sync` and adding `GOOGLE_API_KEY` to `.env`.

1. Create a story and show the complete arc:

   `uv run story-writer new "A talented young batter from a neighborhood club earns a place at a state cricket academy and must rebuild her game after a serious injury." --id recording`

   `uv run story-writer plan recording`

2. Approve the plan and draft the first episode:

   `uv run story-writer approve-plan recording`

   `uv run story-writer write recording`

3. Approve episode 1, add a durable creative note, and draft episode 2. Point out that the note is stored in state and will be included in future context:

   `uv run story-writer review recording 1 approve`

   `uv run story-writer feedback recording "Slow down the friendship; trust must be earned through action."`

   `uv run story-writer write recording`

4. Approve episode 2, draft episode 3, then reject it with feedback. Show that the next draft returns to the same episode number:

   `uv run story-writer review recording 2 approve`

   `uv run story-writer write recording`

   `uv run story-writer review recording 3 reject "Let the recovery breathe before the comeback."`

   `uv run story-writer write recording`

5. Finish by showing resumability and the trace:

   `uv run story-writer show recording`

   Open `.story-data/recording.trace.jsonl` and point out draft latency, word count, estimated tokens, cost, approvals, and the rejected review.

The checked-in `demo/` directory contains the same flow already completed through 15 approved episodes, including both interventions.
