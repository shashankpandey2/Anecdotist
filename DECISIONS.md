# Decisions

## Agent design

This is an agentic workflow with an explicit LangGraph writer-critic loop. The planner owns structure, the writer owns prose, and the critic owns the pre-review quality gate. The orchestrator owns persistence and state transitions and never asks an agent to mutate persistent state directly. FastAPI exposes the same controls over HTTP; the CLI remains the simplest demonstration path. This separation keeps retries bounded and makes each decision traceable.

## 1. Memory at episode 150

The system does not place all earlier prose in the generation context. The approved 200-episode arc is structured by phase and episode. A compact canon stores durable facts, character states, timeline entries, and open threads. Generation retrieves the current episode plan, recent canon entries, unresolved threads, and durable human directives. Each approved episode can be summarized into canon by the provider layer; the local demo uses the episode plan as its deterministic summary. This keeps context bounded while preserving the information later episodes need.

## 2. Human intervention

A human approves or edits the full arc before writing starts, then reviews every draft. Approval is the checkpoint because it catches structural problems before they multiply. Feedback is stored as a durable directive, so “let Mira and Kabir earn their partnership” or “let the recovery breathe” affects future prompts, not only the visible draft. A historical edit deletes dependent future episodes and rebuilds canon, making the dependency explicit instead of silently leaving inconsistent prose.

## 3. Consistency and repetition

Before review, the engine checks the 400-700 word bound, sentence shape, hook signal, planned character presence, and near-duplicate similarity against earlier drafts. Structured episode plans declare which characters and threads must advance. Canon and timeline are updated only as episodes are accepted. The remaining gap is deep contradiction detection; a production provider should add fact extraction and semantic consistency checks against the canon before the human approval gate.

## 4. What breaks first

The first likely failure is not storage size but canon quality: an incorrect summary can make a later episode consistently wrong. The next failure is repetitive plotting across a 200-episode plan. The practical fix is to version canon entries, show the retrieved context in the review UI, and add a periodic human arc checkpoint with contradiction and similarity reports. The state model already supports invalidating downstream episodes when a historical decision changes.
