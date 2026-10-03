from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import StoryOrchestrator
from .models import StoryState, to_dict
from .provider import GeminiProvider, LocalProvider, OllamaProvider
from .store import StoryStore


class CreateStoryRequest(BaseModel):
    premise: str = Field(min_length=1)
    story_id: str | None = None


class FeedbackRequest(BaseModel):
    feedback: str = Field(min_length=1)


class ReviewRequest(BaseModel):
    decision: str
    text: str = ""


app = FastAPI(title="Agentic Serial Story Writer", version="0.1.0")


def build_engine() -> StoryOrchestrator:
    data_dir = Path(os.getenv("STORY_DATA_DIR", ".story-data"))
    provider_name = os.getenv("STORY_PROVIDER", "gemini")
    if provider_name == "local":
        provider = LocalProvider()
    elif provider_name == "ollama":
        provider = OllamaProvider(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        provider = GeminiProvider(model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"))
    cost_cap = float(os.getenv("STORY_EPISODE_COST_CAP_USD", "0.02"))
    return StoryOrchestrator(
        StoryStore(data_dir),
        provider=provider,
        max_episode_cost_usd=cost_cap,
    )


def load_story(engine: StoryOrchestrator, story_id: str) -> StoryState:
    try:
        return engine.store.load(story_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Story {story_id} was not found") from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/stories")
def create_story(request: CreateStoryRequest) -> dict:
    engine = build_engine()
    state = engine.create(request.premise, request.story_id)
    engine.create_arc(state)
    return to_dict(state)


@app.get("/stories/{story_id}")
def get_story(story_id: str) -> dict:
    engine = build_engine()
    return to_dict(load_story(engine, story_id))


@app.post("/stories/{story_id}/arc/approve")
def approve_arc(story_id: str) -> dict:
    engine = build_engine()
    state = load_story(engine, story_id)
    try:
        return to_dict(engine.approve_arc(state))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/stories/{story_id}/episodes/next")
def write_episode(story_id: str) -> dict:
    engine = build_engine()
    state = load_story(engine, story_id)
    try:
        engine.write_next(state)
        return to_dict(state)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/stories/{story_id}/feedback")
def add_feedback(story_id: str, request: FeedbackRequest) -> dict:
    engine = build_engine()
    state = load_story(engine, story_id)
    return to_dict(engine.add_directive(state, request.feedback))


@app.post("/stories/{story_id}/episodes/{number}/review")
def review_episode(story_id: str, number: int, request: ReviewRequest) -> dict:
    engine = build_engine()
    state = load_story(engine, story_id)
    try:
        return to_dict(engine.review_episode(state, number, request.decision, request.text))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def main() -> None:
    uvicorn.run(
        "story_writer.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
