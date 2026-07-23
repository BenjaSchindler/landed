"""FastAPI app: serves the support inbox UI and streams pipeline progress.

State is in-memory on purpose — the inbox is a demo surface; persistence,
auth, and ticket-system sync are integration work, not the interesting part.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .indexer import RepoIndex
from .llm import DEFAULT_MODEL, LLM
from .pipeline import Pipeline
from .schemas import AnalysisResult, FeedbackItem

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"


def _ensure_demo_repo() -> Path:
    repo = DEMO / "app-repo"
    if not (repo / ".git").exists():
        import sys
        sys.path.insert(0, str(DEMO.parent))
        from demo.make_demo_repo import main as make_repo
        make_repo()
    return repo


@asynccontextmanager
async def lifespan(app: FastAPI):
    repo_path = os.environ.get("LANDED_REPO") or str(_ensure_demo_repo())
    releases = os.environ.get("LANDED_RELEASES") or str(DEMO / "releases.json")
    index = RepoIndex(repo_path, releases)
    llm = LLM()
    app.state.pipeline = Pipeline(index, llm)
    app.state.mode = llm.mode
    app.state.items = {}
    app.state.results = {}
    for raw in json.loads((DEMO / "seed_feedback.json").read_text()):
        item = FeedbackItem(**raw)
        app.state.items[item.id] = item
    yield


app = FastAPI(title="Landed", lifespan=lifespan)


@app.get("/")
def index_page():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/meta")
def meta():
    pipeline: Pipeline = app.state.pipeline
    return {
        "mode": app.state.mode,
        "model": DEFAULT_MODEL if app.state.mode == "live" else None,
        "commits": len(pipeline.index.commits),
        "releases": [r.model_dump() for r in pipeline.index.releases],
    }


@app.get("/api/items")
def list_items():
    out = []
    for item in app.state.items.values():
        result: AnalysisResult | None = app.state.results.get(item.id)
        out.append({
            **item.model_dump(),
            "verdict": result.verdict.verdict.value if result else None,
        })
    return out


class NewItem(BaseModel):
    text: str
    channel: str = "pasted"


@app.post("/api/items")
def add_item(body: NewItem):
    if not body.text.strip():
        raise HTTPException(400, "empty feedback")
    n = sum(1 for i in app.state.items if i.startswith("fb-user")) + 1
    item = FeedbackItem(id=f"fb-user-{n:03d}", text=body.text.strip(),
                        channel=body.channel, received_at="")
    app.state.items[item.id] = item
    return item.model_dump()


@app.get("/api/items/{item_id}/result")
def get_result(item_id: str):
    result: AnalysisResult | None = app.state.results.get(item_id)
    if result is None:
        raise HTTPException(404, "not analyzed yet")
    return result.model_dump(mode="json")


@app.get("/api/items/{item_id}/analyze")
def analyze(item_id: str):
    item: FeedbackItem | None = app.state.items.get(item_id)
    if item is None:
        raise HTTPException(404, "unknown item")
    pipeline: Pipeline = app.state.pipeline

    def stream():
        for stage, payload in pipeline.analyze_stages(item):
            if stage == "done":
                app.state.results[item.id] = payload
                data = payload.model_dump(mode="json")
            else:
                data = payload
            yield f"event: {stage}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})
