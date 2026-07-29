"""FastAPI app: serves the support inbox UI and streams pipeline progress.

State is in-memory on purpose — the inbox is a demo surface; persistence,
auth, and ticket-system sync are integration work, not the interesting part.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Mapping

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .indexer import RepoIndex
from .llm import DEFAULT_MODEL, LLM
from .pipeline import DEPLOY_TAGS, Pipeline
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


def resolve_sources(env: Mapping[str, str],
                    demo_repo: Callable[[], str]) -> tuple[str, str | None]:
    """Pick the repo to index and the manifest that describes its releases.

    The demo manifest belongs to the demo repo only. Handing it to a foreign
    repo would check that repo's real tags against Ritmo's versions: every
    release_for() lookup misses, so compute_verdict() sees status "internal"
    and every verdict collapses to fix_coming — already_fixed and regression
    become unreachable. A foreign repo without an explicit LANDED_RELEASES
    gets None, which is RepoIndex's cue to treat git tags as released.
    """
    repo = env.get("LANDED_REPO")
    if repo:
        return repo, env.get("LANDED_RELEASES")
    # demo repo generation is lazy: pointing at your own repo must not build Ritmo
    return demo_repo(), env.get("LANDED_RELEASES") or str(DEMO / "releases.json")


def product_name(env: Mapping[str, str], repo_path: str) -> str:
    """What to call the product being supported, for the header and the sign-off.

    The demo carries a brand its directory name doesn't ("Ritmo" lives in
    demo/app-repo); every other repo is named by its directory until someone
    says otherwise with LANDED_PRODUCT. Without this the inbox greets a
    foreign repo as Ritmo and signs its replies "The Ritmo team".
    """
    explicit = env.get("LANDED_PRODUCT")
    if explicit:
        return explicit
    repo = Path(repo_path)
    if repo.resolve() == (DEMO / "app-repo").resolve():
        return "Ritmo"
    return repo.name


@asynccontextmanager
async def lifespan(app: FastAPI):
    repo_path, releases = resolve_sources(os.environ, lambda: str(_ensure_demo_repo()))
    index = RepoIndex(repo_path, releases)
    llm = LLM()
    deploy_model = os.environ.get("LANDED_DEPLOY_MODEL", DEPLOY_TAGS)
    product = product_name(os.environ, repo_path)
    app.state.pipeline = Pipeline(index, llm, team_signature=f"The {product} team",
                                  deploy_model=deploy_model)
    app.state.mode = llm.mode
    app.state.deploy_model = deploy_model
    app.state.credential_error = llm.credential_error
    app.state.product = product
    app.state.repo = {"name": Path(repo_path).name, "path": str(Path(repo_path).resolve())}
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
        "credential_error": app.state.credential_error,
        "deploy_model": app.state.deploy_model,
        "product": app.state.product,
        "repo": app.state.repo,
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
