# Landed — "is this already fixed, and when does the fix reach this user?"

Landed is a support-inbox tool for teams that ship through app stores. A support
agent pastes (or receives) a piece of raw user feedback — any language, any
shape — and gets back an **evidence-gated verdict** (*already fixed / fix on its
way / regression / not fixed / needs info*), a **ready-to-send reply** in the
reporter's language, and, when a human is needed, a **pre-built engineering
handoff** — without interrupting an engineer.

It answers the question support actually has: not "is this bug known?" but
*"has this been fixed, and is the fix in **this reporter's** hands yet?"* —
which requires joining fuzzy user language against the git history **and** the
release/rollout state, then doing version arithmetic the model is never
trusted to do.

See [DESIGN.md](DESIGN.md) for the problem thesis, architecture, and tradeoffs.

## Run it (no API key needed)

The repo ships with recorded model outputs ("replay mode"), a deterministic
demo product (**Ritmo**, a bilingual habit-tracking app with 32 commits and 4
releases), and 9 seeded feedback items:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# (or: uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt)

.venv/bin/uvicorn harness.server:app --port 8712
# open http://localhost:8712 and click through the inbox
```

The demo repo is generated automatically on first boot (deterministic dates and
authors, so commit shas — and therefore fixtures — are stable across machines).

## Live mode

```bash
export OPENAI_API_KEY=sk-...               # any key with gpt-5.5 access
.venv/bin/uvicorn harness.server:app --port 8712
```

With credentials resolving, the pipeline calls the model for real — including
for feedback you paste yourself. `OPENAI_MODEL` overrides the default model
(`gpt-5.5`).

To replace the hand-authored fixtures with real recorded model outputs:

```bash
LANDED_RECORD=1 .venv/bin/python -m scripts.record_fixtures
```

## Evals

```bash
.venv/bin/python -m harness.eval               # live if credentials resolve, else replay
LANDED_REPLAY=1 .venv/bin/python -m harness.eval   # force replay
```

25 labeled cases (Spanish/Chilean and English, vague and precise, including
regression traps and near-miss baits). The headline metric is the
**false-"fixed" rate** — telling a user something is handled when it isn't is
the one failure the tool must not make. In replay mode this grades the
deterministic harness (routing, retrieval, citation checks, version
arithmetic); after `record_fixtures` it grades the live model end to end.

## Point it at your own repo

```bash
LANDED_REPO=/path/to/your/repo .venv/bin/uvicorn harness.server:app --port 8712
```

Without a `LANDED_RELEASES` manifest, every git tag counts as a released
version. Pasting feedback about your own product requires live mode.

## What's real vs. stubbed

**Real:** the whole harness — intake normalization + bilingual query expansion
(structured outputs), BM25 retrieval over git history, adjudication with
citation validation and a corrective retry, deterministic version arithmetic,
reply drafting with a version-allowlist guardrail and a templated safe
fallback, escalation packet generation, SSE progress streaming, the eval
runner, and graceful degradation on every failure path.

**Stubbed / simulated:**
- The product under support (Ritmo) is synthetic — generated git history plus a
  hand-written release manifest standing in for App Store / Play Console APIs.
- Feedback arrives by seed file or paste — no Zendesk/Intercom/store-review
  ingestion.
- The "Copy reply" button is the delivery mechanism; nothing is auto-sent (by
  design, not just scope).
- Inbox state is in-memory; restart re-seeds it.

**Known limitations:** single-repo evidence only (no PR/issue-tracker corpus);
BM25 needs the LLM's query expansion to bridge user↔dev vocabulary — at much
larger corpus sizes a hybrid with embeddings would be the next step; verdicts
trust the release manifest, which in production must come from store APIs, not
a file; duplicate/known-issue clustering across reports is out of scope.

## Repo layout

```
harness/         the pipeline: schemas, indexer (git+BM25), llm seam, prompts,
                 pipeline (guardrails + verdict logic), FastAPI server, eval runner
web/index.html   the support inbox (vanilla JS, no build step)
demo/            generator for the deterministic demo repo + releases + seed feedback
evals/           25 labeled cases (gold verdicts + fix versions)
fixtures/replay/ recorded/authored model outputs for keyless runs
scripts/         author_fixtures (fixtures + logic test), record_fixtures (live re-record)
```
