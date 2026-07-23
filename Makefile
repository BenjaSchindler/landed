PY := .venv/bin/python

.PHONY: setup demo eval eval-replay record test logic

setup:            ## create the venv and install dependencies
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

demo:             ## run the inbox at http://localhost:8712 (keyless replay, or live with .env)
	.venv/bin/uvicorn harness.server:app --port 8712

eval:             ## grade the 25 labeled cases (live if OPENAI_API_KEY resolves, else replay)
	$(PY) -m harness.eval

eval-replay:      ## grade against the shipped recorded fixtures, no key needed
	LANDED_REPLAY=1 $(PY) -m harness.eval --strict

record:           ## re-record every fixture against the live model, grading as it goes (~70 calls)
	LANDED_RECORD=1 $(PY) -m harness.eval

test:             ## unit tests for the deterministic core (no key, no git, no network)
	$(PY) -m unittest discover -s tests -t .

logic:            ## harness-logic gate: hand-authored model outputs through the real pipeline
	$(PY) -m scripts.author_fixtures
