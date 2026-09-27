# Interview-Me Assistant

A LangGraph interview assistant that answers in the candidate's voice using only
approved resume and project evidence.

## Architecture

The top-level graph keeps `MessagesState` as its only state:

```text
START -> guard conditional edge
		 | greeting -> fixed greeting -> END
		 | irrelevant -> fixed relevance response -> END
		 | relevant -> main_agent
					-> END
```

`resume_subagent` and `project_subagent` are tools of `main_agent`. They are not
LangGraph nodes. The main agent calls them, merges their returned evidence, and
composes the final answer internally.

The repository is organized by runtime responsibility:

```text
interview_me/
├── backend/         # graph, agents, ingestion, and evaluation
└── frontend/        # Streamlit presentation layer
interview_me/backend/data/ # local profile, resume, and project artifacts
interview_me/tests/        # unit and graph contract tests
```

## Setup

Requires Python 3.11 or newer.

```bash
cp .env.example .env
uv sync
uv run pytest -q
```

`uv sync` creates the local virtual environment and installs the locked runtime
and development dependencies from `uv.lock`.

Set `OPENROUTER_API_KEY` in `.env` before running a live model request. Keep
`.env` local and never commit it.

To trace supervisor, subagent, tool, and model runs in LangSmith, set
`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` in `.env`.

The graph entry point is `interview_me.backend.graph:graph`.

## Run the agent

Configure `.env` with `OPENROUTER_API_KEY`, `GITHUB_USERNAME`, and the path to
your resume PDF if it is not at `interview_me/backend/data/resume.pdf`. Then
build the approved profile, project summaries, and Chroma semantic index:

```bash
uv run python -m interview_me.backend.ingest --regenerate-json
```

Start the Streamlit application:

```bash
uv run streamlit run interview_me/frontend/app.py
```

Open the local URL printed by Chainlit, then ask questions about the candidate's
background or projects. On later runs, use `uv run python -m
interview_me.backend.ingest` to reuse existing JSON artifacts while refreshing
the Chroma index; pass `--regenerate-json` whenever the resume or GitHub project
data has changed.

The ingest command extracts text from `RESUME_PATH`, uses the configured model to
populate the closed candidate profile schema, and writes `profile.json`. It also
generates `projects.json` from GitHub, embeds project summaries into Chroma, and
writes all artifacts under `DATA_DIR`. Override the resume and output locations
with `--input-resume` and `--output-dir`.

To run the local DeepEval quality suite after ingestion:

```bash
uv run python -m interview_me.backend.evaluation.run
```

The suite uses `EVALUATION_MODEL` through OpenRouter and requires
`OPENROUTER_API_KEY`. It is intentionally local-only and is not part of the
normal pytest run. It writes a JSON report to
`artifacts/evaluations/deepeval-report.json` with route precision, recall, F1,
answer relevancy, faithfulness, project retrieval contextual precision/recall,
robustness pass rate, and per-case p50/p95/max end-to-end latency.

The curated cases in `interview_me/backend/evaluation/datasets/interview_cases.json`
are the source of truth. To generate additional review-only suggestions from the
approved profile and project artifacts, run:

```bash
uv run python -m interview_me.backend.evaluation.generate_cases
```

This writes `artifacts/evaluations/generated-cases.json` without changing the
curated dataset. Review generated rows before manually promoting them into the
version-controlled fixture.

The guard uses a structured model call. Unit tests mock this call; live runtime
uses `OPENROUTER_API_KEY`.

Project summaries are embedded with `EMBEDDING_MODEL` through OpenRouter and
persisted in `CHROMA_DIR`. The project subagent combines Chroma semantic search
with BM25 lexical search using weighted reciprocal-rank fusion, then returns
matching approved README content as evidence.

For Docker:

```bash
docker compose up --build
```

## Current scope

The graph, grounded artifact tools, hybrid project retrieval, ingestion commands,
Streamlit adapter, Docker packaging, structured model decisions, and evaluation
baseline are implemented. DeepEval/JEV integration remains the next
accuracy-focused layer.
