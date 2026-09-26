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
					-> answer evaluator conditional edge
						| approved -> END
						| rejected -> fixed abstention -> END
```

`resume_subagent` and `project_subagent` are tools of `main_agent`. They are not
LangGraph nodes. The main agent calls them, merges their returned evidence, and
composes the final answer internally.

The repository is organized by runtime responsibility:

```text
interview_me/
├── backend/         # graph, agents, ingestion, and evaluation
└── frontend/        # Chainlit presentation layer
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

The graph entry point is `interview_me.backend.graph:graph`.

To ingest the real resume PDF and project index:

```bash
uv run python -m interview_me.backend.ingest
uv run python -m interview_me.backend.evaluation.run
uv run chainlit run interview_me/frontend/app.py
```

The ingest command extracts text from `RESUME_PATH`, uses the configured model to
populate the closed candidate profile schema, and writes `profile.json`. It also
validates the GitHub-generated `projects.json`. Override inputs with `--resume`,
`--projects`, or `--output` when needed.

To refresh public, non-fork repository artifacts using `GITHUB_USERNAME` from
`.env`:

```bash
uv run python -m interview_me.backend.github_ingest
```

An explicit username may still be supplied as the positional argument.

The guard and answer evaluator always use structured model calls. Unit tests mock
these calls; live evaluation and runtime use `OPENROUTER_API_KEY`.

For Docker:

```bash
docker compose up --build
```

## Current scope

The graph, grounded artifact tools, ingestion commands, Chainlit adapter, Docker
packaging, structured model decisions, and evaluation baseline are implemented.
Chroma/BM25 hybrid retrieval and DeepEval/JEV integration remain
the next accuracy-focused layer beyond the current local keyword index.
