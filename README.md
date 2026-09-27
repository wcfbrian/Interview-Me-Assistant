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
src/
├── backend/         # graph, agents, ingestion, and evaluation
└── frontend/        # Streamlit presentation layer
src/backend/data/ # local profile, resume, and project artifacts
tests/            # unit and graph contract tests
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

The graph entry point is `src.backend.graph:graph`.

## Run the agent

Configure `.env` with `OPENROUTER_API_KEY`, `GITHUB_USERNAME`, and the path to
your resume PDF if it is not at `src/backend/data/resume.pdf`. Then
build the approved profile, project summaries, and Chroma semantic index:

```bash
uv run python -m src.backend.ingest --regenerate-json
```

Start the Streamlit application:

```bash
uv run streamlit run src/frontend/app.py
```

Open the local URL printed by Chainlit, then ask questions about the candidate's
background or projects. On later runs, use `uv run python -m
src.backend.ingest` to reuse existing JSON artifacts while refreshing
the Chroma index; pass `--regenerate-json` whenever the resume or GitHub project
data has changed.

The ingest command extracts text from `RESUME_PATH`, uses the configured model to
populate the closed candidate profile schema, and writes `profile.json`. It also
generates `projects.json` from GitHub, embeds project summaries into Chroma, and
writes all artifacts under `DATA_DIR`. Override the resume and output locations
with `--input-resume` and `--output-dir`.


For Docker:

```bash
docker compose up --build
```

## Current scope

The graph, grounded artifact tools, hybrid project retrieval, ingestion commands,
Streamlit adapter, Docker packaging, structured model decisions, and evaluation
baseline are implemented. DeepEval/JEV integration remains the next
accuracy-focused layer.
