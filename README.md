# Interview Me

An evidence-grounded Chainlit interview assistant that answers as the candidate
using only an ingested resume and public, non-fork GitHub project READMEs.

## Architecture

```text
Chainlit -> GLM guard -> LangGraph main workflow -> resume/project retrieval
		 -> DeepSeek answer generation -> GLM final judge -> response
```

The main agent owns short-term conversation memory. Resume and project subagents
are stateless for efficiency. `typesafe/jev-1.13` is used only by DeepEval as an
evaluation LLM judge; it is not used in production routing or answer judging.

## Setup

Requirements: Python 3.12, `uv`, an OpenRouter key, and a text-based PDF resume.

```bash
cp .env.example .env
# Set OPENROUTER_API_KEY, CHAINLIT_AUTH_PASSWORD, and the local resume path.
uv sync --dev
mkdir -p data
cp /path/to/resume.pdf data/resume.pdf
uv run python -m backend.cli.ingest
uv run chainlit run frontend/chainlit_app.py
```

Open `http://localhost:8000`. The Chainlit login uses
`CHAINLIT_AUTH_USERNAME` and `CHAINLIT_AUTH_PASSWORD`.

## Docker

```bash
cp .env.example .env
# Put the resume at data/resume.pdf.
docker compose up --build
```

Generated profiles, GitHub cache, Chroma data, and evaluation artifacts remain in
the gitignored `data/` directory. The first ingestion requires a valid resume and
GitHub sync; later runs can reuse a valid cached project index.

## Model roles

```dotenv
MAIN_MODEL=deepseek/deepseek-v4.1-flash
GUARD_MODEL=z-ai/glm-5.3-flash
EVALUATION_MODEL=typesafe/jev-1.13
EMBEDDING_MODEL=openai/text-embedding-3-small
```

GLM handles the production guard and final judge. DeepSeek handles the main agent,
resume subagent, project subagent, extraction, and summaries. Jev is evaluation-only.

## Testing and evaluation

```bash
uv run pytest
uv run ruff check backend frontend evaluations tests
```

The committed tests are deterministic and use synthetic data. The optional
DeepEval suite uses personalized local fixtures and Jev as the LLM judge. Report:

- guard precision, recall, F1, and confusion matrix;
- retrieval Recall@3, MRR, and nDCG@3;
- groundedness, answer relevance, abstention accuracy, and unsupported-claim rate;
- per-node and end-to-end p50/p95 latency, token usage, and failure rate.

The initial smoke set contains 30 human-reviewed scenarios across HR questions,
hiring-manager questions, projects, multi-turn follow-ups, unsupported facts,
greetings, off-topic prompts, and prompt injection. Metrics are reported before
release thresholds are introduced.

## Safety and privacy

The assistant refuses unsupported claims with `No information found.` Sensitive
resume fields such as contact details, IDs, compensation, and protected traits are
excluded. Production logs should contain metadata only; never commit `.env` or its
credentials. Rotate any OpenRouter key that has been exposed outside the local
environment.
