# Interview-Me Assistant

A LangGraph interview assistant that answers in the candidate's voice using only
approved resume and project evidence.

## Setup

Prerequisites:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An OpenRouter API key

```bash
cp .env.example .env
uv sync
```

`uv sync` creates the local virtual environment and installs the locked runtime
and development dependencies from `uv.lock`.

Set at least these values in `.env` before making live model requests:

```dotenv
OPENROUTER_API_KEY=your-api-key
```

The defaults in `.env.example` configure the runtime, guard, embedding, and
evaluation models. Keep `.env` local and never commit it.

To trace supervisor, subagent, tool, and model runs in LangSmith, set
`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` in `.env`.

The graph entry point is `src.backend.graph:graph`.



## Run the Agent

Configure `.env` with `OPENROUTER_API_KEY`, `GITHUB_USERNAME`

Start the Streamlit application:

```bash
uv run streamlit run src/frontend/app.py
```

For Docker:

```bash
docker compose up --build
```

## Ingest Data (Optional)

Run the following command to extract text from the resume, populate the candidate profile, generate project summaries from GitHub, embed them into Chroma, and write all artifacts under the data directory:

```bash
uv run python -m src.backend.ingest --regenerate-json
```

## Run the Tests

Run the complete pytest suite from the repository root:

```bash
uv run pytest
```

Use `-q` for concise output or target a specific file or test while developing:

```bash
uv run pytest -q
uv run pytest tests/test_graph.py
uv run pytest tests/test_graph.py::test_guard_routes_greeting_to_fixed_response
```

These tests cover graph routing, artifact cleanup, ingestion, project indexing,
and hybrid project retrieval. The DeepEval workflow below is separate and makes
live model requests.

## Run the Evaluation

The DeepEval suite runs 27 golden interview cases against the graph, then scores
the answers for hallucination, answer relevancy, faithfulness, contextual
precision, and contextual recall. To pass the evaluation, all metrics must meet the required thresholds (>0.5).

```bash
uv run eval/eval_agent.py
```

### Recorded baseline

Run on 2026-09-28 with 27 test cases:

```text
Test Results (27 total tests):
	Pass Rate: 74.07% | Passed: 20 | Failed: 7

Latency: p50=11.691s, p95=69.168s
```

The full per-case report is available at
[`eval/eval_reports/evaluation_20260928_082553.md`](eval/eval_reports/evaluation_20260928_082553.md).

### Result interpretation

- **Grounding is strong:** hallucination and faithfulness both scored 1.00 with
  a 100% metric pass rate. Answer relevancy averaged 0.95 and passed 26 of 27
  cases.
- **Contextual precision determines the overall result:** all seven failed test
  cases fell below the 0.50 contextual-precision threshold. Four otherwise
  relevant and faithful project answers (`lstm-forecasting-target`,
  `video-slide-report`, `java-server-unsupported-benchmark`, and
  `healthcare-vague-intent`) indicate that retrieval ranking or excess context
  needs improvement.
- **Three failures are evaluation artifacts:** `drought-unknown-accuracy`,
  `fabricated-award`, and `qabot-prompt-injection` returned the exact expected
  refusal and did not hallucinate, but contextual precision and recall scored
  zero. Refusal cases should be evaluated with safety and answer-correctness
  metrics instead of requiring supporting retrieval context.
- **Tail latency is high:** p95 latency was 69.168 seconds versus an 11.691-second
  median, so a small number of graph calls dominate response time. Tracing those
  cases by route, retrieval count, and model duration is the next performance
  priority.

The 74.07% test pass rate therefore understates behavioral correctness: it mixes
genuine retrieval-quality weaknesses with metrics that do not fit no-evidence
refusal cases. Future baselines should report retrieval cases and refusal cases
separately.
