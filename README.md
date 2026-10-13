# Interview-Me Assistant

A LangGraph-powered interview assistant that delivers first-person responses grounded exclusively in approved resume and project evidence.

Key capabilities:

- Guarded routing for interview questions, greetings, and prohibited requests
- Evidence-scoped answers from approved resume fields and project records
- Hybrid project retrieval using Chroma, BM25, and reciprocal rank fusion
- Streaming responses in a conversational Streamlit interface

## Quick Start

For a local run, you need Python 3.11 or newer, [uv](https://docs.astral.sh/uv/),
and an OpenRouter API key.

1. Install dependencies and create the environment file:

```bash
cp .env.example .env
uv sync
```

2. Add your API key to `.env`:

```dotenv
OPENROUTER_API_KEY=your-api-key
```

3. Start the application:

```bash
uv run streamlit run src/frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501), enter an interview
question, and watch the grounded response stream into the chat. Conversation
history is retained for follow-up questions during the browser session.

Alternatively, run the application with Docker:

```bash
docker compose up --build
```

The container exposes the same interface at
[http://localhost:8501](http://localhost:8501).

The defaults in `.env.example` configure the main, guard, embedding, and
evaluation models. `GITHUB_USERNAME` is only required when regenerating project
data. Keep `.env` local and never commit it.

To trace graph, agent, tool, and model runs in LangSmith, set
`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` in `.env`.

The graph entry point is `src.backend.graph:graph`.

## Agent Architecture

The LangGraph workflow has three nodes. `guard_route` is a conditional router
that selects one node based on the conversation.

```mermaid
flowchart TD
  Start([START]) --> Guard{guard_route}
  Guard -->|safe| Main[main_agent]
  Guard -->|greeting| Greeting[greeting]
  Guard -->|prohibited| Refusal[refuse_to_answer]
  Main -.->|field lookup| Resume[Resume field tools]
  Main -.->|project query| Project[  RAG subagent \n get_project_detail]
  Resume -.->|profile evidence| Main
  Project -.->|ranked project evidence| Main
  Main --> End([END])
  Greeting --> End
  Refusal --> End
```

Solid arrows are LangGraph transitions. Dashed arrows show internal tool routing
within `main_agent`; the tools are not additional graph nodes.

| Graph element              | Type               | Purpose                                                                              |
| -------------------------- | ------------------ | ------------------------------------------------------------------------------------ |
| `guard_route`              | Conditional router | Classifies the conversation as safe, greeting, or prohibited.                        |
| `main_agent`               | LangGraph node     | Routes questions to the appropriate tool and streams a grounded first-person answer. |
| Resume field tools         | Tool               | Return approved profile evidence for general interview questions.                    |
| `get_project_detail`       | Tool               | Runs hybrid Chroma and BM25 search and returns ranked project evidence.              |
| `greeting`                 | LangGraph node     | Returns a deterministic greeting without invoking the main agent.                    |
| `refuse_to_answer` | LangGraph node     | Returns a deterministic response when the guard rejects a request.                   |

## Ingest Data (Optional)

The repository includes generated profile and project artifacts. To rebuild
them from a resume and public GitHub repositories, set `GITHUB_USERNAME` and
run:

```bash
uv run python -m src.backend.ingest --regenerate-json
```

This extracts resume text, populates the candidate profile, generates project
summaries, and rebuilds the Chroma and BM25 indexes under `src/backend/data`.

## Project Layout

| Path                  | Purpose                                                            |
| --------------------- | ------------------------------------------------------------------ |
| `src/backend/graph`   | Request routing, main agent, evidence tools, and hybrid retrieval. |
| `src/backend/ingest`  | Resume extraction and GitHub project indexing pipelines.           |
| `src/backend/data`    | Generated profile, project records, and persisted search indexes.  |
| `src/frontend/app.py` | Streamlit chat interface and response streaming adapter.           |
| `tests`               | Deterministic unit and integration tests.                          |
| `eval`                | Golden cases, DeepEval runner, cached results, and reports.        |

## Run the Tests

Run the complete pytest suite from the repository root:

```bash
uv run pytest
```

These tests cover graph routing, artifact cleanup, ingestion, project indexing,
and hybrid project retrieval. The DeepEval workflow below is separate and makes
live model requests.

## Run the Evaluation

The DeepEval suite runs 27 golden interview cases against the graph. The 24
answerable cases are scored for hallucination, answer relevancy, faithfulness,
contextual precision, and contextual recall. The 3 cases tagged
`should_abstain` are scored only with the `Correct Abstention` G-Eval metric.
Metrics must meet their configured thresholds (default `0.5`). This command
makes live model requests and may incur provider costs.

```bash
uv run eval/eval_agent.py
```

By default, the evaluator reuses `eval/interview_agent_test_cases.jsonl` when
the cache is present and valid. Regenerate the agent responses and overwrite
that cache with:

```bash
uv run eval/eval_agent.py --rerun
```

### Latest evaluation results

Run on 2026-10-13, with answerable and abstention cases evaluated separately:

| Case group | Cases | Passed | Failed | Pass rate |
| ---------- | ----: | -----: | -----: | --------: |
| Answerable | 24    | 24     | 0      | 100.00%  |
| Abstention | 3     | 3      | 0      | 100.00%  |
| **Total**  | **27**| **27** | **0**  | **100.00%** |

| Group      | Metric                      | Average score | Metric pass rate |
| ---------- | --------------------------- | ------------: | ---------------: |
| Answerable | Hallucination               | 1.00          | 24 / 24          |
| Answerable | Answer relevancy             | 0.96          | 24 / 24          |
| Answerable | Faithfulness                 | 1.00          | 24 / 24          |
| Answerable | Contextual precision         | 0.81          | 24 / 24          |
| Answerable | Contextual recall            | 0.98          | 24 / 24          |
| Abstention | Correct Abstention (G-Eval) | 1.00          | 3 / 3            |

Per-case reports: [answerable cases](eval/eval_reports/answerable/evaluation_20261013_133707.md) · [abstention cases](eval/eval_reports/abstention/evaluation_20261013_133716.md).

### Findings

- **All 27 cases passed:** all 24 answerable cases passed each of their five
  metrics, and all 3 abstention cases passed the `Correct Abstention` metric.
- **Grounding and retrieval scores were strong:** hallucination and faithfulness
  averaged 1.00, answer relevancy 0.96, contextual precision 0.81, and
  contextual recall 0.98. Each metric passed on all 24 answerable cases.
- **Abstentions were handled correctly:** the three unsupported-claim and
  prompt-injection cases averaged 1.00 on `Correct Abstention` and all passed.

### Future work

1. Add model latency, token usage, estimated cost, and retrieval diagnostics to
   LangSmith traces and an optional local debug view.
2. Harden deployment with authentication, rate limiting, request timeouts, and
   user-friendly recovery when model or retrieval providers are unavailable.
3. Expand evaluation with multi-turn and guard-route cases, format compliance,
   and multilingual prompts.
