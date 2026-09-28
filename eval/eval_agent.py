import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from time import perf_counter

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric)
from deepeval.models.llms.openrouter_model import OpenRouterModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from deepeval.evaluate import DisplayConfig

from src.backend.graph import build_graph

load_dotenv()

DATASET_PATH = Path(__file__).with_name("interview_agent_goldens.jsonl")
TEST_CASE_CACHE_PATH = Path(__file__).with_name("interview_agent_test_cases.jsonl")



def _percentile(values: list[float], percentile: float) -> float:
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1 - weight)
        + sorted_values[upper_index] * weight
    )


def _load_cached_test_cases(dataset: EvaluationDataset) -> list[LLMTestCase] | None:
    if not TEST_CASE_CACHE_PATH.exists():
        return None

    try:
        records = [
            json.loads(line)
            for line in TEST_CASE_CACHE_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        test_cases = [LLMTestCase(**record) for record in records]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

    return test_cases


def _save_test_cases(test_cases: list[LLMTestCase]) -> None:
    fields = (
        "input",
        "actual_output",
        "expected_output",
        "context",
        "retrieval_context",
        "metadata",
        "comments",
        "completion_time",
        "name",
    )
    content = "\n".join(
        json.dumps(
            {field: getattr(test_case, field) for field in fields},
            ensure_ascii=False,
        )
        for test_case in test_cases
    )
    TEST_CASE_CACHE_PATH.write_text(f"{content}\n", encoding="utf-8")


async def build_test_cases(
    dataset: EvaluationDataset,
    rerun: bool = False,
) -> list[LLMTestCase]:
    if not rerun:
        cached_test_cases = _load_cached_test_cases(dataset)
        if cached_test_cases is not None:
            print(f"Loaded {len(cached_test_cases)} cached test cases")
            return cached_test_cases

    graph = build_graph()
    test_cases = []

    for index, golden in enumerate(dataset.goldens, start=1):
        started_at = perf_counter()
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": golden.input}]}
        )
        completion_time = perf_counter() - started_at
        actual_output = str(result["messages"][-1].content)
        retrieval_context = [str(item) for item in result.get("context", [])]

        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=actual_output,
                expected_output=golden.expected_output,
                context=golden.context,
                retrieval_context=retrieval_context,
                metadata=golden.additional_metadata,
                comments=golden.comments,
                completion_time=completion_time,
                name=golden.name,
            )
        )
        print(f"Generated {index}/{len(dataset.goldens)}: {golden.name}")

    _save_test_cases(test_cases)
    print(f"Saved {len(test_cases)} test cases to {TEST_CASE_CACHE_PATH}")
    return test_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the interview agent.")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="regenerate test cases instead of loading the cached responses",
    )
    args = parser.parse_args()

    dataset = EvaluationDataset()
    dataset.add_goldens_from_jsonl_file(str(DATASET_PATH))
    test_cases = asyncio.run(build_test_cases(dataset, rerun=args.rerun))

    judge = OpenRouterModel(
        model=os.environ["EVALUATION_MODEL"],
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    metrics = [
        HallucinationMetric(model=judge, include_reason=False),
        AnswerRelevancyMetric(model=judge, include_reason=False),
        FaithfulnessMetric(model=judge, include_reason=False),
        ContextualPrecisionMetric(model=judge, include_reason=False),
        ContextualRecallMetric(model=judge, include_reason=False),
    ]

    
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        display_config=DisplayConfig(
            results_folder="eval/eval_result",
            file_type="md",
            file_output_dir="eval/eval_reports"),
    )
    print("")
    print("================================================================================")
    print("")
    latencies = [test_case.completion_time for test_case in test_cases]
    print(
        f"Latency: p50={_percentile(latencies, 0.50):.3f}s, "
        f"p95={_percentile(latencies, 0.95):.3f}s"
    )



if __name__ == "__main__":
    main()