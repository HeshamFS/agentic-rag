"""
Standard benchmarks for RAG evaluation.

Provides datasets and evaluation harnesses for
measuring RAG pipeline performance.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.config import Settings, get_settings
from agentic_rag.evaluation.ragas import RAGASEvaluator


class BenchmarkQuestion(BaseModel):
    """A benchmark question."""

    question: str
    ground_truth: str
    context: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Result for a single benchmark question."""

    question: str
    ground_truth: str
    predicted: str
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    latency_ms: float = 0.0


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run."""

    total_questions: int = 0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_latency_ms: float = 0.0
    results: list[BenchmarkResult] = field(default_factory=list)


class RAGBenchmark:
    """
    RAG benchmark harness.

    Evaluates pipelines on standard QA datasets.
    """

    def __init__(
        self,
        pipeline: Any,  # BasePipeline
        evaluator: RAGASEvaluator | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize benchmark.

        Args:
            pipeline: Pipeline to evaluate.
            evaluator: RAGAS evaluator.
            settings: Settings instance.
        """
        self._pipeline = pipeline
        self._settings = settings or get_settings()
        self._evaluator = evaluator

    async def run(
        self,
        questions: list[BenchmarkQuestion],
        collection: str,
    ) -> BenchmarkSummary:
        """
        Run benchmark on questions.

        Args:
            questions: Benchmark questions.
            collection: Collection to query.

        Returns:
            BenchmarkSummary with results.
        """
        import time

        results: list[BenchmarkResult] = []

        for q in questions:
            start_time = time.perf_counter()

            # Query pipeline
            result = await self._pipeline.query(
                question=q.question,
                collection=collection,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Evaluate with RAGAS if available
            if self._evaluator:
                ragas_result = await self._evaluator.evaluate(
                    query=q.question,
                    response=result.response,
                    chunks=result.sources,
                    ground_truth=q.ground_truth,
                )

                bench_result = BenchmarkResult(
                    question=q.question,
                    ground_truth=q.ground_truth,
                    predicted=result.response,
                    context_precision=ragas_result.context_precision,
                    context_recall=ragas_result.context_recall,
                    faithfulness=ragas_result.faithfulness,
                    answer_relevancy=ragas_result.answer_relevancy,
                    latency_ms=latency_ms,
                )
            else:
                bench_result = BenchmarkResult(
                    question=q.question,
                    ground_truth=q.ground_truth,
                    predicted=result.response,
                    latency_ms=latency_ms,
                )

            results.append(bench_result)

        # Calculate summary
        n = len(results)
        summary = BenchmarkSummary(
            total_questions=n,
            avg_context_precision=sum(r.context_precision for r in results) / n if n else 0,
            avg_context_recall=sum(r.context_recall for r in results) / n if n else 0,
            avg_faithfulness=sum(r.faithfulness for r in results) / n if n else 0,
            avg_answer_relevancy=sum(r.answer_relevancy for r in results) / n if n else 0,
            avg_latency_ms=sum(r.latency_ms for r in results) / n if n else 0,
            results=results,
        )

        return summary


# Standard benchmark datasets
class HotPotQA:
    """HotPotQA multi-hop QA benchmark."""

    @staticmethod
    def load_sample(n: int = 100) -> list[BenchmarkQuestion]:
        """Load sample questions."""
        # This would load from HuggingFace datasets
        # For now, return placeholder
        return [
            BenchmarkQuestion(
                question="What is the capital of France?",
                ground_truth="Paris",
                context=["Paris is the capital of France."],
            )
        ]


class NaturalQuestions:
    """Natural Questions benchmark."""

    @staticmethod
    def load_sample(n: int = 100) -> list[BenchmarkQuestion]:
        """Load sample questions."""
        return [
            BenchmarkQuestion(
                question="Who wrote Romeo and Juliet?",
                ground_truth="William Shakespeare",
                context=["Romeo and Juliet was written by William Shakespeare."],
            )
        ]


class TriviaQA:
    """TriviaQA benchmark."""

    @staticmethod
    def load_sample(n: int = 100) -> list[BenchmarkQuestion]:
        """Load sample questions."""
        return [
            BenchmarkQuestion(
                question="What is the largest planet in our solar system?",
                ground_truth="Jupiter",
                context=["Jupiter is the largest planet in our solar system."],
            )
        ]


def load_benchmark(name: str, n: int = 100) -> list[BenchmarkQuestion]:
    """
    Load a benchmark dataset.

    Args:
        name: Benchmark name (hotpotqa, nq, triviaqa).
        n: Number of questions.

    Returns:
        List of benchmark questions.
    """
    benchmarks = {
        "hotpotqa": HotPotQA.load_sample,
        "nq": NaturalQuestions.load_sample,
        "triviaqa": TriviaQA.load_sample,
    }

    loader = benchmarks.get(name.lower())
    if not loader:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(benchmarks.keys())}")

    return loader(n)
