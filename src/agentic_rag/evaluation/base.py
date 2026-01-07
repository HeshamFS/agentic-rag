"""
Base evaluation interface and utilities.

Provides common interface for all evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.core.models import Chunk


class EvaluationResult(BaseModel):
    """Result from an evaluation metric."""

    metric_name: str = Field(description="Name of the metric")
    score: float = Field(ge=0.0, le=1.0, description="Score between 0 and 1")
    details: dict[str, Any] = Field(default_factory=dict, description="Detailed breakdown")
    reasoning: str = Field(default="", description="Explanation of the score")


class EvaluationSuite(BaseModel):
    """Collection of evaluation results."""

    query: str = Field(description="Original query")
    response: str = Field(description="Generated response")
    results: list[EvaluationResult] = Field(
        default_factory=list, description="Individual metric results"
    )
    overall_score: float = Field(ge=0.0, le=1.0, description="Aggregate score")
    pass_threshold: bool = Field(description="Whether evaluation passed")


class BaseEvaluator(ABC):
    """
    Base class for all evaluation metrics.

    Evaluators assess different aspects of RAG pipeline outputs.
    """

    def __init__(self, name: str):
        """
        Initialize evaluator.

        Args:
            name: Metric name.
        """
        self.name = name

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a response.

        Args:
            query: User query.
            response: Generated response.
            contexts: Retrieved context chunks.
            ground_truth: Optional ground truth answer.

        Returns:
            Evaluation result.
        """
        ...

    async def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple samples.

        Args:
            samples: List of dicts with query, response, contexts, ground_truth.

        Returns:
            List of evaluation results.
        """
        results = []
        for sample in samples:
            result = await self.evaluate(
                query=sample["query"],
                response=sample["response"],
                contexts=sample.get("contexts", []),
                ground_truth=sample.get("ground_truth"),
            )
            results.append(result)
        return results


class CompositeEvaluator:
    """
    Combines multiple evaluators into one.

    Runs all evaluators and aggregates results.
    """

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        weights: dict[str, float] | None = None,
        threshold: float = 0.7,
    ):
        """
        Initialize composite evaluator.

        Args:
            evaluators: List of evaluators to run.
            weights: Optional weights for each metric (by name).
            threshold: Minimum score to pass.
        """
        self.evaluators = evaluators
        self.weights = weights or {}
        self.threshold = threshold

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationSuite:
        """
        Run all evaluators.

        Args:
            query: User query.
            response: Generated response.
            contexts: Retrieved contexts.
            ground_truth: Optional ground truth.

        Returns:
            Complete evaluation suite.
        """
        results = []

        for evaluator in self.evaluators:
            result = await evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts,
                ground_truth=ground_truth,
            )
            results.append(result)

        # Calculate weighted overall score
        total_weight = 0.0
        weighted_sum = 0.0

        for result in results:
            weight = self.weights.get(result.metric_name, 1.0)
            weighted_sum += result.score * weight
            total_weight += weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        return EvaluationSuite(
            query=query,
            response=response,
            results=results,
            overall_score=overall,
            pass_threshold=overall >= self.threshold,
        )


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.

    Args:
        precision: Precision score.
        recall: Recall score.

    Returns:
        F1 score.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_overlap(set1: set[str], set2: set[str]) -> float:
    """
    Calculate overlap between two sets.

    Args:
        set1: First set.
        set2: Second set.

    Returns:
        Overlap ratio.
    """
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0
