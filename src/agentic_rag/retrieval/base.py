"""
Base retrieval protocol and utilities.

Defines the interface for all retrieval implementations.
"""

from abc import ABC, abstractmethod
from typing import Any

from agentic_rag.core.models import Chunk, RetrievalResult


class BaseRetriever(ABC):
    """
    Base class for all retrievers.

    Defines the common interface for retrieval operations.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query.
            collection: Vector DB collection name.
            top_k: Number of results to return.
            **kwargs: Additional retriever-specific arguments.

        Returns:
            RetrievalResult with chunks and scores.
        """
        ...

    @abstractmethod
    async def batch_retrieve(
        self,
        queries: list[str],
        collection: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of search queries.
            collection: Vector DB collection name.
            top_k: Number of results per query.
            **kwargs: Additional retriever-specific arguments.

        Returns:
            List of RetrievalResults.
        """
        ...

    async def close(self) -> None:
        """Clean up resources."""
        pass


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Normalize scores to [0, 1] range.

    Args:
        scores: Raw scores.

    Returns:
        Normalized scores.
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def deduplicate_chunks(
    chunks: list[Chunk],
    scores: list[float],
) -> tuple[list[Chunk], list[float]]:
    """
    Remove duplicate chunks, keeping highest scores.

    Args:
        chunks: List of chunks.
        scores: Corresponding scores.

    Returns:
        Deduplicated chunks and scores.
    """
    seen: dict[str, tuple[Chunk, float]] = {}

    for chunk, score in zip(chunks, scores, strict=False):
        if chunk.id not in seen or score > seen[chunk.id][1]:
            seen[chunk.id] = (chunk, score)

    result_chunks = []
    result_scores = []
    for chunk_id in seen:
        chunk, score = seen[chunk_id]
        result_chunks.append(chunk)
        result_scores.append(score)

    # Sort by score descending
    paired = sorted(zip(result_scores, result_chunks, strict=False), reverse=True)
    if paired:
        result_scores, result_chunks = zip(*paired, strict=False)  # type: ignore
        return list(result_chunks), list(result_scores)
    return [], []
