"""
Base reranking protocol and interfaces.

Rerankers improve retrieval quality by reordering chunks based on
relevance to the query using cross-encoder models.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.core.models import Chunk


class RerankResult(BaseModel):
    """Result of reranking operation."""

    chunks: list[Chunk] = Field(description="Reranked chunks in order of relevance")
    scores: list[float] = Field(description="Relevance scores for each chunk")
    original_indices: list[int] = Field(description="Original indices before reranking")


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.

    Rerankers take a query and list of chunks, returning them
    reordered by relevance with scores.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the reranker model name."""
        ...

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RerankResult:
        """
        Rerank chunks by relevance to query.

        Args:
            query: The search query.
            chunks: List of chunks to rerank.
            top_k: Number of top chunks to return (None = all).
            **kwargs: Additional parameters.

        Returns:
            RerankResult with reordered chunks and scores.
        """
        ...

    async def rerank_with_threshold(
        self,
        query: str,
        chunks: list[Chunk],
        threshold: float = 0.5,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RerankResult:
        """
        Rerank and filter by score threshold.

        Args:
            query: The search query.
            chunks: List of chunks to rerank.
            threshold: Minimum score threshold.
            top_k: Max chunks to return.
            **kwargs: Additional parameters.

        Returns:
            RerankResult with filtered chunks above threshold.
        """
        result = await self.rerank(query, chunks, top_k=None, **kwargs)

        # Filter by threshold
        filtered_chunks = []
        filtered_scores = []
        filtered_indices = []

        for chunk, score, idx in zip(
            result.chunks, result.scores, result.original_indices, strict=False
        ):
            if score >= threshold:
                filtered_chunks.append(chunk)
                filtered_scores.append(score)
                filtered_indices.append(idx)

        # Apply top_k after filtering
        if top_k is not None:
            filtered_chunks = filtered_chunks[:top_k]
            filtered_scores = filtered_scores[:top_k]
            filtered_indices = filtered_indices[:top_k]

        return RerankResult(
            chunks=filtered_chunks,
            scores=filtered_scores,
            original_indices=filtered_indices,
        )
