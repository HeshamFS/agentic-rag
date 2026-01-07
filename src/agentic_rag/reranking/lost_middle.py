"""
Lost-in-the-Middle mitigation.

Research shows LLMs pay less attention to content in the middle of long contexts.
This module reorders chunks to place important content at the start and end.

Reference: "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)
"""

from agentic_rag.core.models import Chunk
from agentic_rag.reranking.base import BaseReranker, RerankResult


def reorder_for_attention(
    chunks: list[Chunk],
    scores: list[float] | None = None,
) -> tuple[list[Chunk], list[float], list[int]]:
    """
    Reorder chunks to mitigate lost-in-the-middle effect.

    Places chunks in alternating positions:
    - 1st best → position 0 (start)
    - 2nd best → position -1 (end)
    - 3rd best → position 1 (near start)
    - 4th best → position -2 (near end)
    - ...

    This ensures the most relevant content is at positions where
    LLMs pay the most attention.

    Args:
        chunks: List of chunks (assumed pre-sorted by relevance).
        scores: Optional scores for each chunk.

    Returns:
        Tuple of (reordered_chunks, reordered_scores, original_indices).
    """
    if not chunks:
        return [], [], []

    n = len(chunks)
    scores = scores or [1.0 - (i / n) for i in range(n)]

    # Create indexed list
    indexed = list(range(n))

    # Result arrays
    result_chunks: list[Chunk | None] = [None] * n
    result_scores: list[float | None] = [None] * n
    result_indices: list[int | None] = [None] * n

    # Alternate between start and end
    start_pos = 0
    end_pos = n - 1

    for i, idx in enumerate(indexed):
        if i % 2 == 0:
            # Even indices go to start
            result_chunks[start_pos] = chunks[idx]
            result_scores[start_pos] = scores[idx]
            result_indices[start_pos] = idx
            start_pos += 1
        else:
            # Odd indices go to end
            result_chunks[end_pos] = chunks[idx]
            result_scores[end_pos] = scores[idx]
            result_indices[end_pos] = idx
            end_pos -= 1

    return (
        [c for c in result_chunks if c is not None],
        [s for s in result_scores if s is not None],
        [i for i in result_indices if i is not None],
    )


class LostInMiddleReorderer(BaseReranker):
    """
    Reorderer that mitigates the lost-in-the-middle effect.

    Can wrap any reranker to apply attention-aware ordering after reranking.
    """

    def __init__(self, base_reranker: BaseReranker | None = None):
        """
        Initialize reorderer.

        Args:
            base_reranker: Optional base reranker to wrap.
                          If None, assumes input is pre-sorted.
        """
        self._base_reranker = base_reranker

    @property
    def model_name(self) -> str:
        if self._base_reranker:
            return f"lost_middle_wrapper({self._base_reranker.model_name})"
        return "lost_middle_reorderer"

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
        **kwargs,
    ) -> RerankResult:
        """
        Rerank and reorder for attention.

        Args:
            query: The search query.
            chunks: List of chunks to process.
            top_k: Number of chunks to return.
            **kwargs: Additional parameters for base reranker.

        Returns:
            RerankResult with attention-optimized ordering.
        """
        if not chunks:
            return RerankResult(chunks=[], scores=[], original_indices=[])

        # First, get reranked results
        if self._base_reranker:
            result = await self._base_reranker.rerank(query, chunks, top_k=top_k, **kwargs)
            ranked_chunks = result.chunks
            ranked_scores = result.scores
        else:
            # Assume already sorted by relevance
            ranked_chunks = chunks[:top_k] if top_k else chunks
            ranked_scores = [1.0 - (i / len(ranked_chunks)) for i in range(len(ranked_chunks))]

        # Apply lost-in-middle reordering
        reordered_chunks, reordered_scores, reorder_indices = reorder_for_attention(
            ranked_chunks, ranked_scores
        )

        # Map back to original indices
        if self._base_reranker:
            original_indices = [result.original_indices[i] for i in reorder_indices]
        else:
            original_indices = reorder_indices

        return RerankResult(
            chunks=reordered_chunks,
            scores=reordered_scores,
            original_indices=original_indices,
        )


def apply_lost_in_middle(result: RerankResult) -> RerankResult:
    """
    Apply lost-in-middle reordering to an existing RerankResult.

    Convenience function for post-processing.

    Args:
        result: RerankResult to reorder.

    Returns:
        New RerankResult with attention-optimized ordering.
    """
    if not result.chunks:
        return result

    reordered_chunks, reordered_scores, reorder_indices = reorder_for_attention(
        result.chunks, result.scores
    )

    original_indices = [result.original_indices[i] for i in reorder_indices]

    return RerankResult(
        chunks=reordered_chunks,
        scores=reordered_scores,
        original_indices=original_indices,
    )


class InterleavedReorderer:
    """
    Alternative reordering strategy that interleaves by source.

    Useful when chunks come from multiple documents to ensure
    diversity in the context window.
    """

    @staticmethod
    def reorder_by_source(
        chunks: list[Chunk],
        scores: list[float] | None = None,
    ) -> tuple[list[Chunk], list[float], list[int]]:
        """
        Reorder chunks to interleave different sources.

        Groups chunks by source, then interleaves them while
        maintaining relative ranking within each source.

        Args:
            chunks: List of chunks with source metadata.
            scores: Optional relevance scores.

        Returns:
            Tuple of (reordered_chunks, reordered_scores, original_indices).
        """
        if not chunks:
            return [], [], []

        n = len(chunks)
        scores = scores or [1.0 - (i / n) for i in range(n)]

        # Group by source
        source_groups: dict[str, list[tuple[int, Chunk, float]]] = {}
        for i, (chunk, score) in enumerate(zip(chunks, scores, strict=False)):
            source = chunk.metadata.get("source", "unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append((i, chunk, score))

        # Interleave from each source
        result_chunks = []
        result_scores = []
        result_indices = []

        # Round-robin through sources
        source_iters = {k: iter(v) for k, v in source_groups.items()}
        sources = list(source_groups.keys())

        while source_iters:
            for source in list(sources):
                try:
                    idx, chunk, score = next(source_iters[source])
                    result_chunks.append(chunk)
                    result_scores.append(score)
                    result_indices.append(idx)
                except StopIteration:
                    del source_iters[source]
                    sources.remove(source)

        return result_chunks, result_scores, result_indices
