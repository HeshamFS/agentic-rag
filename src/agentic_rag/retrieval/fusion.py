"""
Reciprocal Rank Fusion (RRF) for combining retrieval results.

Combines multiple retrieval results without requiring score normalization.
Research shows +18.5% MRR improvement over single retriever.
"""

from agentic_rag.core.models import Chunk, RetrievalResult


class RRFFusion:
    """
    Reciprocal Rank Fusion for combining retrieval results.

    RRF is a robust fusion method that:
    - Doesn't require score normalization
    - Works well with heterogeneous retrievers
    - Provides consistent improvements over single retrievers

    Formula: RRF_score(d) = Σ 1/(k + rank_i)
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.

        Args:
            k: Ranking constant (typically 60).
                Higher k reduces impact of top ranks.
        """
        self.k = k

    def fuse(
        self,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Fuse multiple retrieval results using RRF.

        Args:
            results: List of RetrievalResults to fuse.
            top_k: Number of final results. Defaults to max of inputs.

        Returns:
            Fused RetrievalResult.
        """
        if not results:
            return RetrievalResult(
                chunks=[],
                scores=[],
                retrieval_type="rrf_fusion",
            )

        # Default top_k to max of all inputs
        if top_k is None:
            top_k = max(len(r.chunks) for r in results)

        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for result in results:
            for rank, chunk in enumerate(result.chunks, start=1):
                # RRF formula
                rrf_score = 1.0 / (self.k + rank)

                if chunk.id in rrf_scores:
                    rrf_scores[chunk.id] += rrf_score
                else:
                    rrf_scores[chunk.id] = rrf_score
                    chunk_map[chunk.id] = chunk

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build result
        fused_chunks = []
        fused_scores = []
        for chunk_id in sorted_ids[:top_k]:
            fused_chunks.append(chunk_map[chunk_id])
            fused_scores.append(rrf_scores[chunk_id])

        return RetrievalResult(
            chunks=fused_chunks,
            scores=fused_scores,
            retrieval_type="rrf_fusion",
            metadata={
                "k": self.k,
                "num_sources": len(results),
                "source_types": [r.retrieval_type for r in results],
            },
        )

    def fuse_with_weights(
        self,
        results: list[RetrievalResult],
        weights: list[float],
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Fuse results with custom weights per retriever.

        Args:
            results: List of RetrievalResults.
            weights: Weight for each retriever (should sum to 1.0).
            top_k: Number of final results.

        Returns:
            Weighted fused RetrievalResult.
        """
        if len(results) != len(weights):
            raise ValueError("Number of results must match number of weights")

        if not results:
            return RetrievalResult(
                chunks=[],
                scores=[],
                retrieval_type="weighted_rrf_fusion",
            )

        if top_k is None:
            top_k = max(len(r.chunks) for r in results)

        # Calculate weighted RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for result, weight in zip(results, weights, strict=False):
            for rank, chunk in enumerate(result.chunks, start=1):
                # Weighted RRF
                rrf_score = weight * (1.0 / (self.k + rank))

                if chunk.id in rrf_scores:
                    rrf_scores[chunk.id] += rrf_score
                else:
                    rrf_scores[chunk.id] = rrf_score
                    chunk_map[chunk.id] = chunk

        # Sort by score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_chunks = []
        fused_scores = []
        for chunk_id in sorted_ids[:top_k]:
            fused_chunks.append(chunk_map[chunk_id])
            fused_scores.append(rrf_scores[chunk_id])

        return RetrievalResult(
            chunks=fused_chunks,
            scores=fused_scores,
            retrieval_type="weighted_rrf_fusion",
            metadata={
                "k": self.k,
                "weights": weights,
                "source_types": [r.retrieval_type for r in results],
            },
        )


def linear_combination_fusion(
    results: list[RetrievalResult],
    weights: list[float] | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    """
    Simple linear combination of normalized scores.

    Args:
        results: List of RetrievalResults.
        weights: Weights for each result (default: equal).
        top_k: Number of results.

    Returns:
        Fused RetrievalResult.
    """
    if not results:
        return RetrievalResult(
            chunks=[],
            scores=[],
            retrieval_type="linear_fusion",
        )

    if weights is None:
        weights = [1.0 / len(results)] * len(results)

    if top_k is None:
        top_k = max(len(r.chunks) for r in results)

    # Normalize scores for each result
    combined_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for result, weight in zip(results, weights, strict=False):
        if not result.scores:
            continue

        # Min-max normalize
        min_score = min(result.scores)
        max_score = max(result.scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        for chunk, score in zip(result.chunks, result.scores, strict=False):
            normalized = (score - min_score) / score_range
            weighted = weight * normalized

            if chunk.id in combined_scores:
                combined_scores[chunk.id] += weighted
            else:
                combined_scores[chunk.id] = weighted
                chunk_map[chunk.id] = chunk

    # Sort and return
    sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

    fused_chunks = []
    fused_scores = []
    for chunk_id in sorted_ids[:top_k]:
        fused_chunks.append(chunk_map[chunk_id])
        fused_scores.append(combined_scores[chunk_id])

    return RetrievalResult(
        chunks=fused_chunks,
        scores=fused_scores,
        retrieval_type="linear_fusion",
        metadata={"weights": weights},
    )


def reciprocal_rank_fusion(
    results: list[RetrievalResult],
    k: int = 60,
    top_k: int | None = None,
) -> RetrievalResult:
    """
    Convenience function for RRF fusion.

    Args:
        results: List of RetrievalResults to fuse.
        k: RRF constant (default 60).
        top_k: Number of results.

    Returns:
        Fused RetrievalResult.
    """
    rrf = RRFFusion(k=k)
    return rrf.fuse(results, top_k=top_k)
