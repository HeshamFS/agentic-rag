"""
Hybrid retriever combining dense and sparse retrieval.

Uses RRF fusion to combine vector similarity and BM25 results.
Research shows +18.5% MRR improvement over dense-only retrieval.
"""

from typing import Any

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import RetrievalResult
from agentic_rag.core.protocols import Embedder, VectorDB
from agentic_rag.retrieval.base import BaseRetriever
from agentic_rag.retrieval.dense import DenseRetriever
from agentic_rag.retrieval.fusion import RRFFusion
from agentic_rag.retrieval.sparse import SparseRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense and sparse retrieval.

    Benefits:
    - Dense captures semantic similarity
    - Sparse captures exact keyword matches
    - RRF fusion combines without score calibration
    - +18.5% MRR improvement documented in research
    """

    def __init__(
        self,
        embedder: Embedder,
        vectordb: VectorDB,
        settings: Settings | None = None,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedder: Embedding model for dense retrieval.
            vectordb: Vector database.
            settings: Configuration settings.
            sparse_weight: Weight for sparse retrieval (0-1).
            rrf_k: RRF ranking constant.
            bm25_k1: BM25 k1 parameter.
            bm25_b: BM25 b parameter.
        """
        self._settings = settings or get_settings()
        self._sparse_weight = sparse_weight

        # Initialize sub-retrievers
        self._dense = DenseRetriever(
            embedder=embedder,
            vectordb=vectordb,
            settings=self._settings,
        )
        self._sparse = SparseRetriever(
            vectordb=vectordb,
            settings=self._settings,
            k1=bm25_k1,
            b=bm25_b,
        )

        # Initialize fusion
        self._fusion = RRFFusion(k=rrf_k)

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        use_rrf: bool = True,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieve chunks using a hybrid dense + sparse approach.

        Combines:
        1. Dense retrieval: Captures semantic meaning using vector similarity.
        2. Sparse retrieval: Captures exact keyword matches using BM25.
        3. Fusion: Merges both result sets into a single ranked list.

        Args:
            query: The user search query.
            collection: Vector DB collection to search.
            top_k: Number of final results to return.
            use_rrf: If True, uses Reciprocal Rank Fusion (recommended).
                    If False, uses weighted linear combination.
            **kwargs: Additional parameters for sub-retrievers.

        Returns:
            Fused RetrievalResult containing deduplicated chunks and scores.
        """
        # Retrieve more from each to allow for deduplication
        retrieve_k = min(top_k * 2, 50)

        # Run both retrievers
        dense_result = await self._dense.retrieve(
            query=query,
            collection=collection,
            top_k=retrieve_k,
        )
        sparse_result = await self._sparse.retrieve(
            query=query,
            collection=collection,
            top_k=retrieve_k,
        )

        # Fuse results
        if use_rrf:
            # Use RRF (recommended)
            weights = [1.0 - self._sparse_weight, self._sparse_weight]
            result = self._fusion.fuse_with_weights(
                results=[dense_result, sparse_result],
                weights=weights,
                top_k=top_k,
            )
        else:
            # Use weighted linear combination
            from agentic_rag.retrieval.fusion import linear_combination_fusion

            weights = [1.0 - self._sparse_weight, self._sparse_weight]
            result = linear_combination_fusion(
                results=[dense_result, sparse_result],
                weights=weights,
                top_k=top_k,
            )

        # Update metadata
        result.retrieval_type = "hybrid"
        result.metadata.update(
            {
                "query": query,
                "dense_count": len(dense_result.chunks),
                "sparse_count": len(sparse_result.chunks),
                "sparse_weight": self._sparse_weight,
            }
        )

        return result

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
            queries: List of queries.
            collection: Collection to search.
            top_k: Number of results per query.
            **kwargs: Additional arguments.

        Returns:
            List of RetrievalResults.
        """
        results = []
        for query in queries:
            result = await self.retrieve(
                query=query,
                collection=collection,
                top_k=top_k,
                **kwargs,
            )
            results.append(result)
        return results

    def invalidate_sparse_cache(self, collection: str | None = None) -> None:
        """
        Invalidate sparse retriever cache.

        Args:
            collection: Collection to invalidate, or None for all.
        """
        self._sparse.invalidate_cache(collection)

    async def close(self) -> None:
        """Clean up resources."""
        await self._dense.close()
        await self._sparse.close()
