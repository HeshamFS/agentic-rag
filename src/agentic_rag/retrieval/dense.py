"""
Dense retriever using vector similarity search.

Performs semantic similarity search using embedding vectors.
"""

from typing import Any

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import RetrievalResult
from agentic_rag.core.protocols import Embedder, VectorDB
from agentic_rag.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using vector similarity.

    Uses embeddings to find semantically similar chunks.
    """

    def __init__(
        self,
        embedder: Embedder,
        vectordb: VectorDB,
        settings: Settings | None = None,
    ):
        """
        Initialize dense retriever.

        Args:
            embedder: Embedding model.
            vectordb: Vector database.
            settings: Configuration settings.
        """
        self._embedder = embedder
        self._vectordb = vectordb
        self._settings = settings or get_settings()

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieve chunks using dense vector similarity search.

        1. Embeds the user query using the configured embedding model.
        2. Performs a nearest neighbor search in Qdrant for matching vectors.
        3. Returns the top-k most similar chunks.

        Args:
            query: The user search query.
            collection: Vector DB collection to search.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score (0.0-1.0) to include a result.
            **kwargs: Additional parameters passed to the vector database.

        Returns:
            RetrievalResult containing the ranked chunks and their similarity scores.
        """
        # Embed query
        query_embedding = await self._embedder.embed_text(query)

        # Search vector database
        results = await self._vectordb.search(
            collection=collection,
            query_vector=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        chunks = []
        scores = []
        for chunk, score in results:
            chunks.append(chunk)
            scores.append(score)

        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            retrieval_type="dense",
            metadata={"query": query, "top_k": top_k},
        )

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
        # Embed all queries at once
        query_embeddings = await self._embedder.embed_batch(queries)

        results = []
        for query, embedding in zip(queries, query_embeddings, strict=False):
            search_results = await self._vectordb.search(
                collection=collection,
                query_vector=embedding,
                top_k=top_k,
            )

            chunks = []
            scores = []
            for chunk, score in search_results:
                chunks.append(chunk)
                scores.append(score)

            results.append(
                RetrievalResult(
                    chunks=chunks,
                    scores=scores,
                    retrieval_type="dense",
                    metadata={"query": query, "top_k": top_k},
                )
            )

        return results

    async def close(self) -> None:
        """Clean up resources."""
        await self._vectordb.close()
