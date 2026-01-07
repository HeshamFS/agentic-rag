"""
Jina Reranker implementation.

Uses Jina AI's reranker models for high-quality relevance scoring.
Supports jina-reranker-v2-base-multilingual (recommended).
"""

import asyncio
from typing import Any

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk
from agentic_rag.reranking.base import BaseReranker, RerankResult


class JinaReranker(BaseReranker):
    """
    Jina Reranker using sentence-transformers CrossEncoder.

    Supports:
    - jinaai/jina-reranker-v2-base-multilingual (recommended, 278M params)
    - jinaai/jina-reranker-v1-base-en (English only)
    - jinaai/jina-reranker-v1-turbo-en (faster, English)
    """

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        settings: Settings | None = None,
    ):
        """
        Initialize Jina Reranker.

        Args:
            model: Model ID. Defaults to settings.reranker_model.
            device: Device (cuda, cpu, mps). Defaults to settings.
            batch_size: Batch size for inference.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._model_name = model or self._settings.reranker_model
        self._device = device or self._settings.reranker_device
        self._batch_size = batch_size
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                trust_remote_code=True,
            )
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RerankResult:
        """
        Rerank chunks using Jina cross-encoder.

        Args:
            query: The search query.
            chunks: List of chunks to rerank.
            top_k: Number of top chunks to return.
            **kwargs: Additional parameters.

        Returns:
            RerankResult with reordered chunks.
        """
        if not chunks:
            return RerankResult(chunks=[], scores=[], original_indices=[])

        # Load model
        model = self._load_model()

        # Prepare pairs for scoring
        pairs = [(query, chunk.content) for chunk in chunks]

        # Score in executor (model inference is CPU/GPU bound)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(pairs, batch_size=self._batch_size).tolist(),
        )

        # Create indexed scores and sort by score descending
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        # Build result
        reranked_chunks = []
        reranked_scores = []
        original_indices = []

        for orig_idx, score in indexed_scores:
            reranked_chunks.append(chunks[orig_idx])
            reranked_scores.append(score)
            original_indices.append(orig_idx)

        return RerankResult(
            chunks=reranked_chunks,
            scores=reranked_scores,
            original_indices=original_indices,
        )


class JinaRerankerV3(BaseReranker):
    """
    Jina Reranker V3 using the Jina API.

    For cloud-based reranking with the latest Jina models.
    Requires JINA_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jina-reranker-v2-base-multilingual",
        settings: Settings | None = None,
    ):
        """
        Initialize Jina API Reranker.

        Args:
            api_key: Jina API key.
            model: Model ID.
            settings: Settings instance.
        """
        import os

        self._api_key = api_key or os.getenv("JINA_API_KEY")
        self._model_name = model
        self._settings = settings or get_settings()

        if not self._api_key:
            raise ValueError("JINA_API_KEY not set")

    @property
    def model_name(self) -> str:
        return self._model_name

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RerankResult:
        """
        Rerank chunks using Jina API.

        Args:
            query: The search query.
            chunks: List of chunks to rerank.
            top_k: Number of top chunks to return.
            **kwargs: Additional parameters.

        Returns:
            RerankResult with reordered chunks.
        """
        import httpx

        if not chunks:
            return RerankResult(chunks=[], scores=[], original_indices=[])

        # Prepare documents
        documents = [chunk.content for chunk in chunks]

        # Call Jina API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.jina.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k or len(documents),
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        # Parse results
        results = data.get("results", [])

        reranked_chunks = []
        reranked_scores = []
        original_indices = []

        for item in results:
            idx = item["index"]
            score = item["relevance_score"]
            reranked_chunks.append(chunks[idx])
            reranked_scores.append(score)
            original_indices.append(idx)

        return RerankResult(
            chunks=reranked_chunks,
            scores=reranked_scores,
            original_indices=original_indices,
        )
