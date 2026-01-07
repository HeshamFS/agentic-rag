"""
Cross-Encoder reranker implementation.

Supports any HuggingFace cross-encoder model via sentence-transformers.
"""

import asyncio
from typing import Any

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk
from agentic_rag.reranking.base import BaseReranker, RerankResult


class CrossEncoderReranker(BaseReranker):
    """
    Generic Cross-Encoder reranker using sentence-transformers.

    Supports any HuggingFace cross-encoder model:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, English)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced)
    - BAAI/bge-reranker-v2-m3 (multilingual, high quality)
    - mixedbread-ai/mxbai-rerank-large-v1 (high quality)
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        settings: Settings | None = None,
    ):
        """
        Initialize Cross-Encoder reranker.

        Args:
            model: HuggingFace model ID.
            device: Device (cuda, cpu, mps).
            batch_size: Batch size for inference.
            max_length: Max sequence length.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._model_name = model
        self._device = device or self._settings.reranker_device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
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
        Rerank chunks using cross-encoder.

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

        # Prepare pairs
        pairs = [(query, chunk.content) for chunk in chunks]

        # Score in executor
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            ).tolist(),
        )

        # Sort by score descending
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


class BGEReranker(CrossEncoderReranker):
    """
    BGE Reranker using BAAI's reranker models.

    Models:
    - BAAI/bge-reranker-v2-m3 (multilingual, recommended)
    - BAAI/bge-reranker-large (English, high quality)
    - BAAI/bge-reranker-base (English, balanced)
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 32,
        settings: Settings | None = None,
    ):
        super().__init__(
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=512,
            settings=settings,
        )


class MxbaiReranker(CrossEncoderReranker):
    """
    Mixedbread AI reranker.

    Models:
    - mixedbread-ai/mxbai-rerank-large-v1 (high quality)
    - mixedbread-ai/mxbai-rerank-base-v1 (balanced)
    """

    def __init__(
        self,
        model: str = "mixedbread-ai/mxbai-rerank-large-v1",
        device: str | None = None,
        batch_size: int = 32,
        settings: Settings | None = None,
    ):
        super().__init__(
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=512,
            settings=settings,
        )
