"""
Qwen3-Embedding implementation.

Uses the Qwen3-Embedding family of models which are state-of-the-art
open-source embeddings with:
- 70.58 MTEB score (#1 multilingual)
- Apache 2.0 license
- 100+ languages plus programming languages
- 32K token context
- Flexible dimensions (32-1024) via Matryoshka learning
"""

import asyncio
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from agentic_rag.config import Settings, get_settings
from agentic_rag.embeddings.base import BaseEmbedder, EmbeddingCache


class Qwen3Embedder(BaseEmbedder):
    """
    Qwen3-Embedding model implementation.

    Supports both the 0.6B and 8B variants:
    - Alibaba-NLP/gte-Qwen2-1.5B-instruct (default, good balance)
    - Alibaba-NLP/gte-Qwen2-7B-instruct (highest quality)

    Uses sentence-transformers for efficient inference.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        normalize_embeddings: bool = True,
        use_cache: bool = True,
        settings: Settings | None = None,
    ):
        """
        Initialize Qwen3 embedder.

        Args:
            model_name: HuggingFace model ID. Defaults to settings.
            device: Device to use (cuda, cpu, mps). Defaults to settings.
            batch_size: Batch size for encoding. Defaults to settings.
            max_length: Max sequence length. Defaults to 8192.
            normalize_embeddings: L2 normalize embeddings.
            use_cache: Enable in-memory embedding cache.
            settings: Settings instance. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.embedding_model
        self._device = device or self._settings.embedding_device
        self._batch_size = batch_size or self._settings.embedding_batch_size
        self._max_length_setting = max_length or self._settings.embedding_max_length
        self._normalize = normalize_embeddings
        self._cache = EmbeddingCache() if use_cache else None

        # Load model lazily
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    def _load_model(self) -> SentenceTransformer:
        """Load the model lazily."""
        import logging
        import time

        logger = logging.getLogger("agentic_rag.embedder")

        if self._model is None:
            # Detect available device
            actual_device = self._device
            if self._device == "cuda" and not torch.cuda.is_available():
                actual_device = "cpu"
                logger.warning(
                    "EMBEDDER: CUDA requested but not available - falling back to CPU (SLOW)"
                )
                logger.warning(
                    "EMBEDDER: For faster embedding, install CUDA or use a GPU-enabled environment"
                )
            elif self._device == "cuda":
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                logger.info(f"EMBEDDER: CUDA available - using GPU: {gpu_name}")

            logger.info(f"EMBEDDER: Loading model {self._model_name} on {actual_device}...")
            start = time.time()
            self._model = SentenceTransformer(
                self._model_name,
                device=actual_device,
                trust_remote_code=True,  # Required for Qwen models
            )
            # Set max length
            self._model.max_seq_length = self._max_length_setting
            # Get dimension from model
            self._dimension = self._model.get_sentence_embedding_dimension()

            load_time = time.time() - start
            logger.info(
                f"EMBEDDER: Model loaded in {load_time:.2f}s (dim={self._dimension}, device={actual_device})"
            )

            # Performance warning for CPU
            if actual_device == "cpu":
                logger.warning(
                    "EMBEDDER: Running on CPU - expect ~2-10 seconds per batch of 32 texts"
                )
                logger.warning("EMBEDDER: Consider using a smaller model or GPU for production")
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            # Load model to get dimension
            self._load_model()
        return self._dimension or 1536  # Default fallback

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    @property
    def max_length(self) -> int:
        """Maximum sequence length."""
        return self._max_length_setting

    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text into a vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Check cache first
        if self._cache is not None:
            cached = self._cache.get(text)
            if cached is not None:
                return cached

        # Embed using batch method for consistency
        embeddings = await self.embed_batch([text])
        embedding = embeddings[0]

        # Cache result
        if self._cache is not None:
            self._cache.set(text, embedding)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts into vectors.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Check cache for all texts
        if self._cache is not None:
            results: list[list[float] | None] = []
            uncached_texts: list[str] = []
            uncached_indices: list[int] = []

            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                results.append(cached)
                if cached is None:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # If all cached, return early
            if not uncached_texts:
                return [r for r in results if r is not None]

            # Embed uncached texts
            new_embeddings = await self._embed_uncached(uncached_texts)

            # Fill in results and cache
            for idx, embedding in zip(uncached_indices, new_embeddings, strict=False):
                results[idx] = embedding
                self._cache.set(texts[idx], embedding)

            return [r for r in results if r is not None]

        # No cache, embed all
        return await self._embed_uncached(texts)

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts without cache lookup.

        Runs in executor to avoid blocking async loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """
        Synchronous embedding operation.

        Called in executor from async methods.
        """
        import logging
        import time

        logger = logging.getLogger("agentic_rag.embedder")

        start = time.time()
        model = self._load_model()
        load_time = time.time() - start

        # Encode with batching
        total_chars = sum(len(t) for t in texts)
        logger.info(
            f"EMBEDDER: Encoding {len(texts)} texts ({total_chars} chars), batch_size={self._batch_size}"
        )

        start = time.time()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        encode_time = time.time() - start

        logger.info(f"EMBEDDER: Encoding done in {encode_time:.2f}s (model load: {load_time:.3f}s)")

        # Convert to list of lists
        return embeddings.tolist()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()


# =============================================================================
# Convenience factory
# =============================================================================


def create_embedder(
    model_variant: str = "default",
    **kwargs: Any,
) -> Qwen3Embedder:
    """
    Factory function to create Qwen3 embedder with preset configurations.

    Args:
        model_variant: One of "default", "small", "large".
        **kwargs: Additional arguments passed to Qwen3Embedder.

    Returns:
        Configured Qwen3Embedder instance.
    """
    model_map = {
        "default": "Qwen/Qwen3-Embedding-0.6B",  # From .env, official Qwen3
        "small": "Qwen/Qwen3-Embedding-0.6B",  # Best small option
        "medium": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "large": "Qwen/Qwen3-Embedding-8B",  # Highest quality
    }

    model_name = model_map.get(model_variant, model_variant)
    return Qwen3Embedder(model_name=model_name, **kwargs)
