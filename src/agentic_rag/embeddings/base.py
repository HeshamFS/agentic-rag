"""
Base embedder class and utilities.

Provides the abstract base for all embedding implementations
and common utilities for batch processing.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding models.

    Subclasses must implement embed_text and embed_batch methods.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    def max_length(self) -> int:
        """Maximum sequence length. Override in subclasses."""
        return 8192

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text into a vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts into vectors.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    async def embed_with_context(
        self,
        text: str,
        context_header: str | None = None,
    ) -> list[float]:
        """
        Embed text with optional context header prepended.

        Used for contextual retrieval where we prepend
        context like "This chunk is from section X..."

        Args:
            text: The main text to embed.
            context_header: Optional context to prepend.

        Returns:
            Embedding vector.
        """
        full_text = f"{context_header}\n\n{text}" if context_header else text
        return await self.embed_text(full_text)

    def normalize(self, embedding: list[float]) -> list[float]:
        """
        L2 normalize an embedding vector.

        Args:
            embedding: Raw embedding vector.

        Returns:
            Normalized embedding vector.
        """
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def cosine_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score (-1 to 1).
        """
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)

        dot = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings.

    Useful for avoiding redundant embedding calls
    within a single pipeline run.
    """

    def __init__(self, max_size: int = 10000):
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_size

    def get(self, text: str) -> list[float] | None:
        """Get cached embedding for text."""
        return self._cache.get(text)

    def set(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        if len(self._cache) >= self._max_size:
            # Simple eviction: remove first item
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[text] = embedding

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
