"""
Embedding cache for efficient embedding reuse.

Caches computed embeddings to avoid recomputation
for repeated text inputs.
"""

import hashlib
import time
from pathlib import Path
from typing import Any

from agentic_rag.config import Settings, get_settings


class EmbeddingCache:
    """
    In-memory embedding cache.

    Caches text → embedding mappings using content hashing.
    """

    def __init__(
        self,
        max_entries: int = 100000,
        ttl_seconds: int | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize embedding cache.

        Args:
            max_entries: Maximum cache entries.
            ttl_seconds: Entry TTL (None = no expiry).
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[list[float], float]] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, model: str | None = None) -> str:
        """Create cache key from text and model."""
        content = f"{model or 'default'}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if entry is expired."""
        if self._ttl is None:
            return False
        return time.time() - timestamp > self._ttl

    def _evict_if_needed(self) -> None:
        """Evict old entries if cache is full."""
        if len(self._cache) >= self._max_entries:
            # Remove 10% of oldest entries
            entries = sorted(
                self._cache.items(),
                key=lambda x: x[1][1],  # Sort by timestamp
            )
            to_remove = max(1, len(entries) // 10)
            for key, _ in entries[:to_remove]:
                del self._cache[key]

    def get(self, text: str, model: str | None = None) -> list[float] | None:
        """
        Get cached embedding.

        Args:
            text: Input text.
            model: Model identifier.

        Returns:
            Cached embedding or None.
        """
        key = self._make_key(text, model)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        embedding, timestamp = entry
        if self._is_expired(timestamp):
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return embedding

    def get_batch(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> tuple[list[list[float] | None], list[int]]:
        """
        Get cached embeddings for batch.

        Args:
            texts: List of input texts.
            model: Model identifier.

        Returns:
            Tuple of (embeddings, uncached_indices).
            Embeddings is same length as texts, with None for uncached.
        """
        embeddings: list[list[float] | None] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            emb = self.get(text, model)
            embeddings.append(emb)
            if emb is None:
                uncached_indices.append(i)

        return embeddings, uncached_indices

    def set(self, text: str, embedding: list[float], model: str | None = None) -> None:
        """
        Cache an embedding.

        Args:
            text: Input text.
            embedding: Computed embedding.
            model: Model identifier.
        """
        self._evict_if_needed()
        key = self._make_key(text, model)
        self._cache[key] = (embedding, time.time())

    def set_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        model: str | None = None,
    ) -> None:
        """
        Cache a batch of embeddings.

        Args:
            texts: Input texts.
            embeddings: Computed embeddings.
            model: Model identifier.
        """
        for text, embedding in zip(texts, embeddings, strict=False):
            self.set(text, embedding, model)

    def invalidate(self, text: str, model: str | None = None) -> bool:
        """
        Remove cached embedding.

        Args:
            text: Input text.
            model: Model identifier.

        Returns:
            True if entry was removed.
        """
        key = self._make_key(text, model)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self._ttl,
        }


class DiskEmbeddingCache(EmbeddingCache):
    """
    Disk-backed embedding cache using diskcache.

    Persists embeddings to disk for reuse across sessions.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_size_mb: int = 5000,
        ttl_seconds: int | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize disk embedding cache.

        Args:
            cache_dir: Cache directory.
            max_size_mb: Max cache size in MB.
            ttl_seconds: Entry TTL.
            settings: Settings instance.
        """
        super().__init__(ttl_seconds=ttl_seconds, settings=settings)
        self._max_size_mb = max_size_mb

        try:
            from diskcache import Cache

            cache_path = cache_dir or str(self._settings.cache_directory / "embeddings")
            self._disk_cache = Cache(cache_path, size_limit=max_size_mb * 1024 * 1024)
        except ImportError:
            raise ImportError("diskcache required: pip install diskcache")

    def get(self, text: str, model: str | None = None) -> list[float] | None:
        """Get from memory or disk cache."""
        # Check memory first
        result = super().get(text, model)
        if result is not None:
            return result

        # Check disk
        key = self._make_key(text, model)
        entry = self._disk_cache.get(key)

        if entry is not None:
            embedding, timestamp = entry
            if not self._is_expired(timestamp):
                # Promote to memory cache
                self._cache[key] = entry
                self._hits += 1
                return embedding
            else:
                # Remove expired entry
                self._disk_cache.delete(key)

        self._misses += 1
        return None

    def set(self, text: str, embedding: list[float], model: str | None = None) -> None:
        """Set in both memory and disk cache."""
        super().set(text, embedding, model)

        # Persist to disk
        key = self._make_key(text, model)
        self._disk_cache.set(
            key,
            (embedding, time.time()),
            expire=self._ttl,
        )

    def close(self) -> None:
        """Close disk cache."""
        self._disk_cache.close()


class CachedEmbedder:
    """
    Wrapper that adds caching to any embedder.

    Transparently caches embeddings for repeated inputs.
    """

    def __init__(
        self,
        embedder: Any,  # BaseEmbedder
        cache: EmbeddingCache | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize cached embedder.

        Args:
            embedder: Underlying embedder.
            cache: Embedding cache (creates new if None).
            settings: Settings instance.
        """
        self._embedder = embedder
        self._cache = cache or EmbeddingCache(settings=settings)
        self._model = getattr(embedder, "model_name", "default")

    async def embed(self, text: str) -> list[float]:
        """
        Embed text with caching.

        Args:
            text: Input text.

        Returns:
            Embedding vector.
        """
        cached = self._cache.get(text, self._model)
        if cached is not None:
            return cached

        embedding = await self._embedder.embed(text)
        self._cache.set(text, embedding, self._model)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed batch with caching.

        Args:
            texts: Input texts.

        Returns:
            List of embeddings.
        """
        # Check cache
        embeddings, uncached_indices = self._cache.get_batch(texts, self._model)

        if not uncached_indices:
            return embeddings  # type: ignore

        # Compute uncached embeddings
        uncached_texts = [texts[i] for i in uncached_indices]
        new_embeddings = await self._embedder.embed_batch(uncached_texts)

        # Update cache and results
        for idx, emb in zip(uncached_indices, new_embeddings, strict=False):
            embeddings[idx] = emb
            self._cache.set(texts[idx], emb, self._model)

        return embeddings  # type: ignore

    @property
    def model_name(self) -> str:
        """Get underlying model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return getattr(self._embedder, "dimension", 0)

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()
