"""
Semantic cache for RAG queries.

Caches query-response pairs using semantic similarity,
avoiding repeated LLM calls for similar queries.
"""

import hashlib
import time
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.config import Settings, get_settings
from agentic_rag.embeddings import BaseEmbedder


class CacheEntry(BaseModel):
    """A cached query-response pair."""

    query: str
    query_embedding: list[float]
    response: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    """
    Semantic similarity-based cache.

    Caches query-response pairs and finds matches using
    embedding similarity rather than exact string matching.

    Benefits:
    - Handles paraphrased queries
    - Reduces LLM API costs
    - Improves response latency
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = None,
        max_entries: int = 10000,
        settings: Settings | None = None,
    ):
        """
        Initialize semantic cache.

        Args:
            embedder: Embedding model for query similarity.
            similarity_threshold: Min similarity for cache hit.
            ttl_seconds: Cache entry TTL.
            max_entries: Maximum cache entries.
            settings: Settings instance.
        """
        self._embedder = embedder
        self._settings = settings or get_settings()
        self._threshold = similarity_threshold or self._settings.cache_similarity_threshold
        self._ttl = ttl_seconds or self._settings.cache_ttl_seconds
        self._max_entries = max_entries
        self._cache: dict[str, CacheEntry] = {}

    def _compute_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Compute cosine similarity between embeddings."""
        import numpy as np

        a = np.array(emb1)
        b = np.array(emb2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _make_key(self, query: str) -> str:
        """Create a cache key from query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        return time.time() - entry.created_at > self._ttl

    def _evict_if_needed(self) -> None:
        """Evict old entries if cache is full."""
        if len(self._cache) >= self._max_entries:
            # Remove oldest entries
            entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].created_at,
            )
            # Remove 10% of oldest
            to_remove = max(1, len(entries) // 10)
            for key, _ in entries[:to_remove]:
                del self._cache[key]

    async def get(self, query: str) -> CacheEntry | None:
        """
        Get cached response for query.

        Searches for semantically similar cached queries.

        Args:
            query: Query string.

        Returns:
            CacheEntry if found, None otherwise.
        """
        if not self._cache:
            return None

        # Get query embedding
        query_embedding = await self._embedder.embed(query)

        # Search for similar cached queries
        best_match: CacheEntry | None = None
        best_similarity = 0.0

        for entry in self._cache.values():
            # Skip expired entries
            if self._is_expired(entry):
                continue

            similarity = self._compute_similarity(query_embedding, entry.query_embedding)

            if similarity > best_similarity and similarity >= self._threshold:
                best_similarity = similarity
                best_match = entry

        if best_match:
            best_match.hits += 1
            return best_match

        return None

    async def set(
        self,
        query: str,
        response: str,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Cache a query-response pair.

        Args:
            query: Query string.
            response: Response string.
            sources: Source chunks (as dicts).
            metadata: Additional metadata.
        """
        self._evict_if_needed()

        # Get query embedding
        query_embedding = await self._embedder.embed(query)

        # Create entry
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            response=response,
            sources=sources or [],
            metadata=metadata or {},
        )

        # Store with key
        key = self._make_key(query)
        self._cache[key] = entry

    def invalidate(self, query: str) -> bool:
        """
        Invalidate a cached query.

        Args:
            query: Query to invalidate.

        Returns:
            True if entry was found and removed.
        """
        key = self._make_key(query)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed.
        """
        expired_keys = [key for key, entry in self._cache.items() if self._is_expired(entry)]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hits for e in self._cache.values())
        return {
            "entries": len(self._cache),
            "total_hits": total_hits,
            "max_entries": self._max_entries,
            "threshold": self._threshold,
            "ttl_seconds": self._ttl,
        }


class DiskSemanticCache(SemanticCache):
    """
    Disk-backed semantic cache using diskcache.

    Persists cache entries to disk for durability
    across restarts.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        cache_dir: str | None = None,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = None,
        max_size_mb: int = 1000,
        settings: Settings | None = None,
    ):
        """
        Initialize disk-backed cache.

        Args:
            embedder: Embedding model.
            cache_dir: Cache directory path.
            similarity_threshold: Min similarity for hit.
            ttl_seconds: Entry TTL.
            max_size_mb: Max cache size in MB.
            settings: Settings instance.
        """
        super().__init__(
            embedder=embedder,
            similarity_threshold=similarity_threshold,
            ttl_seconds=ttl_seconds,
            settings=settings,
        )
        self._max_size_mb = max_size_mb

        # Initialize diskcache
        try:
            from diskcache import Cache

            cache_path = cache_dir or str(self._settings.cache_directory / "semantic")
            self._disk_cache = Cache(cache_path, size_limit=max_size_mb * 1024 * 1024)
        except ImportError:
            raise ImportError("diskcache required: pip install diskcache")

    async def get(self, query: str) -> CacheEntry | None:
        """Get from disk cache."""
        # First check memory cache
        result = await super().get(query)
        if result:
            return result

        # Search disk cache
        query_embedding = await self._embedder.embed(query)

        for key in self._disk_cache:
            try:
                entry_dict = self._disk_cache.get(key)
                if entry_dict:
                    entry = CacheEntry(**entry_dict)
                    if not self._is_expired(entry):
                        similarity = self._compute_similarity(
                            query_embedding, entry.query_embedding
                        )
                        if similarity >= self._threshold:
                            # Promote to memory cache
                            self._cache[key] = entry
                            return entry
            except Exception:
                continue

        return None

    async def set(
        self,
        query: str,
        response: str,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set in both memory and disk cache."""
        await super().set(query, response, sources, metadata)

        # Also persist to disk
        key = self._make_key(query)
        if key in self._cache:
            entry = self._cache[key]
            self._disk_cache.set(key, entry.model_dump(), expire=self._ttl)

    def close(self) -> None:
        """Close disk cache."""
        self._disk_cache.close()
