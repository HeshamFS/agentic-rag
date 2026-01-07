"""
Redis-backed semantic cache for production RAG deployments.

Provides distributed caching with semantic similarity search,
enabling multi-instance deployments and persistent caching.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel

from agentic_rag.caching.semantic_cache import CacheEntry, SemanticCache
from agentic_rag.config import Settings
from agentic_rag.embeddings import BaseEmbedder

logger = logging.getLogger(__name__)


class RedisConfig(BaseModel):
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    ssl: bool = False
    prefix: str = "rag:cache:"
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "RedisConfig":
        """
        Create config from Redis URL.

        Args:
            url: Redis URL (redis://host:port/db or redis://:password@host:port/db)
            **kwargs: Additional config overrides.

        Returns:
            RedisConfig instance.
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            db=int(parsed.path.lstrip("/") or 0),
            password=parsed.password,
            ssl=parsed.scheme == "rediss",
            **kwargs,
        )


class RedisSemanticCache(SemanticCache):
    """
    Redis-backed semantic cache for production deployments.

    Features:
    - Distributed caching across multiple workers/instances
    - Native Redis TTL for automatic expiration
    - Semantic similarity search using embedding comparison
    - Atomic operations for cache updates
    - Connection pooling for performance
    - Graceful fallback on connection errors

    Usage:
        cache = RedisSemanticCache(
            embedder=my_embedder,
            redis_config=RedisConfig(host="localhost", port=6379),
        )
        await cache.connect()

        # Cache a response
        await cache.set("What is RAG?", "RAG is...", sources=[...])

        # Get cached response
        entry = await cache.get("What is retrieval augmented generation?")

        await cache.close()
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        redis_config: RedisConfig | None = None,
        redis_url: str | None = None,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = None,
        max_scan_count: int = 1000,
        settings: Settings | None = None,
    ):
        """
        Initialize Redis semantic cache.

        Args:
            embedder: Embedding model for query similarity.
            redis_config: Redis connection configuration.
            redis_url: Alternative: Redis URL (redis://host:port/db).
            similarity_threshold: Min similarity for cache hit (0.0-1.0).
            ttl_seconds: Cache entry TTL in seconds.
            max_scan_count: Max entries to scan for similarity search.
            settings: Settings instance.
        """
        # Initialize parent (for threshold, ttl, embedder)
        super().__init__(
            embedder=embedder,
            similarity_threshold=similarity_threshold,
            ttl_seconds=ttl_seconds,
            settings=settings,
        )

        # Redis config
        if redis_url:
            self._redis_config = RedisConfig.from_url(redis_url)
        else:
            self._redis_config = redis_config or RedisConfig()

        self._max_scan_count = max_scan_count
        self._client: Any = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._client is not None

    async def connect(self) -> None:
        """
        Establish Redis connection.

        Raises:
            ImportError: If redis package not installed.
            ConnectionError: If unable to connect to Redis.
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis package required for RedisSemanticCache. "
                "Install with: pip install redis>=5.0.0"
            )

        try:
            self._client = redis.Redis(
                host=self._redis_config.host,
                port=self._redis_config.port,
                db=self._redis_config.db,
                password=self._redis_config.password,
                ssl=self._redis_config.ssl,
                socket_timeout=self._redis_config.socket_timeout,
                socket_connect_timeout=self._redis_config.socket_connect_timeout,
                max_connections=self._redis_config.max_connections,
                decode_responses=False,  # We handle JSON encoding ourselves
            )
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(
                f"Connected to Redis at {self._redis_config.host}:{self._redis_config.port}"
            )
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._connected = False
            logger.info("Closed Redis connection")

    def _make_redis_key(self, query_hash: str) -> str:
        """Create Redis key with prefix."""
        return f"{self._redis_config.prefix}{query_hash}"

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry to JSON bytes."""
        return json.dumps(entry.model_dump()).encode("utf-8")

    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry from JSON bytes."""
        return CacheEntry(**json.loads(data.decode("utf-8")))

    async def get(self, query: str) -> CacheEntry | None:
        """
        Get cached response for query.

        Searches for semantically similar cached queries in Redis.
        Falls back to in-memory search if Redis is unavailable.

        Args:
            query: Query string.

        Returns:
            CacheEntry if found with sufficient similarity, None otherwise.
        """
        # Fallback to in-memory if not connected
        if not self.is_connected:
            logger.warning("Redis not connected, falling back to in-memory cache")
            return await super().get(query)

        try:
            # Get query embedding
            query_embedding = await self._embedder.embed(query)

            # Scan Redis for similar queries
            best_match: CacheEntry | None = None
            best_similarity = 0.0

            # Use SCAN to iterate through cache keys
            prefix = self._redis_config.prefix
            cursor = 0
            scanned = 0

            while scanned < self._max_scan_count:
                cursor, keys = await self._client.scan(
                    cursor=cursor,
                    match=f"{prefix}*",
                    count=100,
                )

                for key in keys:
                    if scanned >= self._max_scan_count:
                        break
                    scanned += 1

                    try:
                        data = await self._client.get(key)
                        if data:
                            entry = self._deserialize_entry(data)
                            similarity = self._compute_similarity(
                                query_embedding, entry.query_embedding
                            )

                            if similarity > best_similarity and similarity >= self._threshold:
                                best_similarity = similarity
                                best_match = entry
                    except Exception as e:
                        logger.debug(f"Error reading cache entry {key}: {e}")
                        continue

                if cursor == 0:
                    break

            if best_match:
                # Increment hit counter (fire and forget)
                try:
                    key = self._make_redis_key(self._make_key(best_match.query))
                    best_match.hits += 1
                    await self._client.set(
                        key,
                        self._serialize_entry(best_match),
                        ex=self._ttl,
                    )
                except Exception:
                    pass  # Don't fail on hit counter update
                return best_match

            return None

        except Exception as e:
            logger.error(f"Redis get error, falling back to in-memory: {e}")
            return await super().get(query)

    async def set(
        self,
        query: str,
        response: str,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Cache a query-response pair in Redis.

        Args:
            query: Query string.
            response: Response string.
            sources: Source chunks (as dicts).
            metadata: Additional metadata.
        """
        # Always update in-memory cache too
        await super().set(query, response, sources, metadata)

        if not self.is_connected:
            logger.warning("Redis not connected, cached in-memory only")
            return

        try:
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

            # Store in Redis with TTL
            key = self._make_redis_key(self._make_key(query))
            await self._client.set(
                key,
                self._serialize_entry(entry),
                ex=self._ttl,
            )
            logger.debug(f"Cached query in Redis: {key}")

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            # In-memory cache already updated by super().set()

    async def invalidate(self, query: str) -> bool:
        """
        Invalidate a cached query.

        Args:
            query: Query to invalidate.

        Returns:
            True if entry was found and removed.
        """
        # Invalidate in-memory
        removed = super().invalidate(query)

        if not self.is_connected:
            return removed

        try:
            key = self._make_redis_key(self._make_key(query))
            result = await self._client.delete(key)
            return result > 0 or removed
        except Exception as e:
            logger.error(f"Redis invalidate error: {e}")
            return removed

    async def clear(self) -> None:
        """Clear all cached entries."""
        # Clear in-memory
        super().clear()

        if not self.is_connected:
            return

        try:
            # Delete all keys with our prefix
            prefix = self._redis_config.prefix
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self._client.scan(
                    cursor=cursor,
                    match=f"{prefix}*",
                    count=100,
                )

                if keys:
                    await self._client.delete(*keys)
                    deleted += len(keys)

                if cursor == 0:
                    break

            logger.info(f"Cleared {deleted} entries from Redis cache")

        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    async def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (entries, hits, etc.).
        """
        base_stats = super().stats()
        base_stats["backend"] = "redis"
        base_stats["connected"] = self.is_connected

        if not self.is_connected:
            return base_stats

        try:
            # Count Redis entries
            prefix = self._redis_config.prefix
            cursor = 0
            redis_entries = 0
            total_hits = 0

            while True:
                cursor, keys = await self._client.scan(
                    cursor=cursor,
                    match=f"{prefix}*",
                    count=100,
                )
                redis_entries += len(keys)

                # Sum hits from a sample
                for key in keys[:10]:  # Sample first 10
                    try:
                        data = await self._client.get(key)
                        if data:
                            entry = self._deserialize_entry(data)
                            total_hits += entry.hits
                    except Exception:
                        pass

                if cursor == 0:
                    break

            base_stats["redis_entries"] = redis_entries
            base_stats["redis_sample_hits"] = total_hits
            base_stats["redis_host"] = self._redis_config.host
            base_stats["redis_port"] = self._redis_config.port

        except Exception as e:
            base_stats["redis_error"] = str(e)

        return base_stats

    async def __aenter__(self) -> "RedisSemanticCache":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
