"""
Qdrant vector database implementation.

Provides async operations for:
- Collection management
- Vector upsert and search
- Hybrid search (dense + sparse)
- Metadata filtering
"""

from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk


class QdrantVectorDB:
    """
    Async Qdrant vector database client.

    Supports:
    - Dense vector search
    - Hybrid search (dense + sparse via BM25)
    - Metadata filtering
    - Batch operations
    """

    def __init__(
        self,
        settings: Settings | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize Qdrant client.

        Args:
            settings: Settings instance. If None, loads from environment.
            url: Override Qdrant URL.
            api_key: Override API key.
        """
        self._settings = settings or get_settings()
        self._url = url or self._settings.qdrant_url
        self._api_key = api_key or (
            self._settings.qdrant_api_key.get_secret_value()
            if self._settings.qdrant_api_key
            else None
        )
        self._client: AsyncQdrantClient | None = None

    @property
    def db_type(self) -> str:
        """Return the database type identifier."""
        return "qdrant"

    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create async client."""
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self._url,
                api_key=self._api_key,
                prefer_grpc=self._settings.qdrant_prefer_grpc,
            )
        return self._client

    async def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # =========================================================================
    # Collection Management
    # =========================================================================

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
        enable_sparse: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Create a new collection.

        Args:
            name: Collection name.
            dimension: Vector dimension.
            distance: Distance metric (cosine, euclid, dot).
            enable_sparse: Enable sparse vectors for hybrid search.
            **kwargs: Additional Qdrant parameters.
        """
        client = await self._get_client()

        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        qdrant_distance = distance_map.get(distance.lower(), Distance.COSINE)

        # Configure vectors
        vectors_config: dict[str, VectorParams] | VectorParams
        if enable_sparse:
            # Named vectors for hybrid search
            vectors_config = {
                "dense": VectorParams(
                    size=dimension,
                    distance=qdrant_distance,
                ),
            }
        else:
            # Single vector config
            vectors_config = VectorParams(
                size=dimension,
                distance=qdrant_distance,
            )

        # Create collection
        await client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=(
                {"sparse": models.SparseVectorParams()} if enable_sparse else None
            ),
            **kwargs,
        )

    async def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        client = await self._get_client()
        await client.delete_collection(collection_name=name)

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        client = await self._get_client()
        collections = await client.get_collections()
        return any(c.name == name for c in collections.collections)

    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Get collection information."""
        client = await self._get_client()
        info = await client.get_collection(collection_name=name)

        # Handle different Qdrant API versions
        # Newer versions use info.points_count, older used info.vectors_count
        vectors_count = getattr(info, "vectors_count", None)
        points_count = getattr(info, "points_count", None)

        return {
            "name": name,
            "vectors_count": vectors_count or points_count or 0,
            "points_count": points_count or vectors_count or 0,
            "status": info.status.name if hasattr(info, "status") else "unknown",
            "config": info.config.model_dump() if hasattr(info, "config") and info.config else {},
        }

    # =========================================================================
    # Vector Operations
    # =========================================================================

    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        batch_size: int = 100,
    ) -> None:
        """
        Insert or update document chunks in the Qdrant collection.

        The method handles:
        1. Validating that each chunk has an embedding.
        2. Constructing the Qdrant payload with all chunk metadata (context headers, position, etc.).
        3. Efficiently batching the points for high-performance upsert.

        Args:
            collection: The name of the collection.
            chunks: A list of Chunk objects to be indexed.
            batch_size: The number of points per upsert request.
        """
        client = await self._get_client()

        # Prepare points
        points: list[PointStruct] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")

            # Build payload with all metadata
            payload = {
                "content": chunk.content,
                "document_id": chunk.document_id,
                "context_header": chunk.context_header,
                "position": chunk.position,
                "level": chunk.level,
                **chunk.metadata,
            }

            # Create point with named vector for hybrid search support
            point = PointStruct(
                id=chunk.id,
                vector={"dense": chunk.embedding},
                payload=payload,
            )
            points.append(point)

        # Batch upsert
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await client.upsert(
                collection_name=collection,
                points=batch,
            )

    async def delete(
        self,
        collection: str,
        chunk_ids: list[str],
    ) -> None:
        """Delete chunks by ID."""
        client = await self._get_client()
        await client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=chunk_ids),
        )

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for the nearest neighbors of a query vector in the collection.

        Uses Qdrant's high-performance HNSW index for fast semantic retrieval.
        Optionally applies metadata filters and a minimum score threshold.

        Args:
            collection: The name of the collection to search.
            query_vector: The high-dimensional embedding of the query.
            top_k: The number of results to return.
            filters: Metadata filter criteria (e.g., {"document_id": "doc1"}).
            score_threshold: Minimum similarity score (0.0-1.0) to include a result.

        Returns:
            A list of tuples, each containing a matching Chunk and its similarity score.
        """
        client = await self._get_client()

        # Build filter if provided
        qdrant_filter = self._build_filter(filters) if filters else None

        # Search using query_points (new API)
        response = await client.query_points(
            collection_name=collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        # Convert to chunks
        return self._results_to_chunks(response.points)

    async def hybrid_search(
        self,
        collection: str,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Hybrid search combining dense and sparse retrieval.

        Note: This requires sparse vectors to be indexed separately.
        For simpler hybrid search, use the HybridRetriever which combines
        dense Qdrant search with external BM25.

        Args:
            collection: Collection name.
            query_vector: Dense embedding vector.
            query_text: Text for sparse matching.
            top_k: Number of results.
            alpha: Weight for dense vs sparse (0=sparse, 1=dense).
            filters: Optional metadata filters.

        Returns:
            List of (chunk, score) tuples.
        """
        # For now, fall back to dense search
        # Full hybrid requires sparse vector indexing which adds complexity
        # The HybridRetriever class handles this better with external BM25
        return await self.search(
            collection=collection,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

    async def search_by_payload(
        self,
        collection: str,
        key: str,
        value: Any,
        top_k: int = 10,
    ) -> list[Chunk]:
        """
        Search by payload field (metadata).

        Args:
            collection: Collection name.
            key: Payload field key.
            value: Value to match.
            top_k: Maximum results.

        Returns:
            List of matching chunks.
        """
        client = await self._get_client()

        # Scroll with filter
        results, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )

        return [self._point_to_chunk(r) for r in results]

    async def get_all(
        self,
        collection: str,
        batch_size: int = 100,
    ) -> list[Chunk]:
        """
        Get all chunks from a collection.

        Args:
            collection: Collection name.
            batch_size: Batch size for scrolling.

        Returns:
            List of all chunks in collection.
        """
        client = await self._get_client()

        all_chunks: list[Chunk] = []
        offset = None

        while True:
            results, next_offset = await client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not results:
                break

            all_chunks.extend(self._point_to_chunk(r) for r in results)
            offset = next_offset

            if offset is None:
                break

        return all_chunks

    async def list_collections(self) -> list[str]:
        """List all collection names."""
        client = await self._get_client()
        collections = await client.get_collections()
        return [c.name for c in collections.collections]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_filter(self, filters: dict[str, Any]) -> models.Filter:
        """Build Qdrant filter from dict."""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # Match any in list
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                # Exact match
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=conditions)

    def _results_to_chunks(
        self,
        results: list[models.ScoredPoint],
    ) -> list[tuple[Chunk, float]]:
        """Convert Qdrant results to chunks with scores."""
        chunks = []
        for result in results:
            chunk = self._point_to_chunk(result)
            chunks.append((chunk, result.score))
        return chunks

    def _point_to_chunk(
        self,
        point: models.ScoredPoint | models.Record,
    ) -> Chunk:
        """Convert Qdrant point to Chunk."""
        payload = point.payload or {}

        # Extract embedding if available
        embedding = None
        if hasattr(point, "vector") and point.vector:
            if isinstance(point.vector, dict):
                embedding = point.vector.get("dense")
            else:
                embedding = point.vector

        # Build chunk
        return Chunk(
            id=str(point.id),
            content=payload.get("content", ""),
            document_id=payload.get("document_id", ""),
            context_header=payload.get("context_header"),
            position=payload.get("position"),
            level=payload.get("level", 0),
            embedding=embedding,
            metadata={
                k: v
                for k, v in payload.items()
                if k not in {"content", "document_id", "context_header", "position", "level"}
            },
        )


# =============================================================================
# Collection Manager
# =============================================================================


class CollectionManager:
    """
    High-level collection management utilities.

    Provides convenience methods for common collection operations.
    """

    def __init__(self, db: QdrantVectorDB):
        self.db = db

    async def ensure_collection(
        self,
        name: str,
        dimension: int,
        recreate: bool = False,
    ) -> None:
        """
        Ensure a collection exists with the specified configuration.

        Args:
            name: Collection name.
            dimension: Vector dimension.
            recreate: If True, delete and recreate if exists.
        """
        exists = await self.db.collection_exists(name)

        if exists and recreate:
            await self.db.delete_collection(name)
            exists = False

        if not exists:
            await self.db.create_collection(
                name=name,
                dimension=dimension,
            )

    async def get_all_collections(self) -> list[dict[str, Any]]:
        """Get info for all collections."""
        client = await self.db._get_client()
        collections = await client.get_collections()
        return [{"name": c.name} for c in collections.collections]
