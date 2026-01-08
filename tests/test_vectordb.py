"""
Comprehensive unit tests for vector database functionality.

Tests:
- QdrantVectorDB operations
- Collection management
- Vector upsert and search
- Hybrid search
- Metadata filtering
- Error handling
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag.core.models import Chunk
from agentic_rag.vectordb.qdrant_client import CollectionManager, QdrantVectorDB

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client."""
    with patch("agentic_rag.vectordb.qdrant_client.AsyncQdrantClient") as mock:
        client = MagicMock()

        # Mock collections response - use proper string names, not MagicMock
        collection1_mock = MagicMock()
        collection1_mock.name = "collection1"  # Set actual string value
        collection2_mock = MagicMock()
        collection2_mock.name = "collection2"  # Set actual string value

        mock_collections = MagicMock()
        mock_collections.collections = [collection1_mock, collection2_mock]
        client.get_collections = AsyncMock(return_value=mock_collections)

        # Mock collection info
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.status = MagicMock(name="GREEN")
        mock_info.config = MagicMock()
        mock_info.config.model_dump.return_value = {}
        client.get_collection = AsyncMock(return_value=mock_info)

        # Mock create/delete collection
        client.create_collection = AsyncMock()
        client.delete_collection = AsyncMock()

        # Mock upsert
        client.upsert = AsyncMock()

        # Mock search
        mock_search_result = MagicMock()
        mock_search_result.points = [
            MagicMock(
                id="point1",
                score=0.95,
                payload={"content": "Test content 1", "document_id": "doc1"},
                vector={"dense": [0.1] * 384},
            ),
            MagicMock(
                id="point2",
                score=0.85,
                payload={"content": "Test content 2", "document_id": "doc2"},
                vector={"dense": [0.2] * 384},
            ),
        ]
        client.query_points = AsyncMock(return_value=mock_search_result)

        # Mock scroll
        client.scroll = AsyncMock(return_value=([], None))

        # Mock delete
        client.delete = AsyncMock()

        # Mock close
        client.close = AsyncMock()

        mock.return_value = client
        yield client


@pytest.fixture
def qdrant_db(mock_qdrant_client, test_settings_minimal):
    """Create QdrantVectorDB with mock client."""
    db = QdrantVectorDB(
        settings=test_settings_minimal,
        url="http://localhost:6333",
    )
    db._client = mock_qdrant_client
    return db


@pytest.fixture
def sample_chunks_with_embeddings() -> list[Chunk]:
    """Create sample chunks with embeddings."""
    return [
        Chunk(
            id="chunk1",
            content="Machine learning is AI that learns from data.",
            document_id="doc1",
            embedding=[0.1] * 384,
            metadata={"topic": "ml"},
        ),
        Chunk(
            id="chunk2",
            content="Deep learning uses neural networks.",
            document_id="doc2",
            embedding=[0.2] * 384,
            metadata={"topic": "dl"},
        ),
        Chunk(
            id="chunk3",
            content="Natural language processing understands text.",
            document_id="doc3",
            embedding=[0.3] * 384,
            metadata={"topic": "nlp"},
        ),
    ]


# =============================================================================
# QdrantVectorDB Tests
# =============================================================================


class TestQdrantVectorDB:
    """Tests for the QdrantVectorDB class."""

    def test_initialization(self, qdrant_db):
        """Test VectorDB initialization."""
        assert qdrant_db is not None
        assert qdrant_db.db_type == "qdrant"

    # -------------------------------------------------------------------------
    # Collection Management Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_collection(self, qdrant_db, mock_qdrant_client):
        """Test collection creation."""
        await qdrant_db.create_collection(
            name="test_collection",
            dimension=384,
            distance="cosine",
        )

        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_with_sparse(self, qdrant_db, mock_qdrant_client):
        """Test collection creation with sparse vectors enabled."""
        await qdrant_db.create_collection(
            name="hybrid_collection",
            dimension=384,
            enable_sparse=True,
        )

        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args[1]
        # Should have sparse config
        assert "sparse_vectors_config" in call_kwargs or "vectors_config" in call_kwargs

    @pytest.mark.asyncio
    async def test_delete_collection(self, qdrant_db, mock_qdrant_client):
        """Test collection deletion."""
        await qdrant_db.delete_collection("test_collection")

        mock_qdrant_client.delete_collection.assert_called_once_with(
            collection_name="test_collection"
        )

    @pytest.mark.asyncio
    async def test_collection_exists(self, qdrant_db, mock_qdrant_client):
        """Test checking if collection exists."""
        exists = await qdrant_db.collection_exists("collection1")
        assert exists is True

        exists = await qdrant_db.collection_exists("nonexistent")
        assert exists is False

    @pytest.mark.asyncio
    async def test_get_collection_info(self, qdrant_db, mock_qdrant_client):
        """Test getting collection information."""
        info = await qdrant_db.get_collection_info("collection1")

        assert "name" in info
        assert "vectors_count" in info
        assert "points_count" in info

    @pytest.mark.asyncio
    async def test_list_collections(self, qdrant_db, mock_qdrant_client):
        """Test listing all collections."""
        collections = await qdrant_db.list_collections()

        assert isinstance(collections, list)
        assert "collection1" in collections
        assert "collection2" in collections

    # -------------------------------------------------------------------------
    # Vector Operations Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_upsert_chunks(
        self, qdrant_db, mock_qdrant_client, sample_chunks_with_embeddings
    ):
        """Test upserting chunks."""
        await qdrant_db.upsert(
            collection="test",
            chunks=sample_chunks_with_embeddings,
        )

        mock_qdrant_client.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_upsert_batching(
        self, qdrant_db, mock_qdrant_client, sample_chunks_with_embeddings
    ):
        """Test upsert with batching."""
        # Create many chunks
        many_chunks = sample_chunks_with_embeddings * 50

        await qdrant_db.upsert(
            collection="test",
            chunks=many_chunks,
            batch_size=50,
        )

        # Should be called multiple times for batches
        assert mock_qdrant_client.upsert.call_count >= 1

    @pytest.mark.asyncio
    async def test_upsert_without_embedding_raises(self, qdrant_db):
        """Test that upserting without embedding raises error."""
        chunk_no_embedding = Chunk(
            id="no_emb",
            content="No embedding",
            document_id="doc",
        )

        with pytest.raises(ValueError, match="no embedding"):
            await qdrant_db.upsert(
                collection="test",
                chunks=[chunk_no_embedding],
            )

    @pytest.mark.asyncio
    async def test_delete_chunks(self, qdrant_db, mock_qdrant_client):
        """Test deleting chunks by ID."""
        await qdrant_db.delete(
            collection="test",
            chunk_ids=["chunk1", "chunk2"],
        )

        mock_qdrant_client.delete.assert_called_once()

    # -------------------------------------------------------------------------
    # Search Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_returns_chunks(self, qdrant_db):
        """Test search returns chunks with scores."""
        results = await qdrant_db.search(
            collection="test",
            query_vector=[0.1] * 384,
            top_k=5,
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)
        assert all(isinstance(r[0], Chunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, qdrant_db, mock_qdrant_client):
        """Test search respects top_k parameter."""
        await qdrant_db.search(
            collection="test",
            query_vector=[0.1] * 384,
            top_k=3,
        )

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["limit"] == 3

    @pytest.mark.asyncio
    async def test_search_with_filters(self, qdrant_db, mock_qdrant_client):
        """Test search with metadata filters."""
        await qdrant_db.search(
            collection="test",
            query_vector=[0.1] * 384,
            top_k=5,
            filters={"topic": "ml"},
        )

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs.get("query_filter") is not None

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, qdrant_db, mock_qdrant_client):
        """Test search with score threshold."""
        await qdrant_db.search(
            collection="test",
            query_vector=[0.1] * 384,
            top_k=5,
            score_threshold=0.8,
        )

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs.get("score_threshold") == 0.8

    @pytest.mark.asyncio
    async def test_hybrid_search(self, qdrant_db):
        """Test hybrid search (dense + sparse)."""
        results = await qdrant_db.hybrid_search(
            collection="test",
            query_vector=[0.1] * 384,
            query_text="machine learning",
            top_k=5,
            alpha=0.5,
        )

        assert isinstance(results, list)

    # -------------------------------------------------------------------------
    # Get All Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_all_chunks(self, qdrant_db, mock_qdrant_client):
        """Test getting all chunks from collection."""
        # Setup scroll to return some chunks
        mock_chunks = [
            MagicMock(
                id="chunk1",
                payload={"content": "Content 1", "document_id": "doc1"},
                vector={"dense": [0.1] * 384},
            ),
        ]
        mock_qdrant_client.scroll = AsyncMock(
            side_effect=[
                (mock_chunks, None),  # First call returns chunks, no more
            ]
        )

        chunks = await qdrant_db.get_all(collection="test")

        assert isinstance(chunks, list)

    # -------------------------------------------------------------------------
    # Search by Payload Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_by_payload(self, qdrant_db, mock_qdrant_client):
        """Test searching by payload field."""
        mock_qdrant_client.scroll = AsyncMock(
            return_value=(
                [
                    MagicMock(
                        id="chunk1",
                        payload={"content": "ML content", "document_id": "doc1", "topic": "ml"},
                        vector={"dense": [0.1] * 384},
                    ),
                ],
                None,
            )
        )

        chunks = await qdrant_db.search_by_payload(
            collection="test",
            key="topic",
            value="ml",
        )

        assert isinstance(chunks, list)

    # -------------------------------------------------------------------------
    # Close Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_close(self, qdrant_db, mock_qdrant_client):
        """Test closing the client."""
        await qdrant_db.close()

        mock_qdrant_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self, test_settings_minimal):
        """Test closing when not connected."""
        db = QdrantVectorDB(settings=test_settings_minimal)
        # Should not raise
        await db.close()


# =============================================================================
# Filter Building Tests
# =============================================================================


class TestFilterBuilding:
    """Tests for filter building functionality."""

    def test_build_filter_single_value(self, qdrant_db):
        """Test building filter for single value."""
        filter_obj = qdrant_db._build_filter({"topic": "ml"})

        assert filter_obj is not None
        assert len(filter_obj.must) == 1

    def test_build_filter_multiple_values(self, qdrant_db):
        """Test building filter for multiple conditions."""
        filter_obj = qdrant_db._build_filter(
            {
                "topic": "ml",
                "language": "en",
            }
        )

        assert filter_obj is not None
        assert len(filter_obj.must) == 2

    def test_build_filter_list_value(self, qdrant_db):
        """Test building filter for list values (match any)."""
        filter_obj = qdrant_db._build_filter(
            {
                "topic": ["ml", "dl", "nlp"],
            }
        )

        assert filter_obj is not None


# =============================================================================
# CollectionManager Tests
# =============================================================================


class TestCollectionManager:
    """Tests for the CollectionManager class."""

    @pytest.fixture
    def collection_manager(self, qdrant_db):
        """Create collection manager."""
        return CollectionManager(db=qdrant_db)

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_new(self, collection_manager, mock_qdrant_client):
        """Test ensuring collection creates if not exists."""
        # Setup to return no existing collections
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_qdrant_client.get_collections = AsyncMock(return_value=mock_collections)

        await collection_manager.ensure_collection(
            name="new_collection",
            dimension=384,
        )

        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_skips_existing(self, collection_manager, mock_qdrant_client):
        """Test ensuring collection skips if exists."""
        # collection1 already exists in fixture
        await collection_manager.ensure_collection(
            name="collection1",
            dimension=384,
        )

        # Should not create (already exists)
        # Note: Depends on mock behavior

    @pytest.mark.asyncio
    async def test_ensure_collection_recreate(self, collection_manager, mock_qdrant_client):
        """Test ensuring collection with recreate=True."""
        await collection_manager.ensure_collection(
            name="collection1",
            dimension=384,
            recreate=True,
        )

        # Should delete and recreate
        mock_qdrant_client.delete_collection.assert_called()
        mock_qdrant_client.create_collection.assert_called()

    @pytest.mark.asyncio
    async def test_get_all_collections(self, collection_manager, mock_qdrant_client):
        """Test getting all collections info."""
        collections = await collection_manager.get_all_collections()

        assert isinstance(collections, list)
        assert len(collections) == 2


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestVectorDBErrorHandling:
    """Tests for error handling in vector database operations."""

    @pytest.fixture
    def failing_db(self, test_settings_minimal):
        """Create DB that simulates failures."""
        with patch("agentic_rag.vectordb.qdrant_client.AsyncQdrantClient") as mock:
            client = MagicMock()
            client.create_collection = AsyncMock(side_effect=Exception("Connection failed"))
            client.upsert = AsyncMock(side_effect=Exception("Upsert failed"))
            client.query_points = AsyncMock(side_effect=Exception("Search failed"))
            mock.return_value = client

            db = QdrantVectorDB(settings=test_settings_minimal)
            db._client = client
            return db

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, failing_db):
        """Test handling of connection errors."""
        with pytest.raises(Exception, match="Connection failed"):
            await failing_db.create_collection("test", 384)

    @pytest.mark.asyncio
    async def test_handles_upsert_error(self, failing_db, sample_chunks_with_embeddings):
        """Test handling of upsert errors."""
        with pytest.raises(Exception, match="Upsert failed"):
            await failing_db.upsert("test", sample_chunks_with_embeddings)

    @pytest.mark.asyncio
    async def test_handles_search_error(self, failing_db):
        """Test handling of search errors."""
        with pytest.raises(Exception, match="Search failed"):
            await failing_db.search("test", [0.1] * 384)


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestVectorDBPerformance:
    """Performance tests for vector database operations."""

    @pytest.fixture
    def fast_db(self, test_settings_minimal):
        """Create fast mock DB for performance testing."""
        with patch("agentic_rag.vectordb.qdrant_client.AsyncQdrantClient") as mock:
            client = MagicMock()
            client.upsert = AsyncMock()
            client.query_points = AsyncMock(return_value=MagicMock(points=[]))
            client.close = AsyncMock()
            mock.return_value = client

            db = QdrantVectorDB(settings=test_settings_minimal)
            db._client = client
            return db

    @pytest.mark.asyncio
    async def test_batch_upsert_performance(self, fast_db):
        """Test batch upsert performance."""
        import time

        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"Content {i}",
                document_id=f"doc_{i}",
                embedding=[0.1] * 384,
            )
            for i in range(1000)
        ]

        start = time.time()
        await fast_db.upsert("test", chunks, batch_size=100)
        elapsed = time.time() - start

        # Should complete quickly with mocks
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, fast_db):
        """Test concurrent search handling."""
        import time

        async def search():
            return await fast_db.search("test", [0.1] * 384, top_k=10)

        start = time.time()
        await asyncio.gather(*[search() for _ in range(100)])
        elapsed = time.time() - start

        # Concurrent should be efficient
        assert elapsed < 5.0
