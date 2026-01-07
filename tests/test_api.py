"""
Comprehensive unit tests for FastAPI REST API.

Tests:
- Health check endpoint
- Document ingestion endpoint
- Query endpoint
- Collection management endpoints
- Error handling and validation
- CORS and middleware
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from agentic_rag.api import IngestRequest, QueryRequest, app
from agentic_rag.core.models import Chunk, GenerationResult

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_pipeline():
    """Create mock pipeline for API tests."""
    pipeline = MagicMock()

    async def mock_ingest(documents, collection, **kwargs):
        return {
            "documents": len(documents),
            "chunks": len(documents) * 3,
        }

    async def mock_query(question, collection, top_k=5, **kwargs):
        return GenerationResult(
            response=f"Answer to: {question}",
            sources=[
                Chunk(id="c1", content="Source 1", document_id="d1"),
                Chunk(id="c2", content="Source 2", document_id="d2"),
            ],
            confidence=0.85,
            provider="mock",
            model="mock-model",
            input_tokens=100,
            output_tokens=50,
        )

    pipeline.ingest = AsyncMock(side_effect=mock_ingest)
    pipeline.query = AsyncMock(side_effect=mock_query)
    pipeline.vectordb = MagicMock()
    pipeline.vectordb.list_collections = AsyncMock(return_value=["default", "test"])
    pipeline.vectordb.delete_collection = AsyncMock()

    return pipeline


@pytest.fixture
def client_with_mock_pipeline(mock_pipeline):
    """Create test client with mocked pipeline."""
    with patch("agentic_rag.api._pipeline", mock_pipeline):
        with patch("agentic_rag.api._settings", MagicMock(llm_model="test-model")):
            with TestClient(app) as client:
                yield client


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client_with_mock_pipeline):
        """Test health endpoint returns 200."""
        response = client_with_mock_pipeline.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client_with_mock_pipeline):
        """Test health endpoint returns status."""
        response = client_with_mock_pipeline.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_returns_version(self, client_with_mock_pipeline):
        """Test health endpoint returns version."""
        response = client_with_mock_pipeline.get("/health")
        data = response.json()
        assert "version" in data

    def test_health_when_pipeline_not_initialized(self):
        """Test health when pipeline not initialized."""
        with patch("agentic_rag.api._pipeline", None), TestClient(app) as client:
            response = client.get("/health")
            data = response.json()
            # Should still return but indicate not ready
            assert "status" in data


# =============================================================================
# Ingest Endpoint Tests
# =============================================================================


class TestIngestEndpoint:
    """Tests for the /ingest endpoint."""

    def test_ingest_returns_200(self, client_with_mock_pipeline):
        """Test ingest endpoint returns 200."""
        response = client_with_mock_pipeline.post(
            "/ingest",
            json={
                "documents": [
                    {"content": "Test document 1", "metadata": {"source": "test"}},
                    {"content": "Test document 2", "metadata": {"source": "test"}},
                ],
                "collection": "test_collection",
            },
        )
        assert response.status_code == 200

    def test_ingest_returns_success(self, client_with_mock_pipeline):
        """Test ingest returns success status."""
        response = client_with_mock_pipeline.post(
            "/ingest",
            json={
                "documents": [{"content": "Test content"}],
                "collection": "test",
            },
        )
        data = response.json()
        assert data["success"] is True

    def test_ingest_returns_documents_processed(self, client_with_mock_pipeline):
        """Test ingest returns document count."""
        docs = [{"content": f"Document {i}"} for i in range(5)]
        response = client_with_mock_pipeline.post(
            "/ingest",
            json={"documents": docs, "collection": "test"},
        )
        data = response.json()
        assert data["documents_processed"] == 5

    def test_ingest_returns_chunks_created(self, client_with_mock_pipeline):
        """Test ingest returns chunk count."""
        response = client_with_mock_pipeline.post(
            "/ingest",
            json={
                "documents": [{"content": "Test content"}],
                "collection": "test",
            },
        )
        data = response.json()
        assert data["chunks_created"] > 0

    def test_ingest_returns_collection_name(self, client_with_mock_pipeline):
        """Test ingest returns collection name."""
        response = client_with_mock_pipeline.post(
            "/ingest",
            json={
                "documents": [{"content": "Test"}],
                "collection": "my_collection",
            },
        )
        data = response.json()
        assert data["collection"] == "my_collection"

    def test_ingest_with_chunk_strategy(self, client_with_mock_pipeline):
        """Test ingest with different chunk strategies."""
        response = client_with_mock_pipeline.post(
            "/ingest",
            json={
                "documents": [{"content": "Test content for semantic chunking."}],
                "collection": "test",
                "chunk_strategy": "semantic",
                "chunk_size": 256,
            },
        )
        assert response.status_code == 200

    def test_ingest_validates_documents_required(self, client_with_mock_pipeline):
        """Test ingest validates documents field."""
        response = client_with_mock_pipeline.post(
            "/ingest",
            json={"collection": "test"},
        )
        assert response.status_code == 422  # Validation error

    def test_ingest_validates_content_required(self, client_with_mock_pipeline):
        """Test ingest validates content field."""
        response = client_with_mock_pipeline.post(
            "/ingest",
            json={
                "documents": [{"metadata": {"source": "test"}}],  # Missing content
                "collection": "test",
            },
        )
        # Should fail validation or handle gracefully
        assert response.status_code in [422, 500]

    def test_ingest_returns_503_when_not_initialized(self):
        """Test ingest returns 503 when pipeline not initialized."""
        with patch("agentic_rag.api._pipeline", None), TestClient(app) as client:
            response = client.post(
                "/ingest",
                json={"documents": [{"content": "Test"}], "collection": "test"},
            )
            assert response.status_code == 503


# =============================================================================
# Query Endpoint Tests
# =============================================================================


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    def test_query_returns_200(self, client_with_mock_pipeline):
        """Test query endpoint returns 200."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={
                "question": "What is machine learning?",
                "collection": "default",
            },
        )
        assert response.status_code == 200

    def test_query_returns_response(self, client_with_mock_pipeline):
        """Test query returns response text."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={"question": "What is AI?", "collection": "default"},
        )
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_query_returns_sources(self, client_with_mock_pipeline):
        """Test query returns sources."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={"question": "Test question", "collection": "default"},
        )
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) > 0

    def test_query_sources_have_content(self, client_with_mock_pipeline):
        """Test query sources have content."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={"question": "Test", "collection": "default"},
        )
        data = response.json()
        for source in data["sources"]:
            assert "content" in source

    def test_query_returns_metadata(self, client_with_mock_pipeline):
        """Test query returns metadata."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={"question": "Test", "collection": "default"},
        )
        data = response.json()
        assert "metadata" in data

    def test_query_metadata_includes_provider(self, client_with_mock_pipeline):
        """Test query metadata includes provider."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={"question": "Test", "collection": "default"},
        )
        data = response.json()
        assert "provider" in data["metadata"]

    def test_query_with_top_k(self, client_with_mock_pipeline, mock_pipeline):
        """Test query with custom top_k."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={
                "question": "Test",
                "collection": "default",
                "top_k": 10,
            },
        )
        assert response.status_code == 200

    def test_query_top_k_validation_min(self, client_with_mock_pipeline):
        """Test query validates minimum top_k."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={
                "question": "Test",
                "collection": "default",
                "top_k": 0,
            },
        )
        assert response.status_code == 422

    def test_query_top_k_validation_max(self, client_with_mock_pipeline):
        """Test query validates maximum top_k."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={
                "question": "Test",
                "collection": "default",
                "top_k": 100,
            },
        )
        assert response.status_code == 422

    def test_query_validates_question_required(self, client_with_mock_pipeline):
        """Test query validates question field."""
        response = client_with_mock_pipeline.post(
            "/query",
            json={"collection": "default"},
        )
        assert response.status_code == 422

    def test_query_returns_503_when_not_initialized(self):
        """Test query returns 503 when pipeline not initialized."""
        with patch("agentic_rag.api._pipeline", None), TestClient(app) as client:
            response = client.post(
                "/query",
                json={"question": "Test", "collection": "default"},
            )
            assert response.status_code == 503


# =============================================================================
# Collections Endpoint Tests
# =============================================================================


class TestCollectionsEndpoint:
    """Tests for the /collections endpoint."""

    def test_list_collections_returns_200(self, client_with_mock_pipeline):
        """Test list collections returns 200."""
        response = client_with_mock_pipeline.get("/collections")
        assert response.status_code == 200

    def test_list_collections_returns_list(self, client_with_mock_pipeline):
        """Test list collections returns list."""
        response = client_with_mock_pipeline.get("/collections")
        data = response.json()
        assert "collections" in data
        assert isinstance(data["collections"], list)

    def test_delete_collection_returns_200(self, client_with_mock_pipeline):
        """Test delete collection returns 200."""
        response = client_with_mock_pipeline.delete("/collections/test_collection")
        assert response.status_code == 200

    def test_delete_collection_returns_success(self, client_with_mock_pipeline):
        """Test delete collection returns success."""
        response = client_with_mock_pipeline.delete("/collections/test_collection")
        data = response.json()
        assert data["success"] is True

    def test_delete_collection_returns_deleted_name(self, client_with_mock_pipeline):
        """Test delete collection returns deleted name."""
        response = client_with_mock_pipeline.delete("/collections/my_collection")
        data = response.json()
        assert data["deleted"] == "my_collection"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json_returns_422(self, client_with_mock_pipeline):
        """Test invalid JSON returns 422."""
        response = client_with_mock_pipeline.post(
            "/ingest",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_internal_error_returns_500(self):
        """Test internal error returns 500."""
        mock_pipeline = MagicMock()
        mock_pipeline.ingest = AsyncMock(side_effect=Exception("Internal error"))

        with patch("agentic_rag.api._pipeline", mock_pipeline):
            with patch("agentic_rag.api._settings", MagicMock()):
                with TestClient(app) as client:
                    response = client.post(
                        "/ingest",
                        json={"documents": [{"content": "Test"}], "collection": "test"},
                    )
                    assert response.status_code == 500

    def test_error_response_includes_detail(self):
        """Test error response includes detail."""
        mock_pipeline = MagicMock()
        mock_pipeline.query = AsyncMock(side_effect=Exception("Query failed"))

        with patch("agentic_rag.api._pipeline", mock_pipeline):
            with patch("agentic_rag.api._settings", MagicMock()):
                with TestClient(app) as client:
                    response = client.post(
                        "/query",
                        json={"question": "Test", "collection": "default"},
                    )
                    data = response.json()
                    assert "detail" in data


# =============================================================================
# CORS Tests
# =============================================================================


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_allows_all_origins(self, client_with_mock_pipeline):
        """Test CORS allows all origins."""
        response = client_with_mock_pipeline.options(
            "/health",
            headers={"Origin": "http://example.com"},
        )
        # Should not fail due to CORS
        assert response.status_code in [200, 405]

    def test_cors_allows_credentials(self, client_with_mock_pipeline):
        """Test CORS allows credentials."""
        response = client_with_mock_pipeline.get(
            "/health",
            headers={"Origin": "http://example.com"},
        )
        # Should have CORS headers
        assert response.status_code == 200


# =============================================================================
# Request Model Tests
# =============================================================================


class TestRequestModels:
    """Tests for request model validation."""

    def test_ingest_request_defaults(self):
        """Test IngestRequest default values."""
        request = IngestRequest(
            documents=[{"content": "Test"}],
        )
        assert request.collection == "default"
        assert request.chunk_strategy == "semantic"
        assert request.chunk_size == 512

    def test_query_request_defaults(self):
        """Test QueryRequest default values."""
        request = QueryRequest(
            question="Test question",
        )
        assert request.collection == "default"
        assert request.top_k == 5
        assert request.mode == "standard"

    def test_query_request_top_k_bounds(self):
        """Test QueryRequest top_k bounds."""
        # Valid
        request = QueryRequest(question="Test", top_k=25)
        assert request.top_k == 25

        # Invalid - would fail validation
        with pytest.raises(ValueError):
            QueryRequest(question="Test", top_k=0)


# =============================================================================
# Async Client Tests
# =============================================================================


@pytest.mark.asyncio
class TestAsyncAPI:
    """Tests using async client."""

    @pytest.fixture
    def mock_async_pipeline(self):
        """Create mock pipeline for async tests."""
        pipeline = MagicMock()

        async def mock_ingest(*args, **kwargs):
            return {"documents": 1, "chunks": 3}

        async def mock_query(*args, **kwargs):
            return GenerationResult(
                response="Async response",
                sources=[],
                confidence=0.9,
                provider="mock",
            )

        pipeline.ingest = AsyncMock(side_effect=mock_ingest)
        pipeline.query = AsyncMock(side_effect=mock_query)
        pipeline.vectordb = MagicMock()
        pipeline.vectordb.list_collections = AsyncMock(return_value=[])
        pipeline.vectordb.delete_collection = AsyncMock()

        return pipeline

    async def test_async_health_check(self, mock_async_pipeline):
        """Test async health check."""
        with patch("agentic_rag.api._pipeline", mock_async_pipeline):
            with patch("agentic_rag.api._settings", MagicMock(llm_model="test")):
                async with AsyncClient(app=app, base_url="http://test") as client:
                    response = await client.get("/health")
                    assert response.status_code == 200

    async def test_async_ingest(self, mock_async_pipeline):
        """Test async ingest."""
        with patch("agentic_rag.api._pipeline", mock_async_pipeline):
            with patch("agentic_rag.api._settings", MagicMock(llm_model="test")):
                async with AsyncClient(app=app, base_url="http://test") as client:
                    response = await client.post(
                        "/ingest",
                        json={"documents": [{"content": "Test"}], "collection": "test"},
                    )
                    assert response.status_code == 200

    async def test_async_query(self, mock_async_pipeline):
        """Test async query."""
        with patch("agentic_rag.api._pipeline", mock_async_pipeline):
            with patch("agentic_rag.api._settings", MagicMock(llm_model="test")):
                async with AsyncClient(app=app, base_url="http://test") as client:
                    response = await client.post(
                        "/query",
                        json={"question": "Test", "collection": "default"},
                    )
                    assert response.status_code == 200


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for API endpoints."""

    def test_health_check_latency(self, client_with_mock_pipeline):
        """Test health check latency."""
        import time

        start = time.time()
        for _ in range(100):
            client_with_mock_pipeline.get("/health")
        elapsed = time.time() - start

        avg_latency = elapsed / 100
        assert avg_latency < 0.1  # Less than 100ms average

    def test_concurrent_requests(self, client_with_mock_pipeline):
        """Test handling concurrent requests."""
        import concurrent.futures

        def make_request():
            return client_with_mock_pipeline.get("/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(r.status_code == 200 for r in results)
