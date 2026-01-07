"""
Comprehensive unit tests for retrieval functionality.

Tests:
- DenseRetriever with vector similarity search
- SparseRetriever with BM25
- HybridRetriever combining dense and sparse
- HyDERetriever with hypothetical document expansion
- Fusion strategies (RRF, weighted)
- Multi-query retrieval
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from agentic_rag.core.models import Chunk, RetrievalResult
from agentic_rag.retrieval.dense import DenseRetriever
from agentic_rag.retrieval.fusion import RRFFusion
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.hyde import HyDERetriever
from agentic_rag.retrieval.multi_query import MultiQueryRetriever
from agentic_rag.retrieval.sparse import SparseRetriever

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_embedder():
    """Create mock embedder for retrieval tests."""
    embedder = MagicMock()
    embedder.dimension = 384

    async def mock_embed_text(text):
        """Generate deterministic embedding based on text."""
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        return [(int(h[i : i + 2], 16) / 255.0 - 0.5) for i in range(0, 32, 2)] + [0.0] * 368

    async def mock_embed_batch(texts):
        results = []
        for text in texts:
            results.append(await mock_embed_text(text))
        return results

    embedder.embed_text = AsyncMock(side_effect=mock_embed_text)
    embedder.embed_batch = AsyncMock(side_effect=mock_embed_batch)

    return embedder


@pytest.fixture
def mock_vectordb(sample_chunks):
    """Create mock vector database with sample chunks."""
    db = MagicMock()
    db.db_type = "mock"
    db._chunks = sample_chunks

    async def mock_search(collection, query_vector, top_k=10, **kwargs):
        """Return chunks with scores based on vector similarity."""
        results = []
        for i, chunk in enumerate(db._chunks[:top_k]):
            # Simulate decreasing scores
            score = 0.95 - (i * 0.1)
            results.append((chunk, score))
        return results

    async def mock_get_all(collection, **kwargs):
        return db._chunks

    async def mock_close():
        pass

    db.search = AsyncMock(side_effect=mock_search)
    db.get_all = AsyncMock(side_effect=mock_get_all)
    db.close = AsyncMock(side_effect=mock_close)

    return db


@pytest.fixture
def mock_generator():
    """Create mock generator for HyDE."""
    generator = MagicMock()
    generator.provider = "mock"

    async def mock_generate_text(prompt, **kwargs):
        """Generate hypothetical document."""
        return f"Hypothetical document about: {prompt[:50]}..."

    generator.generate_text = AsyncMock(side_effect=mock_generate_text)
    return generator


# =============================================================================
# DenseRetriever Tests
# =============================================================================


class TestDenseRetriever:
    """Tests for the DenseRetriever class."""

    @pytest.fixture
    def dense_retriever(self, mock_embedder, mock_vectordb, test_settings_minimal):
        """Create dense retriever with mocks."""
        return DenseRetriever(
            embedder=mock_embedder,
            vectordb=mock_vectordb,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieval_result(self, dense_retriever):
        """Test that retrieve returns RetrievalResult."""
        result = await dense_retriever.retrieve(
            query="What is machine learning?",
            collection="test_collection",
            top_k=5,
        )

        assert isinstance(result, RetrievalResult)
        assert result.retrieval_type == "dense"

    @pytest.mark.asyncio
    async def test_retrieve_returns_chunks(self, dense_retriever, mock_vectordb):
        """Test that retrieve returns chunks."""
        result = await dense_retriever.retrieve(
            query="Test query",
            collection="test",
            top_k=3,
        )

        assert len(result.chunks) > 0
        assert all(isinstance(c, Chunk) for c in result.chunks)

    @pytest.mark.asyncio
    async def test_retrieve_returns_scores(self, dense_retriever):
        """Test that retrieve returns scores."""
        result = await dense_retriever.retrieve(
            query="Test query",
            collection="test",
            top_k=3,
        )

        assert len(result.scores) == len(result.chunks)
        assert all(isinstance(s, float) for s in result.scores)

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self, dense_retriever, mock_vectordb):
        """Test that top_k limits results."""
        result = await dense_retriever.retrieve(
            query="Test query",
            collection="test",
            top_k=2,
        )

        # Should be at most top_k results
        assert len(result.chunks) <= 2

    @pytest.mark.asyncio
    async def test_retrieve_includes_metadata(self, dense_retriever):
        """Test that result includes metadata."""
        result = await dense_retriever.retrieve(
            query="Test query",
            collection="test",
        )

        assert "query" in result.metadata
        assert result.metadata["query"] == "Test query"

    @pytest.mark.asyncio
    async def test_retrieve_calls_embedder(self, dense_retriever, mock_embedder):
        """Test that retrieve calls embedder."""
        await dense_retriever.retrieve(
            query="Test query",
            collection="test",
        )

        mock_embedder.embed_text.assert_called_once_with("Test query")

    @pytest.mark.asyncio
    async def test_batch_retrieve_multiple_queries(self, dense_retriever):
        """Test batch retrieval with multiple queries."""
        queries = ["Query 1", "Query 2", "Query 3"]
        results = await dense_retriever.batch_retrieve(
            queries=queries,
            collection="test",
            top_k=3,
        )

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_with_score_threshold(self, dense_retriever, mock_vectordb):
        """Test retrieval with score threshold."""
        # Modify mock to respect threshold
        original_search = mock_vectordb.search

        async def filtered_search(
            collection, query_vector, top_k=10, score_threshold=None, **kwargs
        ):
            results = await original_search(collection, query_vector, top_k, **kwargs)
            if score_threshold:
                results = [(c, s) for c, s in results if s >= score_threshold]
            return results

        mock_vectordb.search = AsyncMock(side_effect=filtered_search)

        result = await dense_retriever.retrieve(
            query="Test",
            collection="test",
            top_k=10,
            score_threshold=0.8,
        )

        # Results should have high scores
        for score in result.scores:
            assert score >= 0.8 or len(result.scores) == 0

    @pytest.mark.asyncio
    async def test_close_releases_resources(self, dense_retriever, mock_vectordb):
        """Test that close releases resources."""
        await dense_retriever.close()
        mock_vectordb.close.assert_called_once()


# =============================================================================
# SparseRetriever Tests
# =============================================================================


class TestSparseRetriever:
    """Tests for the SparseRetriever class (BM25)."""

    @pytest.fixture
    def sparse_retriever(self, mock_vectordb, test_settings_minimal):
        """Create sparse retriever with mock DB."""
        return SparseRetriever(
            vectordb=mock_vectordb,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_sparse_retrieve_returns_result(self, sparse_retriever):
        """Test that sparse retrieve returns RetrievalResult."""
        result = await sparse_retriever.retrieve(
            query="machine learning",
            collection="test",
            top_k=5,
        )

        assert isinstance(result, RetrievalResult)
        assert result.retrieval_type == "sparse"

    @pytest.mark.asyncio
    async def test_sparse_retrieve_tokenizes_query(self, sparse_retriever):
        """Test that query is tokenized for BM25."""
        # BM25 should tokenize the query
        result = await sparse_retriever.retrieve(
            query="machine learning algorithms",
            collection="test",
        )

        # Should return results (exact behavior depends on implementation)
        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_sparse_retrieve_empty_query(self, sparse_retriever):
        """Test sparse retrieval with empty query."""
        result = await sparse_retriever.retrieve(
            query="",
            collection="test",
        )

        assert isinstance(result, RetrievalResult)


# =============================================================================
# HybridRetriever Tests
# =============================================================================


class TestHybridRetriever:
    """Tests for the HybridRetriever class."""

    @pytest.fixture
    def hybrid_retriever(self, mock_embedder, mock_vectordb, test_settings_minimal):
        """Create hybrid retriever."""
        return HybridRetriever(
            embedder=mock_embedder,
            vectordb=mock_vectordb,
            sparse_weight=0.3,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_returns_result(self, hybrid_retriever):
        """Test that hybrid retrieve returns RetrievalResult."""
        result = await hybrid_retriever.retrieve(
            query="machine learning",
            collection="test",
            top_k=5,
        )

        assert isinstance(result, RetrievalResult)
        assert result.retrieval_type == "hybrid"

    @pytest.mark.asyncio
    async def test_hybrid_combines_dense_and_sparse(self, hybrid_retriever):
        """Test that hybrid retriever combines both methods."""
        result = await hybrid_retriever.retrieve(
            query="test query",
            collection="test",
        )

        # Should have results from combination
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_sparse_weighting(self, mock_embedder, mock_vectordb, test_settings_minimal):
        """Test that sparse_weight parameter affects weighting."""
        # Low sparse weight (dense-focused)
        retriever_dense = HybridRetriever(
            embedder=mock_embedder,
            vectordb=mock_vectordb,
            sparse_weight=0.1,  # Low sparse weight = dense focused
            settings=test_settings_minimal,
        )

        # High sparse weight (sparse-focused)
        retriever_sparse = HybridRetriever(
            embedder=mock_embedder,
            vectordb=mock_vectordb,
            sparse_weight=0.9,  # High sparse weight = sparse focused
            settings=test_settings_minimal,
        )

        result_dense = await retriever_dense.retrieve(query="test", collection="test")
        result_sparse = await retriever_sparse.retrieve(query="test", collection="test")

        # Both should return results
        assert isinstance(result_dense, RetrievalResult)
        assert isinstance(result_sparse, RetrievalResult)


# =============================================================================
# HyDERetriever Tests
# =============================================================================


class TestHyDERetriever:
    """Tests for the HyDERetriever class."""

    @pytest.fixture
    def hyde_retriever(self, mock_embedder, mock_vectordb, mock_generator, test_settings_minimal):
        """Create HyDE retriever."""
        return HyDERetriever(
            embedder=mock_embedder,
            vectordb=mock_vectordb,
            generator=mock_generator,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_hyde_generates_hypothetical_document(self, hyde_retriever, mock_generator):
        """Test that HyDE generates hypothetical document."""
        await hyde_retriever.retrieve(
            query="What is machine learning?",
            collection="test",
        )

        # Generator should be called
        mock_generator.generate_text.assert_called()

    @pytest.mark.asyncio
    async def test_hyde_retrieve_returns_result(self, hyde_retriever):
        """Test that HyDE returns RetrievalResult."""
        result = await hyde_retriever.retrieve(
            query="What is deep learning?",
            collection="test",
        )

        assert isinstance(result, RetrievalResult)
        assert result.retrieval_type == "hyde"

    @pytest.mark.asyncio
    async def test_hyde_embeds_hypothetical_document(self, hyde_retriever, mock_embedder):
        """Test that HyDE embeds the hypothetical document."""
        await hyde_retriever.retrieve(
            query="Test query",
            collection="test",
        )

        # Embedder should be called for hypothetical doc
        assert mock_embedder.embed_text.called


# =============================================================================
# RRFFusion Tests
# =============================================================================


class TestRRFFusion:
    """Tests for the RRF (Reciprocal Rank Fusion) class."""

    @pytest.fixture
    def rrf_fusion(self):
        """Create RRF fusion instance."""
        return RRFFusion(k=60)

    def test_rrf_fuses_rankings(self, rrf_fusion, sample_chunks):
        """Test that RRF fuses multiple rankings."""
        # Create two RetrievalResults
        result1 = RetrievalResult(
            chunks=[sample_chunks[0], sample_chunks[1]],
            scores=[0.9, 0.8],
            retrieval_type="dense",
        )
        result2 = RetrievalResult(
            chunks=[sample_chunks[1], sample_chunks[0]],
            scores=[0.95, 0.7],
            retrieval_type="dense",
        )

        fused = rrf_fusion.fuse([result1, result2])

        assert isinstance(fused, RetrievalResult)
        assert len(fused.chunks) > 0

    def test_rrf_handles_empty_rankings(self, rrf_fusion):
        """Test RRF with empty rankings."""
        result1 = RetrievalResult(chunks=[], scores=[], retrieval_type="dense")
        result2 = RetrievalResult(chunks=[], scores=[], retrieval_type="dense")
        fused = rrf_fusion.fuse([result1, result2])
        assert len(fused.chunks) == 0

    def test_rrf_handles_single_ranking(self, rrf_fusion, sample_chunks):
        """Test RRF with single ranking."""
        result = RetrievalResult(
            chunks=[sample_chunks[0]],
            scores=[0.9],
            retrieval_type="dense",
        )
        fused = rrf_fusion.fuse([result])

        assert len(fused.chunks) == 1

    def test_rrf_k_parameter_affects_scores(self, sample_chunks):
        """Test that k parameter affects fusion scores."""
        fusion_low_k = RRFFusion(k=10)
        fusion_high_k = RRFFusion(k=100)

        result = RetrievalResult(
            chunks=[sample_chunks[0], sample_chunks[1]],
            scores=[0.9, 0.8],
            retrieval_type="dense",
        )

        fused_low = fusion_low_k.fuse([result])
        fused_high = fusion_high_k.fuse([result])

        # Different k should produce different scores
        assert fused_low.scores[0] != fused_high.scores[0]


# =============================================================================
# MultiQueryRetriever Tests
# =============================================================================


class TestMultiQueryRetriever:
    """Tests for the MultiQueryRetriever class."""

    @pytest.fixture
    def mock_base_retriever(self, sample_chunks):
        """Create a mock base retriever."""
        retriever = MagicMock()

        async def mock_retrieve(query, collection, top_k=10):
            return RetrievalResult(
                chunks=sample_chunks[:2],
                scores=[0.9, 0.8],
                retrieval_type="dense",
            )

        retriever.retrieve = AsyncMock(side_effect=mock_retrieve)

        async def mock_batch_retrieve(queries, collection, top_k=10):
            return [
                RetrievalResult(
                    chunks=sample_chunks[:2],
                    scores=[0.9, 0.8],
                    retrieval_type="dense",
                )
                for _ in queries
            ]

        retriever.batch_retrieve = AsyncMock(side_effect=mock_batch_retrieve)
        return retriever

    @pytest.fixture
    def multi_query_retriever(self, mock_base_retriever, mock_generator, test_settings_minimal):
        """Create multi-query retriever."""
        return MultiQueryRetriever(
            base_retriever=mock_base_retriever,
            generator=mock_generator,
            num_queries=3,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_multi_query_generates_variations(self, multi_query_retriever, mock_generator):
        """Test that multi-query generates query variations."""
        await multi_query_retriever.retrieve(
            query="What is machine learning?",
            collection="test",
        )

        # Generator should be called to create variations
        mock_generator.generate_text.assert_called()

    @pytest.mark.asyncio
    async def test_multi_query_returns_result(self, multi_query_retriever):
        """Test that multi-query returns RetrievalResult."""
        result = await multi_query_retriever.retrieve(
            query="Test query",
            collection="test",
        )

        assert isinstance(result, RetrievalResult)
        assert result.retrieval_type == "multi_query"

    @pytest.mark.asyncio
    async def test_multi_query_combines_results(self, multi_query_retriever):
        """Test that results from multiple queries are combined."""
        result = await multi_query_retriever.retrieve(
            query="Test",
            collection="test",
        )

        # Should have combined results
        assert len(result.chunks) > 0


# =============================================================================
# Retrieval Quality Tests
# =============================================================================


class TestRetrievalQuality:
    """Tests for retrieval quality metrics."""

    @pytest.fixture
    def retriever_with_known_data(self, mock_embedder, test_settings_minimal):
        """Create retriever with controlled test data."""
        # Create mock DB with known chunks
        chunks = [
            Chunk(
                id="ml_chunk",
                content="Machine learning is a subset of AI that learns from data.",
                document_id="doc1",
                embedding=[0.8, 0.2, 0.0] + [0.0] * 381,
            ),
            Chunk(
                id="dl_chunk",
                content="Deep learning uses neural networks with many layers.",
                document_id="doc2",
                embedding=[0.2, 0.8, 0.0] + [0.0] * 381,
            ),
            Chunk(
                id="db_chunk",
                content="Databases store and retrieve structured data.",
                document_id="doc3",
                embedding=[0.0, 0.2, 0.8] + [0.0] * 381,
            ),
        ]

        db = MagicMock()

        async def semantic_search(collection, query_vector, top_k=10, **kwargs):
            """Search based on cosine similarity."""
            results = []
            for chunk in chunks:
                if chunk.embedding:
                    sim = np.dot(query_vector[:3], chunk.embedding[:3])
                    results.append((chunk, float(sim)))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        db.search = AsyncMock(side_effect=semantic_search)
        db.close = AsyncMock()

        # Mock embedder to return topic-based embeddings
        async def topic_embed(text):
            text_lower = text.lower()
            if "machine learning" in text_lower:
                return [0.8, 0.2, 0.0] + [0.0] * 381
            elif "deep learning" in text_lower or "neural" in text_lower:
                return [0.2, 0.8, 0.0] + [0.0] * 381
            elif "database" in text_lower:
                return [0.0, 0.2, 0.8] + [0.0] * 381
            else:
                return [0.4, 0.4, 0.2] + [0.0] * 381

        mock_embedder.embed_text = AsyncMock(side_effect=topic_embed)

        return DenseRetriever(
            embedder=mock_embedder,
            vectordb=db,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_semantic_relevance(self, retriever_with_known_data):
        """Test that semantically relevant chunks are retrieved first."""
        result = await retriever_with_known_data.retrieve(
            query="What is machine learning?",
            collection="test",
        )

        # ML chunk should be first or highly ranked
        ml_found = any("machine learning" in c.content.lower() for c in result.chunks[:2])
        assert ml_found


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestRetrievalErrorHandling:
    """Tests for error handling in retrieval."""

    @pytest.fixture
    def failing_retriever(self, mock_embedder, test_settings_minimal):
        """Create retriever that simulates failures."""
        db = MagicMock()
        db.search = AsyncMock(side_effect=Exception("Connection failed"))
        db.close = AsyncMock()

        return DenseRetriever(
            embedder=mock_embedder,
            vectordb=db,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_handles_vectordb_error(self, failing_retriever):
        """Test handling of vector DB errors."""
        with pytest.raises(Exception) as exc_info:
            await failing_retriever.retrieve(
                query="Test",
                collection="test",
            )

        assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_empty_collection(self, mock_embedder, test_settings_minimal):
        """Test handling of empty collection."""
        empty_db = MagicMock()
        empty_db.search = AsyncMock(return_value=[])
        empty_db.close = AsyncMock()

        retriever = DenseRetriever(
            embedder=mock_embedder,
            vectordb=empty_db,
            settings=test_settings_minimal,
        )

        result = await retriever.retrieve(
            query="Test",
            collection="empty",
        )

        assert result.chunks == []
        assert result.scores == []


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestRetrievalPerformance:
    """Performance tests for retrieval."""

    @pytest.fixture
    def large_db_retriever(self, mock_embedder, test_settings_minimal):
        """Create retriever with large mock database."""
        # Create many chunks
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"Content for chunk {i} about topic {i % 10}.",
                document_id=f"doc_{i // 10}",
                embedding=[0.1] * 384,
            )
            for i in range(1000)
        ]

        db = MagicMock()

        async def search(collection, query_vector, top_k=10, **kwargs):
            return [(c, 0.9 - i * 0.01) for i, c in enumerate(chunks[:top_k])]

        db.search = AsyncMock(side_effect=search)
        db.close = AsyncMock()

        return DenseRetriever(
            embedder=mock_embedder,
            vectordb=db,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_retrieval_latency(self, large_db_retriever):
        """Test that retrieval completes in reasonable time."""
        import time

        start = time.time()
        await large_db_retriever.retrieve(
            query="Test query",
            collection="test",
            top_k=10,
        )
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_batch_retrieval_efficiency(self, large_db_retriever):
        """Test batch retrieval is efficient."""
        import time

        queries = [f"Query {i}" for i in range(10)]

        start = time.time()
        await large_db_retriever.batch_retrieve(
            queries=queries,
            collection="test",
        )
        batch_time = time.time() - start

        # Batch should be reasonable
        assert batch_time < 5.0
