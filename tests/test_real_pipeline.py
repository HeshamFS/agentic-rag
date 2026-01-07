"""
Real End-to-End Integration Tests for RAG Pipeline.

Uses REAL:
- Academic PDFs (Attention Is All You Need, BERT, GPT-3, etc.)
- Qwen3 Embedding Model
- Qdrant Cloud Vector Database
- LLM Generation (Gemini/OpenAI/Claude)

Tests cover:
- PDF ingestion and chunking
- Vector database operations
- Full pipeline integration
- Redis caching
- Context compression
- RAPTOR hierarchical chunking

NO MOCKS for integration tests - unit tests use mocks for speed.
"""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from agentic_rag.chunking.semantic import SemanticChunker
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk
from agentic_rag.embeddings.qwen3_embedder import create_embedder
from agentic_rag.ingestion.file_loader import FileLoader
from agentic_rag.retrieval.dense import DenseRetriever
from agentic_rag.vectordb import QdrantVectorDB

# =============================================================================
# Test Configuration
# =============================================================================

# Collection name for tests
TEST_COLLECTION = "agentic_rag_integration_tests"

# Path to test PDFs
PAPERS_DIR = Path(__file__).parent / "test_data" / "papers"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def settings():
    """Get real settings."""
    return get_settings()


@pytest.fixture(scope="module")
def embedder():
    """Create real embedding model - shared across module."""
    return create_embedder("small")


@pytest.fixture
def vectordb(settings):
    """Create real Qdrant connection (function-scoped to avoid event loop issues)."""
    return QdrantVectorDB(settings=settings)


@pytest.fixture(scope="module")
def file_loader():
    """Create real file loader for PDFs."""
    return FileLoader()


@pytest.fixture(scope="module")
def chunker(embedder):
    """Create real semantic chunker."""
    return SemanticChunker(
        embedder=embedder,
        chunk_size=512,
        min_chunk_size=100,
        max_chunk_size=2000,
    )


@pytest.fixture(scope="module")
def retriever(embedder, vectordb, settings):
    """Create real retriever."""
    return DenseRetriever(
        embedder=embedder,
        vectordb=vectordb,
        settings=settings,
    )


# =============================================================================
# PDF Ingestion Tests
# =============================================================================


class TestRealPDFIngestion:
    """Test PDF ingestion with real academic papers."""

    def test_pdf_files_exist(self):
        """Verify test PDFs are available."""
        expected_papers = [
            "attention_is_all_you_need.pdf",
            "bert_paper.pdf",
            "rag_paper.pdf",
        ]

        for paper in expected_papers:
            path = PAPERS_DIR / paper
            assert path.exists(), f"Missing test paper: {paper}"

    def test_ingest_attention_paper(self, file_loader):
        """Test ingesting 'Attention Is All You Need' paper."""
        pdf_path = PAPERS_DIR / "attention_is_all_you_need.pdf"

        if not pdf_path.exists():
            pytest.skip("Attention paper not downloaded")

        result = file_loader.load(pdf_path)

        assert result.success, f"Failed to load PDF: {result.error}"
        assert result.document is not None
        assert len(result.document.content) > 1000, "Document content too short"
        assert "attention" in result.document.content.lower()

    def test_ingest_bert_paper(self, file_loader):
        """Test ingesting BERT paper."""
        pdf_path = PAPERS_DIR / "bert_paper.pdf"

        if not pdf_path.exists():
            pytest.skip("BERT paper not downloaded")

        result = file_loader.load(pdf_path)

        assert result.success, f"Failed to load PDF: {result.error}"
        assert result.document is not None
        assert len(result.document.content) > 1000

    def test_ingest_rag_paper(self, file_loader):
        """Test ingesting RAG paper."""
        pdf_path = PAPERS_DIR / "rag_paper.pdf"

        if not pdf_path.exists():
            pytest.skip("RAG paper not downloaded")

        result = file_loader.load(pdf_path)

        assert result.success, f"Failed to load PDF: {result.error}"
        assert result.document is not None
        assert len(result.document.content) > 1000


# =============================================================================
# Chunking Tests with Real Content
# =============================================================================


class TestRealChunking:
    """Test chunking with real document content."""

    @pytest.mark.asyncio
    async def test_chunk_attention_paper(self, file_loader, chunker):
        """Test chunking the Attention paper."""
        pdf_path = PAPERS_DIR / "attention_is_all_you_need.pdf"

        if not pdf_path.exists():
            pytest.skip("Attention paper not downloaded")

        result = file_loader.load(pdf_path)
        assert result.success and result.document
        document = result.document

        chunks = await chunker.chunk_async(document)

        assert len(chunks) > 5, "Should create multiple chunks"

        # Verify chunk quality
        for chunk in chunks:
            assert len(chunk.content) >= 50, "Chunks too small"
            assert len(chunk.content) <= 3000, "Chunks too large"
            assert chunk.document_id == document.id

    @pytest.mark.asyncio
    async def test_chunks_cover_document(self, file_loader, chunker):
        """Test that chunks cover the full document."""
        pdf_path = PAPERS_DIR / "attention_is_all_you_need.pdf"

        if not pdf_path.exists():
            pytest.skip("Attention paper not downloaded")

        result = file_loader.load(pdf_path)
        assert result.success and result.document
        document = result.document

        chunks = await chunker.chunk_async(document)

        # Check that key terms from document appear in chunks
        all_chunk_text = " ".join(c.content for c in chunks).lower()

        key_terms = ["attention", "transformer", "encoder", "decoder"]
        for term in key_terms:
            assert term in all_chunk_text, f"Key term '{term}' not found in chunks"


# =============================================================================
# Vector Database Tests
# =============================================================================


class TestRealVectorDB:
    """Test Qdrant operations with real data."""

    @pytest.mark.asyncio
    async def test_create_and_delete_collection(self, vectordb, embedder):
        """Test creating and deleting a collection."""
        collection_name = f"{TEST_COLLECTION}_create_delete"

        # Create
        await vectordb.create_collection(
            name=collection_name,
            dimension=embedder.dimension,
        )

        # Verify exists
        exists = await vectordb.collection_exists(collection_name)
        assert exists, "Collection should exist after creation"

        # Delete
        await vectordb.delete_collection(collection_name)

        # Verify deleted
        exists = await vectordb.collection_exists(collection_name)
        assert not exists, "Collection should not exist after deletion"

    @pytest.mark.asyncio
    async def test_upsert_and_search(self, vectordb, embedder):
        """Test inserting chunks and searching."""
        collection_name = f"{TEST_COLLECTION}_upsert_search"

        try:
            # Create collection
            await vectordb.create_collection(
                name=collection_name,
                dimension=embedder.dimension,
            )

            # Create test chunks with embeddings
            texts = [
                "The Transformer model uses self-attention mechanisms.",
                "BERT is trained on masked language modeling.",
                "GPT generates text using next token prediction.",
                "Cooking recipes require precise measurements.",
            ]

            embeddings = await embedder.embed_batch(texts)

            chunks = [
                Chunk(
                    id=str(uuid.uuid4()),
                    content=text,
                    document_id=str(uuid.uuid4()),
                    embedding=emb,
                )
                for i, (text, emb) in enumerate(zip(texts, embeddings, strict=False))
            ]

            # Upsert
            await vectordb.upsert(collection_name, chunks)

            # Search for ML-related query
            query = "What is the Transformer architecture?"
            query_vector = await embedder.embed_text(query)

            results = await vectordb.search(
                collection=collection_name,
                query_vector=query_vector,
                top_k=3,
            )

            assert len(results) > 0, "Should return search results"

            # First result should be about Transformers (results are (chunk, score) tuples)
            first_chunk, first_score = results[0]
            assert (
                "transformer" in first_chunk.content.lower()
                or "attention" in first_chunk.content.lower()
            )

        finally:
            # Cleanup
            await vectordb.delete_collection(collection_name)

    @pytest.mark.asyncio
    async def test_semantic_search_quality(self, vectordb, embedder):
        """Test that semantic search returns relevant results."""
        collection_name = f"{TEST_COLLECTION}_semantic"

        try:
            await vectordb.create_collection(
                name=collection_name,
                dimension=embedder.dimension,
            )

            # Create diverse test chunks
            texts = [
                "Neural networks learn patterns from training data.",
                "Python is a popular programming language.",
                "Machine learning models require large datasets.",
                "Database optimization improves query performance.",
                "Deep learning uses multiple neural network layers.",
            ]

            embeddings = await embedder.embed_batch(texts)
            doc_id = str(uuid.uuid4())
            chunks = [
                Chunk(id=str(uuid.uuid4()), content=t, document_id=doc_id, embedding=e)
                for i, (t, e) in enumerate(zip(texts, embeddings, strict=False))
            ]

            await vectordb.upsert(collection_name, chunks)

            # Query about ML
            query = "How do neural networks learn?"
            query_emb = await embedder.embed_text(query)

            results = await vectordb.search(
                collection=collection_name,
                query_vector=query_emb,
                top_k=3,
            )

            # Top results should be ML-related (results are (chunk, score) tuples)
            ml_terms = ["neural", "learning", "machine", "deep"]
            top_chunk, top_score = results[0]
            top_result_text = top_chunk.content.lower()

            has_ml_term = any(term in top_result_text for term in ml_terms)
            assert has_ml_term, f"Top result should be ML-related: {top_chunk.content}"

        finally:
            await vectordb.delete_collection(collection_name)


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


class TestFullPipeline:
    """End-to-end pipeline tests with real components."""

    @pytest.mark.asyncio
    async def test_ingest_and_retrieve(self, file_loader, chunker, embedder, vectordb):
        """Test full ingest -> chunk -> embed -> store -> retrieve flow."""
        pdf_path = PAPERS_DIR / "attention_is_all_you_need.pdf"

        if not pdf_path.exists():
            pytest.skip("Attention paper not downloaded")

        collection_name = f"{TEST_COLLECTION}_full_pipeline"

        try:
            # 1. Ingest PDF
            result = file_loader.load(pdf_path)
            assert result.success and result.document
            document = result.document

            # 2. Chunk
            chunks = await chunker.chunk_async(document)
            assert len(chunks) > 0

            # 3. Embed chunks
            for chunk in chunks:
                chunk.embedding = await embedder.embed_text(chunk.content)

            # 4. Create collection and store
            await vectordb.create_collection(
                name=collection_name,
                dimension=embedder.dimension,
            )
            await vectordb.upsert(collection_name, chunks)

            # 5. Search
            query = "What is multi-head attention?"
            query_vector = await embedder.embed_text(query)

            results = await vectordb.search(
                collection=collection_name,
                query_vector=query_vector,
                top_k=5,
            )

            # Verify results are relevant (results are (chunk, score) tuples)
            assert len(results) > 0
            all_results_text = " ".join(chunk.content.lower() for chunk, score in results)
            assert "attention" in all_results_text, "Search results should contain 'attention'"

        finally:
            await vectordb.delete_collection(collection_name)

    @pytest.mark.asyncio
    async def test_retrieval_quality_on_paper(self, file_loader, chunker, embedder, vectordb):
        """Test retrieval quality with specific questions about the paper."""
        pdf_path = PAPERS_DIR / "attention_is_all_you_need.pdf"

        if not pdf_path.exists():
            pytest.skip("Attention paper not downloaded")

        collection_name = f"{TEST_COLLECTION}_quality"

        try:
            # Ingest and index
            result = file_loader.load(pdf_path)
            assert result.success and result.document
            document = result.document

            chunks = await chunker.chunk_async(document)

            for chunk in chunks:
                chunk.embedding = await embedder.embed_text(chunk.content)

            await vectordb.create_collection(
                name=collection_name,
                dimension=embedder.dimension,
            )
            await vectordb.upsert(collection_name, chunks)

            # Test specific questions
            test_queries = [
                (
                    "What is the Transformer architecture?",
                    ["transformer", "attention", "encoder", "decoder"],
                ),
                ("How does self-attention work?", ["attention", "query", "key", "value"]),
                ("What are the advantages of attention?", ["attention", "parallel", "sequence"]),
            ]

            for query, expected_terms in test_queries:
                query_emb = await embedder.embed_text(query)
                results = await vectordb.search(
                    collection=collection_name,
                    query_vector=query_emb,
                    top_k=3,
                )

                # Results are (chunk, score) tuples
                results_text = " ".join(chunk.content.lower() for chunk, score in results)

                # At least one expected term should appear
                has_expected = any(term in results_text for term in expected_terms)
                assert has_expected, f"Query '{query}' should return results with {expected_terms}"

        finally:
            await vectordb.delete_collection(collection_name)


# =============================================================================
# Multi-Paper Tests
# =============================================================================


class TestMultiPaperIngestion:
    """Test ingesting and querying across multiple papers."""

    @pytest.mark.asyncio
    async def test_ingest_multiple_papers(self, file_loader, chunker, embedder, vectordb):
        """Test ingesting multiple papers into one collection."""
        papers = [
            "attention_is_all_you_need.pdf",
            "bert_paper.pdf",
            "rag_paper.pdf",
        ]

        available_papers = [p for p in papers if (PAPERS_DIR / p).exists()]

        if len(available_papers) < 2:
            pytest.skip("Need at least 2 papers for multi-paper test")

        collection_name = f"{TEST_COLLECTION}_multi_paper"

        try:
            await vectordb.create_collection(
                name=collection_name,
                dimension=embedder.dimension,
            )

            total_chunks = 0
            for paper in available_papers:
                pdf_path = PAPERS_DIR / paper

                result = file_loader.load(pdf_path)
                if not result.success or not result.document:
                    continue

                document = result.document
                chunks = await chunker.chunk_async(document)

                for chunk in chunks:
                    chunk.embedding = await embedder.embed_text(chunk.content)
                    chunk.metadata["source_paper"] = paper

                await vectordb.upsert(collection_name, chunks)
                total_chunks += len(chunks)

            assert total_chunks > 10, "Should have many chunks from multiple papers"

            # Query that should match BERT paper
            query = "What is masked language modeling?"
            query_emb = await embedder.embed_text(query)

            results = await vectordb.search(
                collection=collection_name,
                query_vector=query_emb,
                top_k=5,
            )

            assert len(results) > 0

        finally:
            await vectordb.delete_collection(collection_name)


# =============================================================================
# Performance Benchmarks
# =============================================================================


class TestPipelinePerformance:
    """Performance tests for the pipeline."""

    @pytest.mark.asyncio
    async def test_ingestion_speed(self, file_loader, chunker, embedder):
        """Benchmark ingestion speed."""
        import time

        pdf_path = PAPERS_DIR / "attention_is_all_you_need.pdf"

        if not pdf_path.exists():
            pytest.skip("Attention paper not downloaded")

        # Measure PDF processing
        start = time.time()
        result = file_loader.load(pdf_path)
        pdf_time = time.time() - start

        assert result.success and result.document
        document = result.document

        # Measure chunking
        start = time.time()
        chunks = await chunker.chunk_async(document)
        chunk_time = time.time() - start

        # Measure embedding
        start = time.time()
        await embedder.embed_batch([c.content for c in chunks])
        embed_time = time.time() - start

        print("\n=== Pipeline Performance ===")
        print(f"PDF Processing: {pdf_time:.2f}s")
        print(f"Chunking ({len(chunks)} chunks): {chunk_time:.2f}s")
        chunks_per_sec = len(chunks) / embed_time if embed_time > 0 else float("inf")
        print(f"Embedding: {embed_time:.2f}s ({chunks_per_sec:.1f} chunks/sec)")
        print(f"Total: {pdf_time + chunk_time + embed_time:.2f}s")

        # Reasonable performance expectations
        assert pdf_time < 30, "PDF processing too slow"
        assert chunk_time < 60, "Chunking too slow"  # Increased for semantic chunking
        assert embed_time < 120, "Embedding too slow"  # Increased for CPU

    @pytest.mark.asyncio
    async def test_search_latency(self, embedder, vectordb):
        """Benchmark search latency."""
        import time

        collection_name = f"{TEST_COLLECTION}_latency"

        try:
            await vectordb.create_collection(
                name=collection_name,
                dimension=embedder.dimension,
            )

            # Add some test data
            texts = [f"Test document number {i} with content." for i in range(100)]
            embeddings = await embedder.embed_batch(texts)

            doc_id = str(uuid.uuid4())
            chunks = [
                Chunk(id=str(uuid.uuid4()), content=t, document_id=doc_id, embedding=e)
                for i, (t, e) in enumerate(zip(texts, embeddings, strict=False))
            ]

            await vectordb.upsert(collection_name, chunks)

            # Measure search latency
            query = "Find relevant documents"
            query_emb = await embedder.embed_text(query)

            latencies = []
            for _ in range(10):
                start = time.time()
                await vectordb.search(
                    collection=collection_name,
                    query_vector=query_emb,
                    top_k=10,
                )
                latencies.append(time.time() - start)

            avg_latency = sum(latencies) / len(latencies)
            print(f"\nSearch latency (avg of 10): {avg_latency * 1000:.1f}ms")

            assert avg_latency < 1.0, f"Search too slow: {avg_latency}s"

        finally:
            await vectordb.delete_collection(collection_name)


# =============================================================================
# Redis Caching Tests
# =============================================================================


class TestRedisConfig:
    """Test Redis configuration parsing."""

    def test_default_config(self):
        """Default config uses localhost:6379."""
        from agentic_rag.caching.redis_cache import RedisConfig

        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.prefix == "rag:cache:"
        assert config.ssl is False

    def test_from_url_basic(self):
        """Parse basic Redis URL."""
        from agentic_rag.caching.redis_cache import RedisConfig

        config = RedisConfig.from_url("redis://myhost:6380/1")
        assert config.host == "myhost"
        assert config.port == 6380
        assert config.db == 1

    def test_from_url_with_password(self):
        """Parse Redis URL with password."""
        from agentic_rag.caching.redis_cache import RedisConfig

        config = RedisConfig.from_url("redis://:secret@myhost:6380/2")
        assert config.host == "myhost"
        assert config.port == 6380
        assert config.db == 2
        assert config.password == "secret"

    def test_from_url_ssl(self):
        """Parse rediss:// URL enables SSL."""
        from agentic_rag.caching.redis_cache import RedisConfig

        config = RedisConfig.from_url("rediss://secure.host:6379/0")
        assert config.ssl is True
        assert config.host == "secure.host"


class TestSemanticCacheUnit:
    """Unit tests for semantic cache (with mocks)."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder for testing."""
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 768)
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 768])
        return embedder

    def test_cache_initialization(self, mock_embedder):
        """Cache initializes with correct settings."""
        from agentic_rag.caching.semantic_cache import SemanticCache

        cache = SemanticCache(
            embedder=mock_embedder,
            similarity_threshold=0.9,
            ttl_seconds=1800,
        )
        assert cache._threshold == 0.9
        assert cache._ttl == 1800

    def test_cache_stats_empty(self, mock_embedder):
        """Empty cache returns zero stats."""
        from agentic_rag.caching.semantic_cache import SemanticCache

        cache = SemanticCache(embedder=mock_embedder)
        stats = cache.stats()

        assert stats["entries"] == 0
        assert stats["total_hits"] == 0

    def test_cache_clear(self, mock_embedder):
        """Clear removes all entries."""
        from agentic_rag.caching.semantic_cache import SemanticCache

        cache = SemanticCache(embedder=mock_embedder)
        cache._cache["test"] = "value"
        assert len(cache._cache) == 1

        cache.clear()
        assert len(cache._cache) == 0


class TestRedisSemanticCache:
    """Test Redis-backed semantic cache."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder for testing."""
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 768)
        return embedder

    def test_init_with_url(self, mock_embedder):
        """Initialize with Redis URL."""
        from agentic_rag.caching.redis_cache import RedisSemanticCache

        cache = RedisSemanticCache(
            embedder=mock_embedder,
            redis_url="redis://localhost:6379/0",
        )
        assert cache._redis_config.host == "localhost"
        assert cache._redis_config.port == 6379
        assert cache._redis_config.db == 0

    def test_init_with_config(self, mock_embedder):
        """Initialize with RedisConfig object."""
        from agentic_rag.caching.redis_cache import RedisConfig, RedisSemanticCache

        config = RedisConfig(host="redis.example.com", port=6380)
        cache = RedisSemanticCache(
            embedder=mock_embedder,
            redis_config=config,
        )
        assert cache._redis_config.host == "redis.example.com"
        assert cache._redis_config.port == 6380

    def test_not_connected_initially(self, mock_embedder):
        """Cache is not connected until connect() is called."""
        from agentic_rag.caching.redis_cache import RedisSemanticCache

        cache = RedisSemanticCache(embedder=mock_embedder)
        assert cache.is_connected is False


class TestCacheEntry:
    """Test CacheEntry model."""

    def test_create_entry(self):
        """Create cache entry with required fields."""
        from agentic_rag.caching.semantic_cache import CacheEntry

        entry = CacheEntry(
            query="What is RAG?",
            query_embedding=[0.1] * 10,
            response="RAG is Retrieval Augmented Generation.",
        )
        assert entry.query == "What is RAG?"
        assert entry.response == "RAG is Retrieval Augmented Generation."
        assert entry.hits == 0

    def test_entry_defaults(self):
        """Entry has sensible defaults."""
        from agentic_rag.caching.semantic_cache import CacheEntry

        entry = CacheEntry(
            query="test",
            query_embedding=[0.1],
            response="response",
        )
        assert entry.sources == []
        assert entry.metadata == {}
        assert entry.hits == 0


class TestDiskSemanticCache:
    """Test disk-based persistent cache."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder for testing."""
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 768)
        return embedder

    def test_init_creates_directory(self, mock_embedder, tmp_path):
        """Disk cache creates cache directory."""
        from agentic_rag.caching.semantic_cache import DiskSemanticCache

        cache_dir = tmp_path / "test_cache"
        cache = DiskSemanticCache(
            embedder=mock_embedder,
            cache_dir=str(cache_dir),
        )
        assert hasattr(cache, "_disk_cache")
        assert cache._disk_cache is not None
        cache.close()


class TestCacheConfigSettings:
    """Test cache-related configuration settings."""

    def test_cache_settings_exist(self):
        """Config has all cache-related settings."""

        settings = Settings()
        assert hasattr(settings, "cache_backend")
        assert hasattr(settings, "cache_similarity_threshold")
        assert hasattr(settings, "cache_ttl_seconds")
        assert hasattr(settings, "redis_url")

    def test_cache_defaults(self):
        """Cache settings have sensible defaults."""

        settings = Settings()
        assert settings.cache_backend == "memory"
        assert settings.cache_similarity_threshold == 0.95
        assert settings.cache_ttl_seconds == 3600
        assert settings.redis_url == "redis://localhost:6379/0"


class TestPipelineBuilderCache:
    """Test PipelineBuilder cache configuration."""

    def test_with_cache_memory(self):
        """Configure in-memory cache."""
        from agentic_rag.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder().with_cache(backend="memory")
        assert builder._cache_config["backend"] == "memory"

    def test_with_cache_redis(self):
        """Configure Redis cache with URL."""
        from agentic_rag.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder().with_cache(
            backend="redis",
            redis_url="redis://localhost:6379/0",
        )
        assert builder._cache_config["backend"] == "redis"
        assert builder._cache_config["redis_url"] == "redis://localhost:6379/0"


# =============================================================================
# Context Compression Tests
# =============================================================================


class TestCompressionResult:
    """Test CompressionResult model."""

    def test_default_result(self):
        """Default result has empty chunks and 1.0 ratio."""
        from agentic_rag.compression.base import CompressionResult

        result = CompressionResult()
        assert result.compressed_chunks == []
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0
        assert result.compression_ratio == 1.0

    def test_tokens_saved_calculation(self):
        """tokens_saved returns difference."""
        from agentic_rag.compression.base import CompressionResult

        result = CompressionResult(
            original_tokens=1000,
            compressed_tokens=300,
        )
        assert result.tokens_saved == 700

    def test_savings_percent_calculation(self):
        """savings_percent based on compression ratio."""
        from agentic_rag.compression.base import CompressionResult

        result = CompressionResult(
            original_tokens=1000,
            compressed_tokens=300,
            compression_ratio=0.3,
        )
        assert result.savings_percent == 70.0


class TestExtractiveCompressor:
    """Test ExtractiveCompressor using reranker."""

    @pytest.fixture
    def mock_reranker(self):
        """Create mock reranker."""
        reranker = AsyncMock()
        return reranker

    def test_initialization(self, mock_reranker):
        """Initialize with reranker and ratio."""
        from agentic_rag.compression.extractive import ExtractiveCompressor

        compressor = ExtractiveCompressor(
            reranker=mock_reranker,
            compression_ratio=0.5,
        )
        assert compressor._reranker == mock_reranker
        assert compressor._compression_ratio == 0.5

    def test_split_sentences_basic(self, mock_reranker):
        """Split text into sentences."""
        from agentic_rag.compression.extractive import ExtractiveCompressor

        compressor = ExtractiveCompressor(reranker=mock_reranker)
        text = "This is the first sentence. This is the second sentence. And here is a third one that is longer."
        sentences = compressor._split_sentences(text)

        assert len(sentences) >= 1
        assert all(len(s) > 20 for s in sentences)

    def test_estimate_tokens(self, mock_reranker):
        """Estimate tokens as chars/4."""
        from agentic_rag.compression.extractive import ExtractiveCompressor

        compressor = ExtractiveCompressor(reranker=mock_reranker)
        assert compressor._estimate_tokens("hello world!") == 3
        assert compressor._estimate_tokens("a" * 100) == 25

    @pytest.mark.asyncio
    async def test_compress_empty_chunks(self, mock_reranker):
        """Compress empty list returns empty result."""
        from agentic_rag.compression.extractive import ExtractiveCompressor

        compressor = ExtractiveCompressor(reranker=mock_reranker)
        result = await compressor.compress(query="test", chunks=[])

        assert result.compressed_chunks == []
        assert result.original_tokens == 0


class TestLongLLMLinguaCompressor:
    """Test LongLLMLinguaCompressor using LLM scoring."""

    @pytest.fixture
    def mock_generator(self):
        """Create mock LLM generator."""
        generator = AsyncMock()
        generator.generate = AsyncMock(return_value=MagicMock(response="8, 7, 6, 5, 4"))
        return generator

    def test_initialization(self, mock_generator):
        """Initialize with generator and ratio."""
        from agentic_rag.compression.longllmlingua import LongLLMLinguaCompressor

        compressor = LongLLMLinguaCompressor(
            generator=mock_generator,
            compression_ratio=0.3,
        )
        assert compressor._generator == mock_generator
        assert compressor._compression_ratio == 0.3

    @pytest.mark.asyncio
    async def test_compress_empty_chunks(self, mock_generator):
        """Compress empty list returns empty result."""
        from agentic_rag.compression.longllmlingua import LongLLMLinguaCompressor

        compressor = LongLLMLinguaCompressor(generator=mock_generator)
        result = await compressor.compress(query="test", chunks=[])

        assert result.compressed_chunks == []


class TestCompressionConfigSettings:
    """Test compression-related configuration."""

    def test_compression_settings_exist(self):
        """Config has compression settings."""

        settings = Settings()
        assert hasattr(settings, "enable_compression")
        assert hasattr(settings, "compression_type")
        assert hasattr(settings, "compression_ratio")

    def test_compression_defaults(self):
        """Compression settings have sensible defaults."""

        settings = Settings()
        assert settings.enable_compression is False
        assert settings.compression_type == "extractive"
        assert settings.compression_ratio == 0.5


class TestPipelineBuilderCompression:
    """Test PipelineBuilder compression configuration."""

    def test_with_compression_extractive(self):
        """Configure extractive compression."""
        from agentic_rag.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder().with_compression(
            method="extractive",
            compression_ratio=0.3,
        )
        assert builder._compression_config["method"] == "extractive"
        assert builder._compression_config["compression_ratio"] == 0.3

    def test_with_compression_longllmlingua(self):
        """Configure LongLLMLingua compression."""
        from agentic_rag.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder().with_compression(
            method="longllmlingua",
            compression_ratio=0.4,
        )
        assert builder._compression_config["method"] == "longllmlingua"


# =============================================================================
# RAPTOR Hierarchical Chunking Tests
# =============================================================================


class TestClusterResult:
    """Test ClusterResult model."""

    def test_default_result(self):
        """Default result is empty."""
        from agentic_rag.chunking.clustering import ClusterResult

        result = ClusterResult()
        assert result.labels == []
        assert result.n_clusters == 0
        assert result.centroids == []
        assert result.metadata == {}

    def test_result_with_data(self):
        """Result stores clustering output."""
        from agentic_rag.chunking.clustering import ClusterResult

        result = ClusterResult(
            labels=[0, 0, 1, 1, 2],
            n_clusters=3,
            centroids=[[0.1], [0.5], [0.9]],
            metadata={"algorithm": "kmeans"},
        )
        assert len(result.labels) == 5
        assert result.n_clusters == 3
        assert len(result.centroids) == 3


class TestKMeansClusterer:
    """Test KMeans clustering algorithm."""

    def test_initialization(self):
        """Initialize with random seed."""
        from agentic_rag.chunking.clustering import KMeansClusterer

        clusterer = KMeansClusterer(random_state=42)
        assert clusterer._random_state == 42

    def test_cluster_basic(self):
        """Cluster small dataset."""
        from agentic_rag.chunking.clustering import KMeansClusterer

        clusterer = KMeansClusterer(random_state=42)
        embeddings = np.random.rand(10, 64)
        result = clusterer.cluster(embeddings, n_clusters=3)

        assert len(result.labels) == 10
        assert result.n_clusters == 3
        assert len(result.centroids) == 3
        assert all(0 <= label < 3 for label in result.labels)

    def test_cluster_single_sample(self):
        """Handle single sample gracefully."""
        from agentic_rag.chunking.clustering import KMeansClusterer

        clusterer = KMeansClusterer()
        embeddings = np.random.rand(1, 64)
        result = clusterer.cluster(embeddings)

        assert len(result.labels) == 1
        assert result.n_clusters == 1


class TestGMMClusterer:
    """Test Gaussian Mixture Model clustering."""

    def test_initialization(self):
        """Initialize with parameters."""
        from agentic_rag.chunking.clustering import GMMClusterer

        clusterer = GMMClusterer(random_state=42, covariance_type="diag")
        assert clusterer._random_state == 42
        assert clusterer._covariance_type == "diag"

    def test_cluster_basic(self):
        """Cluster small dataset."""
        from agentic_rag.chunking.clustering import GMMClusterer

        clusterer = GMMClusterer(random_state=42)
        embeddings = np.random.rand(15, 64)
        result = clusterer.cluster(embeddings, n_clusters=3)

        assert len(result.labels) == 15
        assert result.n_clusters == 3

    def test_cluster_single_sample(self):
        """Handle single sample gracefully."""
        from agentic_rag.chunking.clustering import GMMClusterer

        clusterer = GMMClusterer()
        embeddings = np.random.rand(1, 64)
        result = clusterer.cluster(embeddings)

        assert len(result.labels) == 1
        assert result.n_clusters == 1


class TestCreateClusterer:
    """Test clusterer factory function."""

    def test_create_kmeans(self):
        """Factory creates KMeans clusterer."""
        from agentic_rag.chunking.clustering import KMeansClusterer, create_clusterer

        clusterer = create_clusterer("kmeans")
        assert isinstance(clusterer, KMeansClusterer)

    def test_create_gmm(self):
        """Factory creates GMM clusterer."""
        from agentic_rag.chunking.clustering import GMMClusterer, create_clusterer

        clusterer = create_clusterer("gmm")
        assert isinstance(clusterer, GMMClusterer)

    def test_create_unknown_raises(self):
        """Unknown algorithm raises ValueError."""
        from agentic_rag.chunking.clustering import create_clusterer

        with pytest.raises(ValueError, match="Unknown clustering algorithm"):
            create_clusterer("unknown_algorithm")


class TestRAPTORNode:
    """Test RAPTOR tree node model."""

    def test_create_leaf_node(self):
        """Create leaf node (level 0)."""
        from agentic_rag.chunking.raptor import RAPTORNode

        node = RAPTORNode(
            content="Original document content.",
            level=0,
            is_summary=False,
            document_id="doc1",
        )
        assert node.content == "Original document content."
        assert node.level == 0
        assert node.is_summary is False
        assert node.id is not None

    def test_create_summary_node(self):
        """Create summary node (level > 0)."""
        from agentic_rag.chunking.raptor import RAPTORNode

        node = RAPTORNode(
            content="Summary of clustered content.",
            level=1,
            is_summary=True,
            child_ids=["child1", "child2"],
        )
        assert node.level == 1
        assert node.is_summary is True
        assert len(node.child_ids) == 2

    def test_node_to_chunk(self):
        """Convert node to Chunk for retrieval."""
        from agentic_rag.chunking.raptor import RAPTORNode

        node = RAPTORNode(
            content="Test content",
            level=2,
            is_summary=True,
            document_id="doc1",
            cluster_id=5,
        )
        chunk = node.to_chunk()

        assert chunk.content == "Test content"
        assert chunk.document_id == "doc1"
        assert chunk.metadata["raptor_level"] == 2
        assert chunk.metadata["is_summary"] is True
        assert chunk.metadata["cluster_id"] == 5


class TestRAPTORTree:
    """Test RAPTOR tree structure."""

    def test_empty_tree(self):
        """Empty tree has no nodes."""
        from agentic_rag.chunking.raptor import RAPTORTree

        tree = RAPTORTree()
        assert tree.total_nodes == 0
        assert tree.leaf_count == 0
        assert tree.summary_count == 0

    def test_get_level(self):
        """Get nodes at specific level."""
        from agentic_rag.chunking.raptor import RAPTORNode, RAPTORTree

        tree = RAPTORTree()
        leaf1 = RAPTORNode(content="leaf1", level=0)
        leaf2 = RAPTORNode(content="leaf2", level=0)
        summary = RAPTORNode(content="summary", level=1, is_summary=True)

        tree.nodes[leaf1.id] = leaf1
        tree.nodes[leaf2.id] = leaf2
        tree.nodes[summary.id] = summary

        level0 = tree.get_level(0)
        level1 = tree.get_level(1)

        assert len(level0) == 2
        assert len(level1) == 1

    def test_get_leaves(self):
        """Get all leaf nodes (level 0)."""
        from agentic_rag.chunking.raptor import RAPTORNode, RAPTORTree

        tree = RAPTORTree()
        tree.nodes["leaf"] = RAPTORNode(content="leaf", level=0)
        tree.nodes["summary"] = RAPTORNode(content="summary", level=1, is_summary=True)

        leaves = tree.get_leaves()
        assert len(leaves) == 1
        assert leaves[0].level == 0

    def test_all_chunks(self):
        """Convert all nodes to chunks."""
        from agentic_rag.chunking.raptor import RAPTORNode, RAPTORTree

        tree = RAPTORTree()
        tree.nodes["n1"] = RAPTORNode(content="c1", level=0)
        tree.nodes["n2"] = RAPTORNode(content="c2", level=1)
        tree.nodes["n3"] = RAPTORNode(content="c3", level=2)

        chunks = tree.all_chunks()
        assert len(chunks) == 3
        assert all(isinstance(c, Chunk) for c in chunks)


class TestRAPTORChunker:
    """Test RAPTOR chunker."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder for testing."""
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 768)
        embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
        return embedder

    @pytest.fixture
    def mock_generator(self):
        """Mock LLM generator for summarization."""
        generator = AsyncMock()
        generator.generate = AsyncMock(return_value=MagicMock(response="Summary of the content."))
        return generator

    def test_initialization(self, mock_embedder, mock_generator):
        """Initialize with components."""
        from agentic_rag.chunking.raptor import RAPTORChunker

        chunker = RAPTORChunker(
            embedder=mock_embedder,
            generator=mock_generator,
            max_levels=3,
            clustering_algorithm="gmm",
        )
        assert chunker._max_levels == 3

    def test_initialization_with_base_chunker(self, mock_embedder, mock_generator):
        """Initialize with custom base chunker."""
        from agentic_rag.chunking.raptor import RAPTORChunker

        base_chunker = MagicMock()
        chunker = RAPTORChunker(
            embedder=mock_embedder,
            generator=mock_generator,
            base_chunker=base_chunker,
        )
        assert chunker._base_chunker == base_chunker


class TestRAPTORRetriever:
    """Test RAPTOR-aware retriever."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder."""
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 768)
        return embedder

    @pytest.fixture
    def mock_vectordb(self):
        """Mock vector database."""
        vectordb = AsyncMock()
        vectordb.search = AsyncMock(
            return_value=[
                Chunk(
                    id="1",
                    content="Leaf content",
                    document_id="doc1",
                    metadata={"raptor_level": 0, "is_summary": False, "score": 0.9},
                ),
                Chunk(
                    id="2",
                    content="Summary content",
                    document_id="doc1",
                    metadata={"raptor_level": 1, "is_summary": True, "score": 0.8},
                ),
            ]
        )
        return vectordb

    def test_initialization(self, mock_embedder, mock_vectordb):
        """Initialize retriever."""
        from agentic_rag.retrieval.raptor_retriever import RAPTORRetriever

        retriever = RAPTORRetriever(
            embedder=mock_embedder,
            vectordb=mock_vectordb,
        )
        assert retriever._embedder == mock_embedder
        assert retriever._vectordb == mock_vectordb

    @pytest.mark.asyncio
    async def test_collapsed_retrieval(self, mock_embedder, mock_vectordb):
        """Collapsed mode searches all levels."""
        from agentic_rag.retrieval.raptor_retriever import RAPTORRetriever

        retriever = RAPTORRetriever(
            embedder=mock_embedder,
            vectordb=mock_vectordb,
        )

        result = await retriever.retrieve(
            query="test query",
            collection="test",
            top_k=5,
            mode="collapsed",
        )

        assert len(result.chunks) > 0
        assert result.metadata["mode"] == "collapsed"

    @pytest.mark.asyncio
    async def test_tree_traversal_retrieval(self, mock_embedder, mock_vectordb):
        """Tree traversal combines summaries and leaves."""
        from agentic_rag.retrieval.raptor_retriever import RAPTORRetriever

        retriever = RAPTORRetriever(
            embedder=mock_embedder,
            vectordb=mock_vectordb,
        )

        result = await retriever.retrieve(
            query="test query",
            collection="test",
            top_k=5,
            mode="tree_traversal",
        )

        assert result.metadata["mode"] == "tree_traversal"
        assert "summaries_selected" in result.metadata
        assert "leaves_selected" in result.metadata


class TestRAPTORConfigSettings:
    """Test RAPTOR-related configuration."""

    def test_raptor_settings_exist(self):
        """Config has RAPTOR settings."""

        settings = Settings()
        assert hasattr(settings, "raptor_max_levels")
        assert hasattr(settings, "raptor_clustering")
        assert hasattr(settings, "raptor_min_cluster_size")
        assert hasattr(settings, "raptor_summary_tokens")

    def test_raptor_defaults(self):
        """RAPTOR settings have sensible defaults."""

        settings = Settings()
        assert settings.raptor_max_levels == 3
        assert settings.raptor_clustering == "gmm"
        assert settings.raptor_min_cluster_size == 2
        assert settings.raptor_summary_tokens == 200


class TestPipelineBuilderRAPTOR:
    """Test PipelineBuilder RAPTOR configuration."""

    def test_with_chunking_raptor(self):
        """Configure RAPTOR chunking strategy."""
        from agentic_rag.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder().with_chunking(
            strategy="raptor",
            raptor_levels=3,
            raptor_clustering="gmm",
        )
        assert builder._config.chunk_strategy == "raptor"
        assert builder._raptor_config["max_levels"] == 3
        assert builder._raptor_config["clustering_algorithm"] == "gmm"

    def test_with_chunking_raptor_kmeans(self):
        """Configure RAPTOR with KMeans clustering."""
        from agentic_rag.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder().with_chunking(
            strategy="raptor",
            raptor_levels=4,
            raptor_clustering="kmeans",
        )
        assert builder._raptor_config["clustering_algorithm"] == "kmeans"
        assert builder._raptor_config["max_levels"] == 4

    def test_raptor_with_other_features(self):
        """RAPTOR integrates with other pipeline features."""
        from agentic_rag.pipeline.builder import PipelineBuilder

        builder = (
            PipelineBuilder()
            .with_chunking("raptor", raptor_levels=2)
            .with_retrieval("hybrid", top_k=10)
            .with_compression("extractive", compression_ratio=0.5)
        )
        assert builder._config.chunk_strategy == "raptor"
        assert builder._config.top_k == 10
        assert builder._compression_config is not None
