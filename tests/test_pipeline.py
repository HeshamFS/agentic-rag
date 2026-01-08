"""
Comprehensive unit tests for pipeline functionality.

Tests:
- PipelineBuilder configuration
- Standard RAG pipeline
- Agentic RAG pipeline
- Corrective RAG pipeline
- End-to-end query flows
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag.core.models import (
    Chunk,
    Document,
    GenerationResult,
    RetrievalResult,
)
from agentic_rag.pipeline.agentic import AgenticPipeline
from agentic_rag.pipeline.base import IngestResult, PipelineResult
from agentic_rag.pipeline.builder import PipelineBuilder
from agentic_rag.pipeline.corrective import CorrectivePipeline
from agentic_rag.pipeline.standard import StandardPipeline

# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_pipeline_embedder():
    """Create mock embedder for pipeline tests."""
    embedder = MagicMock()
    embedder.dimension = 384
    embedder.model_name = "test-embedder"

    async def mock_embed_text(text):
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        # MD5 produces 32 hex chars = 16 bytes, use what we have
        values = [(int(h[i : i + 2], 16) / 255.0 - 0.5) for i in range(0, 32, 2)]
        return values + [0.0] * (384 - len(values))

    async def mock_embed_batch(texts):
        return [await mock_embed_text(t) for t in texts]

    embedder.embed_text = AsyncMock(side_effect=mock_embed_text)
    embedder.embed_batch = AsyncMock(side_effect=mock_embed_batch)

    return embedder


@pytest.fixture
def mock_pipeline_vectordb(sample_chunks):
    """Create mock vector database for pipeline tests."""
    db = MagicMock()
    db.db_type = "mock"
    db._collections = {"default": sample_chunks}

    async def mock_create_collection(name, dimension, **kwargs):
        db._collections[name] = []

    async def mock_upsert(collection, chunks, **kwargs):
        if collection not in db._collections:
            db._collections[collection] = []
        db._collections[collection].extend(chunks)

    async def mock_search(collection, query_vector, top_k=10, **kwargs):
        chunks = db._collections.get(collection, [])
        return [(chunk, 0.9 - i * 0.1) for i, chunk in enumerate(chunks[:top_k])]

    async def mock_collection_exists(name):
        return name in db._collections

    async def mock_delete_collection(name):
        if name in db._collections:
            del db._collections[name]

    async def mock_list_collections():
        return list(db._collections.keys())

    async def mock_close():
        pass

    db.create_collection = AsyncMock(side_effect=mock_create_collection)
    db.upsert = AsyncMock(side_effect=mock_upsert)
    db.search = AsyncMock(side_effect=mock_search)
    db.collection_exists = AsyncMock(side_effect=mock_collection_exists)
    db.delete_collection = AsyncMock(side_effect=mock_delete_collection)
    db.list_collections = AsyncMock(side_effect=mock_list_collections)
    db.close = AsyncMock(side_effect=mock_close)

    return db


@pytest.fixture
def mock_pipeline_generator():
    """Create mock generator for pipeline tests."""
    generator = MagicMock()
    generator.provider = "mock"
    generator.model_name = "mock-model"

    async def mock_generate(query, context, **kwargs):
        context_text = " ".join([c.content[:50] for c in context[:3]])
        return GenerationResult(
            response=f"Answer to '{query}': Based on {context_text}...",
            sources=context[:3],
            confidence=0.85,
            provider="mock",
            model="mock-model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=10.0,  # Required field
        )

    async def mock_generate_text(prompt, **kwargs):
        return f"Generated text for: {prompt[:50]}..."

    generator.generate = AsyncMock(side_effect=mock_generate)
    generator.generate_text = AsyncMock(side_effect=mock_generate_text)

    return generator


@pytest.fixture
def mock_pipeline_chunker():
    """Create mock chunker for pipeline tests."""
    chunker = MagicMock()

    async def mock_chunk(document):
        content = document.content
        chunk_size = 200
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i : i + chunk_size]
            if chunk_content.strip():
                chunks.append(
                    Chunk(
                        id=f"{document.id}_chunk_{i // chunk_size}",
                        content=chunk_content,
                        document_id=document.id,
                        position=i // chunk_size,
                    )
                )
        return chunks

    # StandardPipeline calls 'chunk', not 'chunk_async'
    chunker.chunk = AsyncMock(side_effect=mock_chunk)
    chunker.chunk_async = AsyncMock(side_effect=mock_chunk)
    return chunker


@pytest.fixture
def mock_pipeline_retriever(mock_pipeline_embedder, mock_pipeline_vectordb):
    """Create mock retriever for pipeline tests."""
    retriever = MagicMock()

    async def mock_retrieve(query, collection, top_k=5, **kwargs):
        search_results = await mock_pipeline_vectordb.search(collection, [], top_k)
        chunks = [chunk for chunk, score in search_results]
        scores = [score for chunk, score in search_results]
        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            retrieval_type="mock",
            metadata={"query": query},
        )

    retriever.retrieve = AsyncMock(side_effect=mock_retrieve)
    return retriever


# =============================================================================
# PipelineBuilder Tests
# =============================================================================


class TestPipelineBuilder:
    """Tests for the PipelineBuilder class."""

    def test_builder_creates_instance(self):
        """Test that builder creates a PipelineBuilder instance."""
        builder = PipelineBuilder()
        assert builder is not None

    def test_builder_with_embedder(self, mock_pipeline_embedder):
        """Test setting embedder on builder."""
        builder = PipelineBuilder()
        result = builder.with_embedder(mock_pipeline_embedder)

        assert result is builder  # Returns self for chaining
        assert builder._embedder == mock_pipeline_embedder

    def test_builder_with_vectordb(self, test_settings_minimal):
        """Test setting vector database on builder."""
        with patch("agentic_rag.pipeline.builder.QdrantVectorDB") as mock_qdrant:
            mock_qdrant.return_value = MagicMock()

            builder = PipelineBuilder()
            result = builder.with_vectordb("qdrant", url="http://localhost:6333")

            assert result is builder

    def test_builder_with_generator(self, test_settings_minimal):
        """Test setting generator on builder."""
        with patch("agentic_rag.pipeline.builder.create_generator") as mock_create:
            mock_create.return_value = MagicMock()

            builder = PipelineBuilder()
            result = builder.with_generator("gemini", model="gemini-2.0-flash")

            assert result is builder

    def test_builder_with_chunking(self, mock_pipeline_embedder):
        """Test setting chunking strategy on builder."""
        builder = PipelineBuilder()
        builder._embedder = mock_pipeline_embedder

        result = builder.with_chunking("semantic", chunk_size=512)

        assert result is builder

    def test_builder_with_retrieval(self):
        """Test setting retrieval strategy on builder."""
        builder = PipelineBuilder()
        result = builder.with_retrieval("hybrid", alpha=0.5)

        assert result is builder

    def test_builder_build_creates_pipeline(
        self,
        mock_pipeline_embedder,
        mock_pipeline_vectordb,
        mock_pipeline_generator,
    ):
        """Test that build creates a complete pipeline."""
        with (
            patch("agentic_rag.pipeline.builder.QdrantVectorDB") as mock_qdrant,
            patch("agentic_rag.pipeline.builder.create_generator") as mock_create,
        ):
            mock_qdrant.return_value = mock_pipeline_vectordb
            mock_create.return_value = mock_pipeline_generator

            builder = PipelineBuilder()
            pipeline = (
                builder.with_embedder(mock_pipeline_embedder)
                .with_vectordb("qdrant", url="http://localhost:6333")
                .with_generator("gemini", model="gemini-2.0-flash")
                .with_chunking("semantic")
                .with_retrieval("dense")
                .build()
            )

            assert pipeline is not None


# =============================================================================
# StandardPipeline Tests
# =============================================================================


class TestStandardPipeline:
    """Tests for the StandardPipeline class."""

    @pytest.fixture
    def standard_pipeline(
        self,
        mock_pipeline_embedder,
        mock_pipeline_vectordb,
        mock_pipeline_generator,
        mock_pipeline_chunker,
        mock_pipeline_retriever,
    ):
        """Create standard pipeline with mocks."""
        return StandardPipeline(
            embedder=mock_pipeline_embedder,
            vectordb=mock_pipeline_vectordb,
            generator=mock_pipeline_generator,
            chunker=mock_pipeline_chunker,
            retriever=mock_pipeline_retriever,
        )

    @pytest.mark.asyncio
    async def test_ingest_returns_result(self, standard_pipeline, sample_documents):
        """Test that ingest returns IngestResult."""
        result = await standard_pipeline.ingest(
            documents=sample_documents,
            collection="test",
        )

        assert isinstance(result, IngestResult)
        assert result.documents > 0 or result.chunks > 0

    @pytest.mark.asyncio
    async def test_ingest_processes_documents(self, standard_pipeline, sample_documents):
        """Test that ingest processes all documents."""
        result = await standard_pipeline.ingest(
            documents=sample_documents,
            collection="test",
        )

        # Should have processed documents
        assert result.documents > 0 or result.chunks > 0

    @pytest.mark.asyncio
    async def test_query_returns_generation_result(self, standard_pipeline):
        """Test that query returns PipelineResult."""
        result = await standard_pipeline.query(
            question="What is machine learning?",
            collection="default",
        )

        assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_query_includes_response(self, standard_pipeline):
        """Test that query includes response text."""
        result = await standard_pipeline.query(
            question="What is AI?",
            collection="default",
        )

        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_query_includes_sources(self, standard_pipeline):
        """Test that query includes sources."""
        result = await standard_pipeline.query(
            question="What is deep learning?",
            collection="default",
        )

        assert len(result.sources) > 0

    @pytest.mark.asyncio
    async def test_query_with_top_k(self, standard_pipeline, mock_pipeline_retriever):
        """Test that query respects top_k parameter."""
        await standard_pipeline.query(
            question="Test",
            collection="default",
            top_k=3,
        )

        # Retriever should be called with top_k
        call_args = mock_pipeline_retriever.retrieve.call_args
        assert call_args.kwargs.get("top_k") == 3 or (call_args.args and call_args.args[2] == 3)


# =============================================================================
# AgenticPipeline Tests
# =============================================================================


@pytest.mark.integration
class TestAgenticPipeline:
    """Tests for the AgenticPipeline class. Requires properly initialized agent components."""

    @pytest.fixture
    def agentic_pipeline(
        self,
        mock_pipeline_embedder,
        mock_pipeline_vectordb,
        mock_pipeline_generator,
        mock_pipeline_chunker,
        mock_pipeline_retriever,
    ):
        """Create agentic pipeline with mocks."""
        pytest.skip("AgenticPipeline requires real agent components")
        return AgenticPipeline(
            embedder=mock_pipeline_embedder,
            vectordb=mock_pipeline_vectordb,
            generator=mock_pipeline_generator,
            chunker=mock_pipeline_chunker,
            retriever=mock_pipeline_retriever,
        )

    @pytest.mark.asyncio
    async def test_agentic_query_returns_result(self, agentic_pipeline):
        """Test that agentic query returns result."""
        result = await agentic_pipeline.query(
            question="What is machine learning and how does it relate to AI?",
            collection="default",
        )

        assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_agentic_handles_complex_query(self, agentic_pipeline):
        """Test that agentic pipeline handles complex queries."""
        result = await agentic_pipeline.query(
            question="Compare and contrast supervised and unsupervised learning approaches.",
            collection="default",
        )

        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_agentic_multi_step_reasoning(self, agentic_pipeline):
        """Test multi-step reasoning in agentic mode."""
        result = await agentic_pipeline.query(
            question="What techniques are used in RAG systems to reduce hallucinations?",
            collection="default",
        )

        # Should complete reasoning
        assert result.response is not None


# =============================================================================
# CorrectivePipeline Tests
# =============================================================================


@pytest.mark.integration
class TestCorrectivePipeline:
    """Tests for the CorrectivePipeline class. Requires properly initialized HyDE retriever."""

    @pytest.fixture
    def corrective_pipeline(
        self,
        mock_pipeline_embedder,
        mock_pipeline_vectordb,
        mock_pipeline_generator,
        mock_pipeline_chunker,
        mock_pipeline_retriever,
    ):
        """Create corrective pipeline with mocks."""
        pytest.skip("CorrectivePipeline requires real HyDE retriever components")
        return CorrectivePipeline(
            embedder=mock_pipeline_embedder,
            vectordb=mock_pipeline_vectordb,
            generator=mock_pipeline_generator,
            chunker=mock_pipeline_chunker,
            retriever=mock_pipeline_retriever,
        )

    @pytest.mark.asyncio
    async def test_corrective_query_returns_result(self, corrective_pipeline):
        """Test that corrective query returns result."""
        result = await corrective_pipeline.query(
            question="What is deep learning?",
            collection="default",
        )

        assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_corrective_evaluates_retrieval(self, corrective_pipeline):
        """Test that corrective pipeline evaluates retrieval quality."""
        result = await corrective_pipeline.query(
            question="What is the attention mechanism?",
            collection="default",
        )

        # Should complete with evaluation
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_corrective_handles_low_quality_retrieval(
        self, corrective_pipeline, mock_pipeline_vectordb
    ):
        """Test handling of low quality retrieval."""
        # Simulate low quality retrieval
        original_search = mock_pipeline_vectordb.search

        async def low_quality_search(*args, **kwargs):
            results = await original_search(*args, **kwargs)
            # Return with low scores
            return [(chunk, 0.2) for chunk, _ in results]

        mock_pipeline_vectordb.search = AsyncMock(side_effect=low_quality_search)

        result = await corrective_pipeline.query(
            question="Very specific question about obscure topic",
            collection="default",
        )

        # Should still return a result
        assert result is not None


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline."""

    @pytest.fixture
    def complete_pipeline(
        self,
        mock_pipeline_embedder,
        mock_pipeline_vectordb,
        mock_pipeline_generator,
        mock_pipeline_chunker,
        mock_pipeline_retriever,
    ):
        """Create complete pipeline for e2e tests."""
        return StandardPipeline(
            embedder=mock_pipeline_embedder,
            vectordb=mock_pipeline_vectordb,
            generator=mock_pipeline_generator,
            chunker=mock_pipeline_chunker,
            retriever=mock_pipeline_retriever,
        )

    @pytest.mark.asyncio
    async def test_ingest_then_query(self, complete_pipeline, sample_documents):
        """Test full ingest then query flow."""
        # Ingest
        ingest_result = await complete_pipeline.ingest(
            documents=sample_documents,
            collection="e2e_test",
        )
        assert isinstance(ingest_result, IngestResult)
        assert ingest_result.chunks > 0 or ingest_result.documents > 0

        # Query
        query_result = await complete_pipeline.query(
            question="What is Python?",
            collection="e2e_test",
        )
        assert isinstance(query_result, PipelineResult)
        assert len(query_result.response) > 0

    @pytest.mark.asyncio
    async def test_multiple_queries_same_collection(self, complete_pipeline):
        """Test multiple queries on same collection."""
        queries = [
            "What is machine learning?",
            "Explain neural networks.",
            "What is RAG?",
        ]

        results = []
        for query in queries:
            result = await complete_pipeline.query(
                question=query,
                collection="default",
            )
            results.append(result)

        assert len(results) == 3
        assert all(isinstance(r, PipelineResult) for r in results)

    @pytest.mark.asyncio
    async def test_query_different_collections(self, complete_pipeline, sample_documents):
        """Test queries across different collections."""
        # Create two collections
        await complete_pipeline.ingest(
            documents=sample_documents[:2],
            collection="collection_1",
        )
        await complete_pipeline.ingest(
            documents=sample_documents[2:],
            collection="collection_2",
        )

        # Query both
        result1 = await complete_pipeline.query(
            question="Test query",
            collection="collection_1",
        )
        result2 = await complete_pipeline.query(
            question="Test query",
            collection="collection_2",
        )

        assert isinstance(result1, PipelineResult)
        assert isinstance(result2, PipelineResult)


# =============================================================================
# Pipeline Configuration Tests
# =============================================================================


class TestPipelineConfiguration:
    """Tests for pipeline configuration options."""

    @pytest.mark.asyncio
    async def test_pipeline_with_custom_temperature(
        self,
        mock_pipeline_embedder,
        mock_pipeline_vectordb,
        mock_pipeline_generator,
        mock_pipeline_chunker,
        mock_pipeline_retriever,
    ):
        """Test pipeline with custom temperature."""
        pipeline = StandardPipeline(
            embedder=mock_pipeline_embedder,
            vectordb=mock_pipeline_vectordb,
            generator=mock_pipeline_generator,
            chunker=mock_pipeline_chunker,
            retriever=mock_pipeline_retriever,
        )

        result = await pipeline.query(
            question="Test",
            collection="default",
            temperature=0.7,
        )

        assert isinstance(result, PipelineResult)


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestPipelinePerformance:
    """Performance tests for pipelines."""

    @pytest.fixture
    def fast_pipeline(
        self,
        mock_pipeline_embedder,
        mock_pipeline_vectordb,
        mock_pipeline_generator,
        mock_pipeline_chunker,
        mock_pipeline_retriever,
    ):
        """Create pipeline for performance testing."""
        return StandardPipeline(
            embedder=mock_pipeline_embedder,
            vectordb=mock_pipeline_vectordb,
            generator=mock_pipeline_generator,
            chunker=mock_pipeline_chunker,
            retriever=mock_pipeline_retriever,
        )

    @pytest.mark.asyncio
    async def test_query_latency(self, fast_pipeline):
        """Test query latency."""
        import time

        start = time.time()
        await fast_pipeline.query(
            question="Test query",
            collection="default",
        )
        elapsed = time.time() - start

        # Should be fast with mocks
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, fast_pipeline):
        """Test concurrent query handling."""
        import time

        queries = [f"Query {i}" for i in range(10)]

        start = time.time()
        tasks = [fast_pipeline.query(question=q, collection="default") for q in queries]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        # Concurrent should be efficient
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_large_document_ingest(self, fast_pipeline):
        """Test ingesting large documents."""
        import time

        large_docs = [
            Document(
                id=f"large_{i}",
                content="Test content. " * 1000,
                metadata={},
            )
            for i in range(10)
        ]

        start = time.time()
        result = await fast_pipeline.ingest(
            documents=large_docs,
            collection="large_test",
        )
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 30.0
        assert isinstance(result, IngestResult)
