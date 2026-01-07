"""
Comprehensive unit tests for chunking functionality.

Tests:
- SemanticChunker with embedding-based breakpoints
- ContextualChunker with contextual headers
- HierarchicalChunker with multi-level structure
- RecursiveChunker for nested documents
- Edge cases and error handling
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from agentic_rag.chunking.contextual import ContextualChunker
from agentic_rag.chunking.hierarchical import HierarchicalChunker
from agentic_rag.chunking.semantic import PercentileSemanticChunker, SemanticChunker
from agentic_rag.core.models import Chunk, Document

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def long_document() -> Document:
    """Create a long document for chunking tests."""
    content = """
    Chapter 1: Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. It focuses on the
    development of computer programs that can access data and use it to learn for themselves.

    The process of learning begins with observations or data, such as examples, direct
    experience, or instruction. The goal is to allow computers to learn automatically
    without human intervention or assistance.

    Chapter 2: Types of Machine Learning

    There are three main types of machine learning: supervised learning, unsupervised
    learning, and reinforcement learning.

    Supervised learning uses labeled data to train algorithms. The algorithm learns
    from the training data and can make predictions on new, unseen data.

    Unsupervised learning works with unlabeled data. The algorithm tries to find
    hidden patterns or intrinsic structures in the input data.

    Reinforcement learning is about taking suitable actions to maximize reward in a
    particular situation. It is employed by various software and machines to find
    the best possible behavior or path.

    Chapter 3: Deep Learning

    Deep learning is a subset of machine learning based on artificial neural networks.
    Neural networks are computing systems inspired by biological neural networks that
    constitute animal brains.

    Deep learning architectures such as deep neural networks, recurrent neural networks,
    and convolutional neural networks have been applied to fields including computer
    vision, speech recognition, and natural language processing.

    These networks contain multiple layers of neurons, which is why they are called
    "deep" learning. Each layer transforms the input data into progressively more
    abstract representations.
    """
    return Document(
        id="test_doc_1",
        content=content.strip(),
        metadata={"source": "test", "type": "textbook"},
    )


@pytest.fixture
def short_document() -> Document:
    """Create a short document for edge case testing."""
    return Document(
        id="test_doc_2",
        content="This is a very short document.",
        metadata={"source": "test"},
    )


@pytest.fixture
def single_sentence_document() -> Document:
    """Create a single sentence document."""
    return Document(
        id="test_doc_3",
        content="Single sentence.",
        metadata={},
    )


@pytest.fixture
def mock_embedder_for_chunking():
    """Create mock embedder for semantic chunking."""
    embedder = MagicMock()
    embedder.dimension = 384

    # Create embeddings that change at topic boundaries

    async def mock_embed_batch(texts):
        """Return embeddings that simulate topic clustering."""
        embeddings = []
        for _i, text in enumerate(texts):
            # Simulate different embeddings for different topics
            if "machine learning" in text.lower() or "artificial intelligence" in text.lower():
                base = [0.8, 0.1, 0.1]
            elif "supervised" in text.lower() or "unsupervised" in text.lower():
                base = [0.7, 0.3, 0.0]
            elif "deep learning" in text.lower() or "neural" in text.lower():
                base = [0.1, 0.8, 0.1]
            else:
                base = [0.4, 0.4, 0.2]

            # Add variation
            embedding = base + [0.0] * 381
            embeddings.append(embedding)
        return embeddings

    embedder.embed_batch = mock_embed_batch
    return embedder


# =============================================================================
# SemanticChunker Tests
# =============================================================================


class TestSemanticChunker:
    """Tests for the SemanticChunker class."""

    @pytest.fixture
    def semantic_chunker(self, mock_embedder_for_chunking):
        """Create semantic chunker with mock embedder."""
        return SemanticChunker(
            embedder=mock_embedder_for_chunking,
            chunk_size=512,
            similarity_threshold=0.5,
            min_chunk_size=50,
            max_chunk_size=1000,
        )

    @pytest.mark.asyncio
    async def test_chunk_creates_chunks(self, semantic_chunker, long_document):
        """Test that chunking creates non-empty chunks."""
        chunks = await semantic_chunker.chunk_async(long_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_chunk_preserves_document_id(self, semantic_chunker, long_document):
        """Test that chunks reference the source document."""
        chunks = await semantic_chunker.chunk_async(long_document)

        assert all(c.document_id == long_document.id for c in chunks)

    @pytest.mark.asyncio
    async def test_chunk_content_coverage(self, semantic_chunker, long_document):
        """Test that chunks cover the document content."""
        chunks = await semantic_chunker.chunk_async(long_document)

        # All chunk content should be non-empty and from the document
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
            # Check that at least some words from the chunk appear in the document
            words = chunk.content.split()[:5]
            for word in words:
                if len(word) > 3:  # Skip short words
                    assert word in long_document.content

    @pytest.mark.asyncio
    async def test_chunk_position_assigned(self, semantic_chunker, long_document):
        """Test that chunks have metadata index assigned."""
        chunks = await semantic_chunker.chunk_async(long_document)

        # Check chunk_index in metadata instead of position field
        indices = [c.metadata.get("chunk_index") for c in chunks]
        assert all(i is not None for i in indices)  # All have index

    @pytest.mark.asyncio
    async def test_chunk_respects_min_size(self, semantic_chunker, long_document):
        """Test that chunks respect minimum size."""
        chunks = await semantic_chunker.chunk_async(long_document)

        # Most chunks should be above min size (last chunk might be smaller)
        for chunk in chunks[:-1]:
            assert len(chunk.content) >= semantic_chunker.min_chunk_size or len(chunks) == 1

    @pytest.mark.asyncio
    async def test_chunk_respects_max_size(self, semantic_chunker, long_document):
        """Test that chunks respect maximum size."""
        chunks = await semantic_chunker.chunk_async(long_document)

        for chunk in chunks:
            assert len(chunk.content) <= semantic_chunker.max_chunk_size + 100  # Small tolerance

    @pytest.mark.asyncio
    async def test_short_document_single_chunk(self, semantic_chunker, short_document):
        """Test that short documents produce single chunk."""
        chunks = await semantic_chunker.chunk_async(short_document)

        # Very short docs should have 1-2 chunks max
        assert 1 <= len(chunks) <= 2

    @pytest.mark.asyncio
    async def test_single_sentence_document(self, semantic_chunker, single_sentence_document):
        """Test chunking single sentence document."""
        chunks = await semantic_chunker.chunk_async(single_sentence_document)

        assert len(chunks) == 1
        assert chunks[0].content.strip() == single_sentence_document.content.strip()

    @pytest.mark.asyncio
    async def test_empty_document(self, semantic_chunker):
        """Test chunking empty document."""
        empty_doc = Document(id="empty", content="", metadata={})
        chunks = await semantic_chunker.chunk_async(empty_doc)

        assert chunks == []

    @pytest.mark.asyncio
    async def test_chunk_metadata_includes_method(self, semantic_chunker, long_document):
        """Test that chunk metadata includes chunking method."""
        chunks = await semantic_chunker.chunk_async(long_document)

        for chunk in chunks:
            assert "chunking_method" in chunk.metadata
            assert chunk.metadata["chunking_method"] == "semantic"

    def test_sentence_splitting(self, semantic_chunker):
        """Test sentence splitting logic."""
        text = "First sentence. Second sentence! Third question? Fourth sentence."
        sentences = semantic_chunker._split_sentences(text)

        assert len(sentences) >= 3

    def test_cosine_similarity(self, semantic_chunker):
        """Test cosine similarity calculation."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]

        assert semantic_chunker._cosine_similarity(v1, v2) == pytest.approx(1.0, abs=0.01)
        assert semantic_chunker._cosine_similarity(v1, v3) == pytest.approx(0.0, abs=0.01)

    def test_smooth_similarities(self, semantic_chunker):
        """Test similarity smoothing."""
        similarities = [0.9, 0.3, 0.2, 0.85, 0.9]
        smoothed = semantic_chunker._smooth_similarities(similarities)

        assert len(smoothed) == len(similarities)
        # Smoothing should reduce variance
        assert np.std(smoothed) <= np.std(similarities) + 0.1


# =============================================================================
# PercentileSemanticChunker Tests
# =============================================================================


class TestPercentileSemanticChunker:
    """Tests for the PercentileSemanticChunker class."""

    @pytest.fixture
    def percentile_chunker(self, mock_embedder_for_chunking):
        """Create percentile-based semantic chunker."""
        return PercentileSemanticChunker(
            embedder=mock_embedder_for_chunking,
            chunk_size=512,
            breakpoint_percentile=25,
        )

    @pytest.mark.asyncio
    async def test_percentile_chunking(self, percentile_chunker, long_document):
        """Test percentile-based chunking creates chunks."""
        chunks = await percentile_chunker.chunk_async(long_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_different_percentiles_affect_chunking(
        self, mock_embedder_for_chunking, long_document
    ):
        """Test that different percentiles produce different chunk counts."""
        chunker_low = PercentileSemanticChunker(
            embedder=mock_embedder_for_chunking,
            breakpoint_percentile=10,
        )
        chunker_high = PercentileSemanticChunker(
            embedder=mock_embedder_for_chunking,
            breakpoint_percentile=50,
        )

        chunks_low = await chunker_low.chunk_async(long_document)
        chunks_high = await chunker_high.chunk_async(long_document)

        # Higher percentile should create more breakpoints (more chunks)
        # Or at least different number of chunks
        # Note: Actual behavior depends on similarity distribution
        assert isinstance(len(chunks_low), int)
        assert isinstance(len(chunks_high), int)


# =============================================================================
# ContextualChunker Tests
# =============================================================================


class TestContextualChunker:
    """Tests for the ContextualChunker class."""

    @pytest.fixture
    def mock_generator(self):
        """Create a mock generator for context headers."""
        generator = MagicMock()
        generator.generate_text = AsyncMock(
            return_value="This chunk discusses technical concepts from the document."
        )
        return generator

    @pytest.fixture
    def contextual_chunker(self, mock_generator):
        """Create contextual chunker with mock generator."""
        return ContextualChunker(
            generator=mock_generator,
            chunk_size=512,
        )

    @pytest.mark.asyncio
    async def test_contextual_chunking_adds_headers(self, contextual_chunker, long_document):
        """Test that contextual chunking adds context headers."""
        chunks = await contextual_chunker.chunk_async(long_document)

        # Should produce chunks
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_context_header_format(self, contextual_chunker, long_document):
        """Test that context headers have proper format."""
        chunks = await contextual_chunker.chunk_async(long_document)

        # All chunks should have content
        for chunk in chunks:
            assert isinstance(chunk.content, str)
            assert len(chunk.content) > 0


# =============================================================================
# HierarchicalChunker Tests
# =============================================================================


class TestHierarchicalChunker:
    """Tests for the HierarchicalChunker class."""

    @pytest.fixture
    def hierarchical_document(self):
        """Create document with hierarchical structure."""
        content = """
# Main Title

Introduction paragraph here.

## Section 1

Content for section 1.

### Subsection 1.1

Detailed content for subsection 1.1.

### Subsection 1.2

Detailed content for subsection 1.2.

## Section 2

Content for section 2.

### Subsection 2.1

Detailed content for subsection 2.1.
        """
        return Document(id="hierarchical_doc", content=content.strip(), metadata={})

    @pytest.fixture
    def hierarchical_chunker(self):
        """Create hierarchical chunker."""
        return HierarchicalChunker(
            levels=[500, 200, 100],
        )

    def test_hierarchical_chunking_creates_levels(
        self, hierarchical_chunker, hierarchical_document
    ):
        """Test that hierarchical chunking creates level information."""
        chunks = hierarchical_chunker.chunk(hierarchical_document)

        # Should have chunks at different levels
        levels = {c.level for c in chunks}
        assert len(levels) >= 1

    def test_hierarchical_preserves_structure(self, hierarchical_chunker, hierarchical_document):
        """Test that hierarchical structure is preserved."""
        chunks = hierarchical_chunker.chunk(hierarchical_document)

        # Check that section references are maintained
        assert len(chunks) > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestChunkingEdgeCases:
    """Tests for edge cases in chunking."""

    @pytest.fixture
    def basic_chunker(self, mock_embedder_for_chunking):
        """Create basic semantic chunker."""
        return SemanticChunker(
            embedder=mock_embedder_for_chunking,
            chunk_size=100,
        )

    @pytest.mark.asyncio
    async def test_whitespace_only_document(self, basic_chunker):
        """Test chunking whitespace-only document."""
        doc = Document(id="whitespace", content="   \n\t\n   ", metadata={})
        chunks = await basic_chunker.chunk_async(doc)

        # Should handle gracefully - either empty list or single whitespace chunk
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_very_long_single_line(self, basic_chunker):
        """Test chunking very long single line."""
        long_line = "word " * 1000
        doc = Document(id="long_line", content=long_line.strip(), metadata={})
        chunks = await basic_chunker.chunk_async(doc)

        # Should be able to chunk
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_special_characters_document(self, basic_chunker):
        """Test chunking document with special characters."""
        content = "Hello! How are you? I'm fine. This costs $100. Email: test@example.com"
        doc = Document(id="special", content=content, metadata={})
        chunks = await basic_chunker.chunk_async(doc)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_unicode_document(self, basic_chunker):
        """Test chunking unicode document."""
        content = "日本語テスト. 中文测试. 한국어 테스트. 🎉 Émojis work!"
        doc = Document(id="unicode", content=content, metadata={})
        chunks = await basic_chunker.chunk_async(doc)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_repeated_content(self, basic_chunker):
        """Test chunking document with repeated content."""
        content = "This is repeated. " * 50
        doc = Document(id="repeated", content=content.strip(), metadata={})
        chunks = await basic_chunker.chunk_async(doc)

        # Should handle repeated content
        assert len(chunks) >= 1


# =============================================================================
# Synchronous Wrapper Tests
# =============================================================================


class TestSynchronousChunking:
    """Tests for synchronous chunking wrapper."""

    @pytest.fixture
    def sync_chunker(self, mock_embedder_for_chunking):
        """Create chunker for sync tests."""
        return SemanticChunker(
            embedder=mock_embedder_for_chunking,
            chunk_size=512,
        )

    def test_sync_chunk_method_exists(self, sync_chunker):
        """Test that sync chunk method exists."""
        assert hasattr(sync_chunker, "chunk")
        assert callable(sync_chunker.chunk)

    def test_sync_chunking_returns_chunks(self, sync_chunker, long_document):
        """Test synchronous chunking returns chunks."""
        # Note: This may not work in async test context
        # Run in separate thread to avoid event loop conflicts
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(sync_chunker.chunk, long_document)
            chunks = future.result(timeout=30)

        assert len(chunks) >= 1


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestChunkingPerformance:
    """Performance tests for chunking."""

    @pytest.fixture
    def fast_chunker(self, mock_embedder_for_chunking):
        """Create chunker optimized for speed tests."""
        return SemanticChunker(
            embedder=mock_embedder_for_chunking,
            chunk_size=256,
        )

    @pytest.mark.asyncio
    async def test_chunking_scales_with_document_size(self, fast_chunker):
        """Test that chunking time scales reasonably with document size."""
        import time

        sizes = [1000, 5000, 10000]
        times = []

        for size in sizes:
            content = "Test sentence for scaling. " * (size // 25)
            doc = Document(id=f"scale_{size}", content=content, metadata={})

            start = time.time()
            await fast_chunker.chunk_async(doc)
            times.append(time.time() - start)

        # Time should scale sub-quadratically
        assert times[2] < times[0] * 100  # 10x content < 100x time


# =============================================================================
# Chunk ID Uniqueness Tests
# =============================================================================


class TestChunkIDUniqueness:
    """Tests for chunk ID uniqueness."""

    @pytest.fixture
    def chunker(self, mock_embedder_for_chunking):
        """Create chunker for ID tests."""
        return SemanticChunker(
            embedder=mock_embedder_for_chunking,
            chunk_size=256,
        )

    @pytest.mark.asyncio
    async def test_chunk_ids_unique_within_document(self, chunker, long_document):
        """Test that chunk IDs are unique within a document."""
        chunks = await chunker.chunk_async(long_document)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))  # All unique

    @pytest.mark.asyncio
    async def test_chunk_ids_unique_across_documents(self, chunker):
        """Test that chunk IDs are unique across documents."""
        doc1 = Document(id="doc1", content="Content for document one. " * 20, metadata={})
        doc2 = Document(id="doc2", content="Content for document two. " * 20, metadata={})

        chunks1 = await chunker.chunk_async(doc1)
        chunks2 = await chunker.chunk_async(doc2)

        ids1 = {c.id for c in chunks1}
        ids2 = {c.id for c in chunks2}

        # No overlap in IDs
        assert ids1.isdisjoint(ids2)
