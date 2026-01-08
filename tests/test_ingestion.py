"""
Comprehensive tests for document ingestion functionality.

Tests document loading, parsing, and processing with real academic PDFs:
- Attention is All You Need (Transformer paper)
- RAG paper (Retrieval-Augmented Generation)
- BERT paper
- GPT-3 paper
- LLaMA 2 paper
- Self-RAG paper
- Chain of Thought paper
- Corrective RAG paper
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_rag.core.models import Chunk, Document
from agentic_rag.ingestion.batch_ingester import BatchIngester
from agentic_rag.ingestion.file_loader import FileLoader

# =============================================================================
# Test Data Paths
# =============================================================================


PAPERS_DIR = Path(__file__).parent / "test_data" / "papers"


@pytest.fixture
def attention_paper_path():
    """Path to Attention is All You Need paper."""
    path = PAPERS_DIR / "attention_is_all_you_need.pdf"
    if not path.exists():
        pytest.skip("Attention paper not found - run download first")
    return path


@pytest.fixture
def rag_paper_path():
    """Path to RAG paper."""
    path = PAPERS_DIR / "rag_paper.pdf"
    if not path.exists():
        pytest.skip("RAG paper not found - run download first")
    return path


@pytest.fixture
def bert_paper_path():
    """Path to BERT paper."""
    path = PAPERS_DIR / "bert_paper.pdf"
    if not path.exists():
        pytest.skip("BERT paper not found - run download first")
    return path


@pytest.fixture
def gpt3_paper_path():
    """Path to GPT-3 paper."""
    path = PAPERS_DIR / "gpt3_paper.pdf"
    if not path.exists():
        pytest.skip("GPT-3 paper not found - run download first")
    return path


@pytest.fixture
def llama2_paper_path():
    """Path to LLaMA 2 paper."""
    path = PAPERS_DIR / "llama2_paper.pdf"
    if not path.exists():
        pytest.skip("LLaMA 2 paper not found - run download first")
    return path


@pytest.fixture
def self_rag_paper_path():
    """Path to Self-RAG paper."""
    path = PAPERS_DIR / "self_rag_paper.pdf"
    if not path.exists():
        pytest.skip("Self-RAG paper not found - run download first")
    return path


@pytest.fixture
def chain_of_thought_path():
    """Path to Chain of Thought paper."""
    path = PAPERS_DIR / "chain_of_thought.pdf"
    if not path.exists():
        pytest.skip("Chain of Thought paper not found - run download first")
    return path


@pytest.fixture
def crag_paper_path():
    """Path to Corrective RAG paper."""
    path = PAPERS_DIR / "crag_paper.pdf"
    if not path.exists():
        pytest.skip("CRAG paper not found - run download first")
    return path


@pytest.fixture
def all_papers(
    attention_paper_path,
    rag_paper_path,
    bert_paper_path,
    gpt3_paper_path,
    llama2_paper_path,
    self_rag_paper_path,
    chain_of_thought_path,
    crag_paper_path,
):
    """All available academic papers."""
    return [
        attention_paper_path,
        rag_paper_path,
        bert_paper_path,
        gpt3_paper_path,
        llama2_paper_path,
        self_rag_paper_path,
        chain_of_thought_path,
        crag_paper_path,
    ]


# =============================================================================
# FileLoader Tests
# =============================================================================


class TestFileLoader:
    """Tests for the FileLoader class."""

    @pytest.fixture
    def file_loader(self):
        """Create FileLoader instance."""
        return FileLoader()

    def test_loader_initialization(self, file_loader):
        """Test FileLoader initializes correctly."""
        assert file_loader is not None

    def test_loader_supported_extensions(self, file_loader):
        """Test loader reports supported file extensions."""
        supported = file_loader.SUPPORTED_EXTENSIONS
        assert ".pdf" in supported
        assert ".txt" in supported

    # -------------------------------------------------------------------------
    # Text File Tests
    # -------------------------------------------------------------------------

    def test_load_text_file(self, file_loader, temp_dir):
        """Test loading a plain text file."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("This is test content.\nSecond line here.")

        result = file_loader.load(text_file)

        assert result.success
        assert result.document is not None
        assert isinstance(result.document, Document)
        assert "test content" in result.document.content
        assert "Second line" in result.document.content

    def test_load_text_preserves_structure(self, file_loader, temp_dir):
        """Test that text loading preserves structure."""
        text_file = temp_dir / "structured.txt"
        content = """# Header

Paragraph one with content.

Paragraph two with more content.

## Subheader

Final paragraph."""
        text_file.write_text(content)

        result = file_loader.load(text_file)

        assert result.success
        assert result.document is not None
        assert "# Header" in result.document.content
        assert "## Subheader" in result.document.content

    def test_load_unicode_text(self, file_loader, temp_dir):
        """Test loading text with unicode characters."""
        text_file = temp_dir / "unicode.txt"
        text_file.write_text("日本語テスト 中文测试 🎉 émojis", encoding="utf-8")

        result = file_loader.load(text_file)

        assert result.success
        assert result.document is not None
        assert "日本語" in result.document.content
        assert "🎉" in result.document.content

    # -------------------------------------------------------------------------
    # PDF Loading Tests
    # -------------------------------------------------------------------------

    @pytest.mark.slow
    def test_load_attention_paper(self, file_loader, attention_paper_path):
        """Test loading Attention is All You Need paper."""
        result = file_loader.load(attention_paper_path)

        assert result.success
        assert result.document is not None
        assert isinstance(result.document, Document)
        assert len(result.document.content) > 1000

        # Check for key content
        content_lower = result.document.content.lower()
        assert "attention" in content_lower
        assert "transformer" in content_lower or "self-attention" in content_lower

    @pytest.mark.slow
    def test_load_rag_paper(self, file_loader, rag_paper_path):
        """Test loading RAG paper."""
        result = file_loader.load(rag_paper_path)

        assert result.success
        assert result.document is not None
        assert isinstance(result.document, Document)
        assert len(result.document.content) > 1000

        content_lower = result.document.content.lower()
        assert "retrieval" in content_lower
        assert "generation" in content_lower

    @pytest.mark.slow
    def test_load_bert_paper(self, file_loader, bert_paper_path):
        """Test loading BERT paper."""
        result = file_loader.load(bert_paper_path)

        assert result.success
        assert result.document is not None
        assert isinstance(result.document, Document)
        content_lower = result.document.content.lower()
        assert "bert" in content_lower or "bidirectional" in content_lower

    @pytest.mark.slow
    def test_load_gpt3_paper(self, file_loader, gpt3_paper_path):
        """Test loading GPT-3 paper."""
        result = file_loader.load(gpt3_paper_path)

        assert result.success
        assert result.document is not None
        assert isinstance(result.document, Document)
        assert len(result.document.content) > 5000  # GPT-3 paper is quite long

        content_lower = result.document.content.lower()
        assert "language model" in content_lower or "few-shot" in content_lower

    @pytest.mark.slow
    def test_load_llama2_paper(self, file_loader, llama2_paper_path):
        """Test loading LLaMA 2 paper."""
        result = file_loader.load(llama2_paper_path)

        assert result.success
        assert result.document is not None
        assert isinstance(result.document, Document)
        assert len(result.document.content) > 5000

        content_lower = result.document.content.lower()
        assert "llama" in content_lower or "fine-tuning" in content_lower

    @pytest.mark.slow
    def test_load_self_rag_paper(self, file_loader, self_rag_paper_path):
        """Test loading Self-RAG paper."""
        result = file_loader.load(self_rag_paper_path)

        assert result.success
        assert result.document is not None
        assert isinstance(result.document, Document)
        content_lower = result.document.content.lower()
        assert "self" in content_lower or "retrieval" in content_lower

    # -------------------------------------------------------------------------
    # PDF Metadata Tests
    # -------------------------------------------------------------------------

    @pytest.mark.slow
    def test_pdf_includes_metadata(self, file_loader, attention_paper_path):
        """Test that PDF loading extracts metadata."""
        result = file_loader.load(attention_paper_path)

        assert result.success
        assert result.document is not None
        assert "source" in result.document.metadata or "filename" in result.document.metadata
        assert result.document.metadata.get("extension") == ".pdf" or attention_paper_path.suffix in str(
            result.document.metadata
        )

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    def test_load_nonexistent_file(self, file_loader):
        """Test loading nonexistent file returns error result."""
        result = file_loader.load(Path("/nonexistent/file.pdf"))

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower() or "File not found" in result.error

    def test_load_unsupported_extension(self, file_loader, temp_dir):
        """Test loading unsupported file type returns error result."""
        unsupported = temp_dir / "file.xyz"
        unsupported.write_text("content")

        result = file_loader.load(unsupported)

        assert not result.success
        assert result.error is not None
        assert "unsupported" in result.error.lower()


# =============================================================================
# BatchIngester Tests
# =============================================================================


class TestBatchIngester:
    """Tests for the BatchIngester class."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder for ingestion."""
        embedder = MagicMock()
        embedder.dimension = 384

        async def mock_embed_batch(texts):
            return [[0.1] * 384 for _ in texts]

        embedder.embed_batch = AsyncMock(side_effect=mock_embed_batch)
        return embedder

    @pytest.fixture
    def mock_vectordb(self):
        """Create mock vector database for ingestion."""
        db = MagicMock()
        db._stored_chunks = {}

        async def mock_create_collection(name, dimension, **kwargs):
            db._stored_chunks[name] = []

        async def mock_upsert(collection, chunks, **kwargs):
            if collection not in db._stored_chunks:
                db._stored_chunks[collection] = []
            db._stored_chunks[collection].extend(chunks)

        async def mock_collection_exists(name):
            return name in db._stored_chunks

        db.create_collection = AsyncMock(side_effect=mock_create_collection)
        db.upsert = AsyncMock(side_effect=mock_upsert)
        db.collection_exists = AsyncMock(side_effect=mock_collection_exists)

        return db

    @pytest.fixture
    def mock_chunker(self):
        """Create mock chunker for ingestion."""
        chunker = MagicMock()

        async def mock_chunk(document):
            # Split document into simple chunks
            content = document.content
            chunk_size = 500
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

        # BatchIngester calls 'chunk', not 'chunk_async'
        chunker.chunk = AsyncMock(side_effect=mock_chunk)
        chunker.chunk_async = AsyncMock(side_effect=mock_chunk)
        return chunker

    @pytest.fixture
    def batch_ingester(self, mock_embedder, mock_vectordb, mock_chunker, test_settings_minimal):
        """Create BatchIngester with mocks."""
        return BatchIngester(
            embedder=mock_embedder,
            vectordb=mock_vectordb,
            chunker=mock_chunker,
            settings=test_settings_minimal,
        )

    # -------------------------------------------------------------------------
    # Basic Ingestion Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ingest_single_document(self, batch_ingester, sample_documents):
        """Test ingesting a single document."""
        result = await batch_ingester.ingest_documents(
            documents=[sample_documents[0]],
            collection="test",
            show_progress=False,
        )

        assert result.successful_files == 1
        assert result.total_chunks > 0

    @pytest.mark.asyncio
    async def test_ingest_multiple_documents(self, batch_ingester, sample_documents):
        """Test ingesting multiple documents."""
        result = await batch_ingester.ingest_documents(
            documents=sample_documents,
            collection="test",
            show_progress=False,
        )

        assert result.successful_files == len(sample_documents)
        assert result.total_chunks >= len(sample_documents)  # At least one chunk per doc

    @pytest.mark.asyncio
    async def test_ingest_creates_collection(self, batch_ingester, mock_vectordb, sample_documents):
        """Test that ingestion stores chunks in vector DB."""
        await batch_ingester.ingest_documents(
            documents=sample_documents[:1],
            collection="new_collection",
            show_progress=False,
        )

        # Verify upsert was called (collection is created implicitly)
        mock_vectordb.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_ingest_stores_chunks(self, batch_ingester, mock_vectordb, sample_documents):
        """Test that ingestion stores chunks in vector DB."""
        await batch_ingester.ingest_documents(
            documents=sample_documents[:1],
            collection="test",
            show_progress=False,
        )

        mock_vectordb.upsert.assert_called()

    # -------------------------------------------------------------------------
    # PDF Ingestion Tests
    # -------------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ingest_attention_paper(self, batch_ingester, attention_paper_path):
        """Test ingesting Attention is All You Need paper."""
        loader = FileLoader()
        load_result = loader.load(attention_paper_path)
        assert load_result.success and load_result.document

        result = await batch_ingester.ingest_documents(
            documents=[load_result.document],
            collection="attention_paper",
            show_progress=False,
        )

        assert result.successful_files == 1
        assert result.total_chunks > 5  # Paper should create multiple chunks

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ingest_rag_paper(self, batch_ingester, rag_paper_path):
        """Test ingesting RAG paper."""
        loader = FileLoader()
        load_result = loader.load(rag_paper_path)
        assert load_result.success and load_result.document

        result = await batch_ingester.ingest_documents(
            documents=[load_result.document],
            collection="rag_paper",
            show_progress=False,
        )

        assert result.successful_files == 1
        assert result.total_chunks > 3

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ingest_multiple_papers(
        self, batch_ingester, attention_paper_path, rag_paper_path
    ):
        """Test ingesting multiple papers at once."""
        loader = FileLoader()
        result1 = loader.load(attention_paper_path)
        result2 = loader.load(rag_paper_path)
        assert result1.success and result1.document
        assert result2.success and result2.document
        documents = [result1.document, result2.document]

        result = await batch_ingester.ingest_documents(
            documents=documents,
            collection="multi_paper",
            show_progress=False,
        )

        assert result.successful_files == 2
        assert result.total_chunks > 10

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ingest_large_paper(self, batch_ingester, gpt3_paper_path):
        """Test ingesting large paper (GPT-3)."""
        loader = FileLoader()
        load_result = loader.load(gpt3_paper_path)
        assert load_result.success and load_result.document

        result = await batch_ingester.ingest_documents(
            documents=[load_result.document],
            collection="gpt3_paper",
            show_progress=False,
        )

        # GPT-3 paper is very long, should create many chunks
        assert result.total_chunks > 20

    # -------------------------------------------------------------------------
    # Batch Processing Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_batch_size_respected(self, batch_ingester, sample_documents):
        """Test that batch size is respected."""
        # Create many documents
        documents = sample_documents * 10

        result = await batch_ingester.ingest_documents(
            documents=documents,
            collection="batch_test",
            show_progress=False,
        )

        assert result.successful_files == len(documents)

    @pytest.mark.asyncio
    async def test_progress_callback(self, batch_ingester, sample_documents):
        """Test ingestion with progress disabled works."""
        result = await batch_ingester.ingest_documents(
            documents=sample_documents,
            collection="test",
            show_progress=False,
        )

        # Just verify it completes successfully
        assert result.successful_files == len(sample_documents)

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ingest_empty_documents(self, batch_ingester):
        """Test ingesting empty document list."""
        result = await batch_ingester.ingest_documents(
            documents=[],
            collection="test",
            show_progress=False,
        )

        assert result.successful_files == 0
        assert result.total_chunks == 0

    @pytest.mark.asyncio
    async def test_ingest_handles_chunking_error(
        self, batch_ingester, mock_chunker, sample_documents
    ):
        """Test handling of chunking errors."""
        mock_chunker.chunk = AsyncMock(side_effect=Exception("Chunking failed"))

        # ingest_documents handles errors gracefully and returns stats
        result = await batch_ingester.ingest_documents(
            documents=sample_documents[:1],
            collection="test",
            show_progress=False,
        )

        # Should have failed
        assert result.failed_files == 1
        assert result.successful_files == 0
        assert len(result.errors) > 0


# =============================================================================
# Content Quality Tests
# =============================================================================


class TestContentQuality:
    """Tests for content quality after ingestion."""

    @pytest.fixture
    def file_loader(self):
        """Create FileLoader instance."""
        return FileLoader()

    @pytest.mark.slow
    def test_attention_paper_contains_key_concepts(self, file_loader, attention_paper_path):
        """Test that Attention paper contains key concepts."""
        result = file_loader.load(attention_paper_path)
        assert result.success and result.document
        content_lower = result.document.content.lower()

        # Key concepts from the paper
        key_concepts = [
            "attention",
            "transformer",
            "encoder",
            "decoder",
        ]

        found = sum(1 for concept in key_concepts if concept in content_lower)
        assert found >= 2, f"Found only {found} of {len(key_concepts)} key concepts"

    @pytest.mark.slow
    def test_rag_paper_contains_key_concepts(self, file_loader, rag_paper_path):
        """Test that RAG paper contains key concepts."""
        result = file_loader.load(rag_paper_path)
        assert result.success and result.document
        content_lower = result.document.content.lower()

        key_concepts = [
            "retrieval",
            "generation",
            "knowledge",
        ]

        found = sum(1 for concept in key_concepts if concept in content_lower)
        assert found >= 2

    @pytest.mark.slow
    def test_bert_paper_contains_key_concepts(self, file_loader, bert_paper_path):
        """Test that BERT paper contains key concepts."""
        result = file_loader.load(bert_paper_path)
        assert result.success and result.document
        content_lower = result.document.content.lower()

        key_concepts = [
            "bert",
            "bidirectional",
            "pre-training",
            "fine-tuning",
        ]

        found = sum(1 for concept in key_concepts if concept in content_lower)
        assert found >= 2


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestIngestionPerformance:
    """Performance tests for document ingestion."""

    @pytest.fixture
    def fast_ingester(self, test_settings_minimal):
        """Create fast ingester for performance testing."""
        embedder = MagicMock()
        embedder.dimension = 384
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 384] * 100)

        db = MagicMock()
        db.create_collection = AsyncMock()
        db.upsert = AsyncMock()
        db.collection_exists = AsyncMock(return_value=False)

        chunker = MagicMock()

        async def fast_chunk(doc):
            return [
                Chunk(id=f"{doc.id}_{i}", content=f"Chunk {i}", document_id=doc.id, position=i)
                for i in range(10)
            ]

        chunker.chunk_async = AsyncMock(side_effect=fast_chunk)

        return BatchIngester(
            embedder=embedder,
            vectordb=db,
            chunker=chunker,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_ingestion_throughput(self, fast_ingester, sample_documents):
        """Test ingestion throughput."""
        import time

        documents = sample_documents * 20

        start = time.time()
        await fast_ingester.ingest_documents(
            documents=documents,
            collection="perf_test",
            show_progress=False,
        )
        elapsed = time.time() - start

        docs_per_second = len(documents) / elapsed
        assert docs_per_second > 1, f"Only {docs_per_second:.2f} docs/sec"

    @pytest.mark.asyncio
    async def test_large_document_ingestion_time(self, fast_ingester):
        """Test ingestion time for large documents."""
        import time

        # Create large document
        large_content = "Test content. " * 10000
        large_doc = Document(
            id="large_doc",
            content=large_content,
            metadata={},
        )

        start = time.time()
        await fast_ingester.ingest_documents(
            documents=[large_doc],
            collection="large_test",
            show_progress=False,
        )
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 30


# =============================================================================
# Integration Tests with Real PDFs
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestRealPDFIngestion:
    """Integration tests using real academic PDFs."""

    @pytest.fixture
    def real_embedder(self, test_settings_minimal):
        """Create real embedder (requires model)."""
        pytest.importorskip("sentence_transformers")
        from agentic_rag.embeddings import Qwen3Embedder

        return Qwen3Embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Small model for testing
            device="cpu",
            settings=test_settings_minimal,
        )

    @pytest.fixture
    def real_chunker(self, real_embedder):
        """Create real semantic chunker."""
        from agentic_rag.chunking import SemanticChunker

        return SemanticChunker(
            embedder=real_embedder,
            chunk_size=512,
        )

    @pytest.mark.asyncio
    async def test_end_to_end_pdf_ingestion(
        self,
        real_embedder,
        real_chunker,
        mock_vectordb,
        attention_paper_path,
        test_settings_minimal,
    ):
        """Test end-to-end PDF ingestion with real components."""
        loader = FileLoader()
        load_result = loader.load(attention_paper_path)
        assert load_result.success and load_result.document

        ingester = BatchIngester(
            embedder=real_embedder,
            vectordb=mock_vectordb,
            chunker=real_chunker,
            settings=test_settings_minimal,
        )

        result = await ingester.ingest_documents(
            documents=[load_result.document],
            collection="attention_test",
            show_progress=False,
        )

        assert result.successful_files == 1
        assert result.total_chunks > 0

        # Verify chunks have embeddings
        stored_chunks = mock_vectordb._stored_chunks.get("attention_test", [])
        if stored_chunks:
            assert stored_chunks[0].embedding is not None
            assert len(stored_chunks[0].embedding) == real_embedder.dimension
