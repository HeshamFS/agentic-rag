"""Tests for core models and protocols."""

from agentic_rag.core.models import Chunk, Document, RAGConfig


class TestDocument:
    """Tests for Document model."""

    def test_create_document(self):
        doc = Document(
            id="test_doc",
            content="Test content",
            metadata={"source": "test"},
        )
        assert doc.id == "test_doc"
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test"

    def test_document_defaults(self):
        doc = Document(content="Test")
        assert doc.id  # Auto-generated
        assert doc.metadata == {}
        assert doc.embedding is None


class TestChunk:
    """Tests for Chunk model."""

    def test_create_chunk(self):
        chunk = Chunk(
            id="chunk_1",
            content="Chunk content",
            document_id="doc_1",
        )
        assert chunk.id == "chunk_1"
        assert chunk.document_id == "doc_1"


class TestRAGConfig:
    """Tests for RAGConfig model."""

    def test_default_config(self):
        config = RAGConfig()
        assert config.chunk_strategy == "semantic"
        assert config.retrieval_strategy == "hybrid"
        assert config.llm_provider == "claude"

    def test_custom_config(self):
        config = RAGConfig(
            llm_provider="openai",
            llm_model="gpt-4o",
            top_k=20,
        )
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o"
        assert config.top_k == 20
