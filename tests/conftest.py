"""
Pytest configuration and comprehensive fixtures for RAG Optimizer testing.

This module provides fixtures for:
- Mock embedders, generators, and vector databases
- Async testing support
- Sample data (documents, chunks, queries)
- Integration test fixtures
- Benchmark data fixtures
"""

import asyncio
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from agentic_rag.config import Settings
from agentic_rag.core.models import (
    Chunk,
    Document,
    GenerationResult,
    RetrievalResult,
)

# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "requires_api: marks tests requiring external API")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Settings Fixtures
# =============================================================================


@pytest.fixture
def settings() -> Settings:
    """Test settings with defaults for local testing."""
    return Settings(
        qdrant_url="http://localhost:6333",
        llm_provider="gemini",
        llm_model="gemini-2.0-flash",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_device="cpu",
        default_temperature=0.3,
        default_max_tokens=1024,
        enable_tracing=False,
    )


@pytest.fixture
def test_settings_minimal() -> Settings:
    """Minimal settings for unit tests (no external dependencies)."""
    return Settings(
        qdrant_url="http://localhost:6333",
        llm_provider="local",
        embedding_device="cpu",
        enable_tracing=False,
    )


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_documents() -> list[Document]:
    """Sample documents for testing."""
    return [
        Document(
            id="doc_1",
            content="Python is a high-level programming language known for its simplicity and readability. "
            "It was created by Guido van Rossum and first released in 1991. Python supports multiple "
            "programming paradigms including procedural, object-oriented, and functional programming.",
            metadata={"source": "test", "topic": "programming", "language": "en"},
        ),
        Document(
            id="doc_2",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn "
            "and improve from experience without being explicitly programmed. It uses algorithms to "
            "identify patterns in data and make predictions or decisions.",
            metadata={"source": "test", "topic": "ai", "language": "en"},
        ),
        Document(
            id="doc_3",
            content="Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. "
            "It retrieves relevant documents from a knowledge base and uses them as context for generating "
            "accurate, grounded responses. This approach reduces hallucinations in LLM outputs.",
            metadata={"source": "test", "topic": "nlp", "language": "en"},
        ),
        Document(
            id="doc_4",
            content="Vector databases store high-dimensional vectors for similarity search. They use algorithms "
            "like HNSW (Hierarchical Navigable Small World) for efficient approximate nearest neighbor search. "
            "Popular options include Qdrant, Pinecone, Milvus, and Weaviate.",
            metadata={"source": "test", "topic": "databases", "language": "en"},
        ),
        Document(
            id="doc_5",
            content="Neural networks are computing systems inspired by biological neural networks. They consist "
            "of layers of interconnected nodes (neurons) that process information. Deep learning uses neural "
            "networks with many layers to learn complex patterns in data.",
            metadata={"source": "test", "topic": "ai", "language": "en"},
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Sample chunks with embeddings for testing."""

    # Create deterministic embeddings based on content hash
    def make_embedding(text: str, dim: int = 384) -> list[float]:
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        return [
            (int(h[i : i + 2], 16) / 255.0 - 0.5) for i in range(0, min(len(h), dim * 2), 2)
        ] + [0.0] * (dim - min(len(h) // 2, dim))

    return [
        Chunk(
            id="chunk_1",
            content="Python is a high-level programming language known for its simplicity.",
            document_id="doc_1",
            position=0,
            context_header="From: Python Programming Guide",
            embedding=make_embedding("python programming simplicity"),
            metadata={"topic": "programming"},
        ),
        Chunk(
            id="chunk_2",
            content="Machine learning uses algorithms to learn from data and make predictions.",
            document_id="doc_2",
            position=0,
            context_header="From: AI Fundamentals",
            embedding=make_embedding("machine learning algorithms data"),
            metadata={"topic": "ai"},
        ),
        Chunk(
            id="chunk_3",
            content="RAG combines retrieval with generation for accurate, grounded responses.",
            document_id="doc_3",
            position=0,
            context_header="From: NLP Techniques",
            embedding=make_embedding("rag retrieval generation"),
            metadata={"topic": "nlp"},
        ),
        Chunk(
            id="chunk_4",
            content="Vector databases use HNSW for efficient similarity search.",
            document_id="doc_4",
            position=0,
            context_header="From: Database Systems",
            embedding=make_embedding("vector databases hnsw similarity"),
            metadata={"topic": "databases"},
        ),
        Chunk(
            id="chunk_5",
            content="Neural networks learn complex patterns through layers of neurons.",
            document_id="doc_5",
            position=0,
            context_header="From: Deep Learning Basics",
            embedding=make_embedding("neural networks deep learning"),
            metadata={"topic": "ai"},
        ),
    ]


@pytest.fixture
def sample_queries() -> list[dict[str, Any]]:
    """Sample queries with expected answers for testing."""
    return [
        {
            "question": "What is Python programming language?",
            "expected_topics": ["programming"],
            "expected_answer_contains": ["programming language", "simplicity"],
        },
        {
            "question": "How does machine learning work?",
            "expected_topics": ["ai"],
            "expected_answer_contains": ["algorithms", "learn", "data"],
        },
        {
            "question": "What is RAG and why is it useful?",
            "expected_topics": ["nlp"],
            "expected_answer_contains": ["retrieval", "generation", "grounded"],
        },
        {
            "question": "What are vector databases used for?",
            "expected_topics": ["databases"],
            "expected_answer_contains": ["similarity", "search", "vectors"],
        },
    ]


@pytest.fixture
def multi_hop_queries() -> list[dict[str, Any]]:
    """Multi-hop queries requiring reasoning across multiple documents."""
    return [
        {
            "question": "How does machine learning relate to neural networks?",
            "expected_documents": ["doc_2", "doc_5"],
            "reasoning_steps": 2,
        },
        {
            "question": "What AI techniques are used in RAG systems?",
            "expected_documents": ["doc_2", "doc_3"],
            "reasoning_steps": 2,
        },
    ]


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic embeddings."""
    embedder = MagicMock()
    embedder.dimension = 384
    embedder.model_name = "mock-embedder"

    def make_embedding(text: str) -> list[float]:
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        return [(int(h[i : i + 2], 16) / 255.0 - 0.5) for i in range(0, 64, 2)] + [0.0] * 352

    async def embed_async(texts):
        if isinstance(texts, str):
            return make_embedding(texts)
        return [make_embedding(t) for t in texts]

    embedder.embed = MagicMock(side_effect=lambda texts: [make_embedding(t) for t in texts])
    embedder.embed_async = AsyncMock(side_effect=embed_async)
    embedder.embed_query = MagicMock(side_effect=make_embedding)
    embedder.embed_query_async = AsyncMock(side_effect=make_embedding)

    return embedder


@pytest.fixture
def mock_generator():
    """Mock generator that returns predictable responses."""
    generator = MagicMock()
    generator.provider = "mock"
    generator.model = "mock-model"

    async def generate_async(query: str, context: list[Chunk], **kwargs) -> GenerationResult:
        context_text = " ".join([c.content for c in context])
        response = f"Based on the provided context about {query[:50]}..., the answer is: {context_text[:100]}..."
        return GenerationResult(
            response=response,
            sources=context[:3],
            confidence=0.85,
            provider="mock",
            model="mock-model",
            input_tokens=len(query.split()),
            output_tokens=len(response.split()),
        )

    generator.generate = AsyncMock(side_effect=generate_async)
    return generator


@pytest.fixture
def mock_vectordb():
    """Mock vector database for testing."""
    db = MagicMock()
    db.db_type = "mock"
    db._stored_chunks: dict[str, list[Chunk]] = {}

    async def create_collection(name: str, dimension: int, **kwargs):
        db._stored_chunks[name] = []

    async def upsert(collection: str, chunks: list[Chunk], **kwargs):
        if collection not in db._stored_chunks:
            db._stored_chunks[collection] = []
        db._stored_chunks[collection].extend(chunks)

    async def search(collection: str, query_vector: list[float], top_k: int = 10, **kwargs):
        chunks = db._stored_chunks.get(collection, [])
        # Return chunks with mock scores
        return [(chunk, 0.9 - i * 0.1) for i, chunk in enumerate(chunks[:top_k])]

    async def get_all(collection: str, **kwargs):
        return db._stored_chunks.get(collection, [])

    async def collection_exists(name: str):
        return name in db._stored_chunks

    async def delete_collection(name: str):
        if name in db._stored_chunks:
            del db._stored_chunks[name]

    db.create_collection = AsyncMock(side_effect=create_collection)
    db.upsert = AsyncMock(side_effect=upsert)
    db.search = AsyncMock(side_effect=search)
    db.get_all = AsyncMock(side_effect=get_all)
    db.collection_exists = AsyncMock(side_effect=collection_exists)
    db.delete_collection = AsyncMock(side_effect=delete_collection)

    return db


@pytest.fixture
def mock_reranker():
    """Mock reranker for testing."""
    reranker = MagicMock()

    async def rerank_async(query: str, chunks: list[Chunk], top_k: int = 5, **kwargs):
        # Simple mock reranking - just return chunks in order with scores
        return [(chunk, 1.0 - i * 0.1) for i, chunk in enumerate(chunks[:top_k])]

    reranker.rerank = AsyncMock(side_effect=rerank_async)
    return reranker


# =============================================================================
# Temporary File Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing (requires pypdf)."""
    pdf_path = temp_dir / "sample.pdf"
    # Create a minimal PDF
    pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test PDF Content) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
300
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def sample_text_files(temp_dir: Path) -> list[Path]:
    """Create sample text files for ingestion testing."""
    files = []
    for i in range(5):
        file_path = temp_dir / f"doc_{i}.txt"
        file_path.write_text(
            f"Document {i}: This is sample content for testing document ingestion. "
            f"It contains information about topic {i} and various related concepts."
        )
        files.append(file_path)
    return files


@pytest.fixture
def large_document(temp_dir: Path) -> Path:
    """Create a large document for stress testing."""
    file_path = temp_dir / "large_doc.txt"
    # Create a ~1MB document
    content = "This is a paragraph of text for testing. " * 1000
    paragraphs = [content] * 25
    file_path.write_text("\n\n".join(paragraphs))
    return file_path


# =============================================================================
# Benchmark Data Fixtures
# =============================================================================


@pytest.fixture
def hotpotqa_sample() -> list[dict[str, Any]]:
    """Sample HotPotQA-style questions for benchmark testing."""
    return [
        {
            "id": "hotpot_1",
            "question": "What programming language was created by Guido van Rossum?",
            "answer": "Python",
            "type": "comparison",
            "level": "easy",
            "supporting_facts": [["Python Programming Guide", 0]],
            "context": [
                [
                    "Python Programming Guide",
                    [
                        "Python is a high-level programming language created by Guido van Rossum.",
                        "It was first released in 1991.",
                    ],
                ]
            ],
        },
        {
            "id": "hotpot_2",
            "question": "What technique combines retrieval with generation to reduce hallucinations?",
            "answer": "RAG (Retrieval-Augmented Generation)",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": [["NLP Techniques", 0]],
            "context": [
                [
                    "NLP Techniques",
                    [
                        "RAG combines retrieval with generation for accurate responses.",
                        "This approach reduces hallucinations in LLM outputs.",
                    ],
                ]
            ],
        },
        {
            "id": "hotpot_3",
            "question": "What algorithm do vector databases use for similarity search?",
            "answer": "HNSW (Hierarchical Navigable Small World)",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": [["Database Systems", 0]],
            "context": [
                [
                    "Database Systems",
                    [
                        "Vector databases store high-dimensional vectors.",
                        "They use HNSW for efficient similarity search.",
                    ],
                ]
            ],
        },
    ]


@pytest.fixture
def natural_questions_sample() -> list[dict[str, Any]]:
    """Sample Natural Questions-style data for benchmark testing."""
    return [
        {
            "id": "nq_1",
            "question": "what is machine learning",
            "short_answer": "a subset of artificial intelligence",
            "long_answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from experience.",
            "document_title": "AI Fundamentals",
        },
        {
            "id": "nq_2",
            "question": "how do neural networks work",
            "short_answer": "through layers of neurons",
            "long_answer": "Neural networks process information through layers of interconnected nodes called neurons.",
            "document_title": "Deep Learning Basics",
        },
    ]


@pytest.fixture
def ragas_evaluation_data() -> list[dict[str, Any]]:
    """Sample data for RAGAS evaluation."""
    return [
        {
            "question": "What is Python?",
            "ground_truth": "Python is a high-level programming language known for its simplicity and readability.",
            "contexts": [
                "Python is a high-level programming language known for its simplicity.",
                "Python was created by Guido van Rossum.",
            ],
            "answer": "Python is a high-level programming language that is known for being simple and readable.",
        },
        {
            "question": "What is machine learning?",
            "ground_truth": "Machine learning is a subset of AI that enables systems to learn from data.",
            "contexts": [
                "Machine learning uses algorithms to learn from data.",
                "ML is a subset of artificial intelligence.",
            ],
            "answer": "Machine learning is a type of AI that uses algorithms to learn from data and make predictions.",
        },
    ]


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def qdrant_test_collection(settings):
    """Create a test collection in Qdrant (requires running Qdrant)."""
    from agentic_rag.vectordb import QdrantVectorDB

    collection_name = f"test_collection_{os.getpid()}"
    db = QdrantVectorDB(settings=settings)

    try:
        if await db.collection_exists(collection_name):
            await db.delete_collection(collection_name)
        await db.create_collection(collection_name, dimension=384)
        yield collection_name, db
    finally:
        try:
            await db.delete_collection(collection_name)
            await db.close()
        except Exception:
            pass


# =============================================================================
# Assertion Helpers
# =============================================================================


class RAGAssertions:
    """Helper class for RAG-specific assertions."""

    @staticmethod
    def assert_retrieval_quality(
        result: RetrievalResult, expected_doc_ids: list[str], min_recall: float = 0.5
    ):
        """Assert retrieval quality metrics."""
        retrieved_doc_ids = [chunk.document_id for chunk in result.chunks]
        hits = sum(1 for doc_id in expected_doc_ids if doc_id in retrieved_doc_ids)
        recall = hits / len(expected_doc_ids) if expected_doc_ids else 0.0
        assert recall >= min_recall, f"Recall {recall:.2f} below minimum {min_recall}"

    @staticmethod
    def assert_generation_quality(
        result: GenerationResult, expected_keywords: list[str], min_match: float = 0.5
    ):
        """Assert generation quality metrics."""
        response_lower = result.response.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        match_ratio = matched / len(expected_keywords) if expected_keywords else 0.0
        assert match_ratio >= min_match, (
            f"Keyword match {match_ratio:.2f} below minimum {min_match}"
        )

    @staticmethod
    def assert_response_grounded(result: GenerationResult, min_sources: int = 1):
        """Assert response is grounded in sources."""
        assert len(result.sources) >= min_sources, (
            f"Expected at least {min_sources} sources, got {len(result.sources)}"
        )

    @staticmethod
    def assert_latency(latency_ms: float, max_latency_ms: float):
        """Assert latency is within acceptable bounds."""
        assert latency_ms <= max_latency_ms, (
            f"Latency {latency_ms:.2f}ms exceeds maximum {max_latency_ms}ms"
        )


@pytest.fixture
def rag_assertions() -> RAGAssertions:
    """Provide RAG assertion helpers."""
    return RAGAssertions()
