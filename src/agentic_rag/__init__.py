"""
RAG Optimizer - Agentic RAG Pipeline with 2025 State-of-the-Art Techniques.

A production-grade, general-purpose RAG framework featuring:
- Multi-agent orchestration (Router, Retriever, Evaluator, Generator)
- Hybrid retrieval (Dense + BM25 with RRF fusion)
- HyDE (Hypothetical Document Embeddings)
- Contextual retrieval with chunk headers
- Self-RAG reflection tokens (ISREL, ISSUP, ISUSE)
- CRAG (Corrective RAG) with confidence-based fallbacks
- GraphRAG for knowledge graph-based retrieval
- Late Chunking for context-aware embeddings
- ColBERT late interaction reranking
- RAGAS evaluation metrics
- Multi-provider LLM support (Claude, OpenAI, Gemini, Local)
"""

__version__ = "0.1.0"

from agentic_rag.core.models import (
    Chunk,
    Document,
    GenerationResult,
    RAGConfig,
    ReflectionToken,
    RetrievalResult,
)

# GraphRAG exports
from agentic_rag.graph import (
    Community,
    CommunityDetector,
    Entity,
    EntityExtractor,
    GraphRAGRetriever,
    GraphRetriever,
    GraphStorage,
    LeidenCommunityDetector,
    LLMEntityExtractor,
    NetworkXStorage,
    Relationship,
)
from agentic_rag.pipeline.builder import PipelineBuilder

__all__ = [
    # Core models
    "Document",
    "Chunk",
    "RetrievalResult",
    "GenerationResult",
    "ReflectionToken",
    "RAGConfig",
    # Pipeline
    "PipelineBuilder",
    # GraphRAG
    "Entity",
    "Relationship",
    "EntityExtractor",
    "LLMEntityExtractor",
    "Community",
    "CommunityDetector",
    "LeidenCommunityDetector",
    "GraphRetriever",
    "GraphRAGRetriever",
    "GraphStorage",
    "NetworkXStorage",
    # Version
    "__version__",
]
