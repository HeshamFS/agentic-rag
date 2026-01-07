# API Reference

> **Complete API Documentation for AgenticRAG**
>
> This reference covers all public classes, methods, and configuration options.

---

## Table of Contents

1. [Pipeline](#pipeline)
2. [Embeddings](#embeddings)
3. [Chunking](#chunking)
4. [Retrieval](#retrieval)
5. [Reranking](#reranking)
6. [Compression](#compression)
7. [Caching](#caching)
8. [Generation](#generation)
9. [Evaluation](#evaluation)
10. [GraphRAG](#graphrag)

---

## Pipeline

### PipelineBuilder

Fluent API for constructing RAG pipelines.

```python
from agentic_rag.pipeline import PipelineBuilder

class PipelineBuilder:
    """Builder for constructing RAG pipelines."""

    def with_embedder(
        self,
        embedder: str | Qwen3Embedder = "default",
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Configure embedding model.

        Args:
            embedder: "default", "small", "large" or Qwen3Embedder instance
            **kwargs: Additional Qwen3Embedder arguments (device, batch_size, etc.)

        Returns:
            Self for chaining
        """

    def with_vectordb(
        self,
        db_type: str = "qdrant",
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Configure vector database.

        Args:
            db_type: "qdrant" (more coming)
            **kwargs: Database-specific options

        Returns:
            Self for chaining
        """

    def with_chunking(
        self,
        strategy: str = "semantic",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Configure chunking strategy.

        Args:
            strategy: "semantic", "hierarchical", "raptor", "contextual"
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            **kwargs: Strategy-specific options

        Returns:
            Self for chaining
        """

    def with_retrieval(
        self,
        strategy: str = "hybrid",
        top_k: int = 10,
        use_hyde: bool = False,
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Configure retrieval strategy.

        Args:
            strategy: "dense", "sparse", "hybrid"
            top_k: Number of documents to retrieve
            use_hyde: Enable HyDE query expansion
            **kwargs: Strategy-specific options

        Returns:
            Self for chaining
        """

    def with_reranker(
        self,
        reranker: str | Any = "colbert",
        top_k: int = 5,
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Configure reranking.

        Args:
            reranker: "colbert", "cross-encoder" or instance
            top_k: Final number after reranking
            **kwargs: Reranker-specific options

        Returns:
            Self for chaining
        """

    def with_compression(
        self,
        method: str = "extractive",
        compression_ratio: float = 0.5,
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Configure context compression.

        Args:
            method: "extractive", "longllmlingua"
            compression_ratio: Target ratio (0.0-1.0)
            **kwargs: Method-specific options

        Returns:
            Self for chaining
        """

    def with_cache(
        self,
        backend: str = "memory",
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Configure semantic caching.

        Args:
            backend: "memory", "disk", "redis"
            similarity_threshold: Cache hit threshold
            ttl_seconds: Cache entry TTL
            **kwargs: Backend-specific options

        Returns:
            Self for chaining
        """

    def with_generator(
        self,
        provider: str = "claude",
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Configure LLM generation.

        Args:
            provider: "claude", "openai", "gemini", "local"
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Maximum output tokens
            **kwargs: Provider-specific options

        Returns:
            Self for chaining
        """

    def with_evaluation(
        self,
        enable_ragas: bool = True,
        enable_self_rag: bool = False,
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Configure evaluation.

        Args:
            enable_ragas: Enable RAGAS metrics
            enable_self_rag: Enable Self-RAG reflection
            **kwargs: Evaluation options

        Returns:
            Self for chaining
        """

    def with_graphrag(
        self,
        enabled: bool = True,
        **kwargs,
    ) -> "PipelineBuilder":
        """
        Enable GraphRAG.

        Args:
            enabled: Enable graph-based retrieval
            **kwargs: GraphRAG options

        Returns:
            Self for chaining
        """

    def build(self) -> "RAGPipeline":
        """Build the configured pipeline."""
```

### RAGPipeline

Main pipeline for ingestion and querying.

```python
class RAGPipeline:
    """Configured RAG pipeline."""

    async def ingest(
        self,
        documents: list[Document],
        collection: str,
        **kwargs,
    ) -> IngestionResult:
        """
        Ingest documents into collection.

        Args:
            documents: Documents to ingest
            collection: Target collection name
            **kwargs: Ingestion options

        Returns:
            IngestionResult with statistics
        """

    async def query(
        self,
        query: str,
        collection: str,
        top_k: int = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Query the pipeline.

        Args:
            query: User query
            collection: Collection to search
            top_k: Override default top_k
            **kwargs: Query options

        Returns:
            GenerationResult with response and sources
        """

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve without generation.

        Args:
            query: Search query
            collection: Collection to search
            top_k: Number to retrieve
            **kwargs: Retrieval options

        Returns:
            RetrievalResult with chunks and scores
        """
```

---

## Embeddings

### Qwen3Embedder

```python
from agentic_rag.embeddings import Qwen3Embedder

class Qwen3Embedder:
    """Qwen3 embedding model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 8192,
        normalize_embeddings: bool = True,
        use_cache: bool = True,
    ):
        """Initialize embedder."""

    @property
    def dimension(self) -> int:
        """Embedding dimension."""

    @property
    def model_name(self) -> str:
        """Model identifier."""

    async def embed_text(self, text: str) -> list[float]:
        """Embed single text."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""

    def clear_cache(self) -> None:
        """Clear embedding cache."""
```

### LateChunkingEmbedder

```python
from agentic_rag.embeddings import LateChunkingEmbedder

class LateChunkingEmbedder:
    """Late chunking embedder for context-aware chunk embeddings."""

    def __init__(
        self,
        model: str = None,
        device: str = "cuda",
        max_length: int = 8192,
    ):
        """Initialize late chunking embedder."""

    async def embed_document_with_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> list[list[float]]:
        """Embed chunks with document context."""
```

---

## Chunking

### SemanticChunker

```python
from agentic_rag.chunking import SemanticChunker

class SemanticChunker:
    """Semantic similarity-based chunking."""

    def __init__(
        self,
        embedder: Embedder,
        threshold_percentile: int = 50,
        window_size: int = 3,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        """Initialize semantic chunker."""

    async def chunk(self, document: Document) -> list[Chunk]:
        """Chunk document based on semantic boundaries."""
```

### RAPTORChunker

```python
from agentic_rag.chunking import RAPTORChunker

class RAPTORChunker:
    """RAPTOR hierarchical tree chunking."""

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        max_levels: int = 3,
        clustering: str = "gmm",
        min_cluster_size: int = 2,
        summary_tokens: int = 200,
    ):
        """Initialize RAPTOR chunker."""

    async def chunk_with_tree(
        self,
        document: Document,
    ) -> RAPTORTree:
        """Build RAPTOR tree with summaries."""
```

---

## Retrieval

### HybridRetriever

```python
from agentic_rag.retrieval import HybridRetriever

class HybridRetriever:
    """Hybrid dense + sparse retrieval."""

    def __init__(
        self,
        embedder: Embedder,
        vectordb: VectorDB,
        dense_weight: float = 0.7,
        fusion: str = "rrf",
    ):
        """Initialize hybrid retriever."""

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve with hybrid fusion."""
```

### HyDERetriever

```python
from agentic_rag.retrieval import HyDERetriever

class HyDERetriever:
    """HyDE (Hypothetical Document Embeddings) retriever."""

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        vectordb: VectorDB,
        num_hypotheses: int = 3,
    ):
        """Initialize HyDE retriever."""

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve using hypothetical document embeddings."""
```

---

## Reranking

### ColBERTReranker

```python
from agentic_rag.reranking import ColBERTReranker

class ColBERTReranker:
    """ColBERT late interaction reranker."""

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cuda",
    ):
        """Initialize ColBERT reranker."""

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Rerank chunks using MaxSim."""
```

---

## Compression

### ExtractiveCompressor

```python
from agentic_rag.compression import ExtractiveCompressor

class ExtractiveCompressor:
    """Extractive sentence-level compression."""

    def __init__(
        self,
        reranker: Reranker = None,
        compression_ratio: float = 0.5,
    ):
        """Initialize extractive compressor."""

    async def compress(
        self,
        query: str,
        chunks: list[Chunk],
        target_tokens: int = None,
    ) -> CompressionResult:
        """Compress by selecting top sentences."""
```

### LongLLMLinguaCompressor

```python
from agentic_rag.compression import LongLLMLinguaCompressor

class LongLLMLinguaCompressor:
    """LongLLMLingua perplexity-based compression."""

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-chat-hf",
        compression_ratio: float = 0.3,
    ):
        """Initialize LongLLMLingua compressor."""

    async def compress(
        self,
        query: str,
        chunks: list[Chunk],
        target_tokens: int = None,
    ) -> CompressionResult:
        """Compress using token-level perplexity."""
```

---

## Caching

### SemanticCache

```python
from agentic_rag.caching import SemanticCache

class SemanticCache:
    """In-memory semantic cache."""

    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float = 0.95,
        max_entries: int = 10000,
        ttl_seconds: int = 3600,
    ):
        """Initialize semantic cache."""

    async def get(self, query: str) -> CacheEntry | None:
        """Get cached response for similar query."""

    async def set(
        self,
        query: str,
        response: str,
        sources: list[Chunk] = None,
    ) -> None:
        """Cache a query-response pair."""

    async def invalidate(
        self,
        collection: str = None,
        pattern: str = None,
    ) -> int:
        """Invalidate cache entries."""

    def stats(self) -> dict:
        """Get cache statistics."""
```

### RedisSemanticCache

```python
from agentic_rag.caching import RedisSemanticCache

class RedisSemanticCache:
    """Redis-backed distributed semantic cache."""

    def __init__(
        self,
        embedder: Embedder,
        redis_url: str = "redis://localhost:6379/0",
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 86400,
    ):
        """Initialize Redis cache."""

    async def connect(self) -> None:
        """Connect to Redis."""

    async def close(self) -> None:
        """Close Redis connection."""
```

---

## Generation

### GeneratorFactory

```python
from agentic_rag.generation import create_generator, GeneratorFactory

def create_generator(
    provider: str = None,
    model: str = None,
    **kwargs,
) -> BaseGenerator:
    """Create generator for specified provider."""

class GeneratorFactory:
    """Factory for creating LLM generators."""

    @classmethod
    def create(
        cls,
        provider: str = None,
        model: str = None,
        **kwargs,
    ) -> BaseGenerator:
        """Create generator instance."""

    @classmethod
    def list_providers(cls) -> list[str]:
        """List available providers."""

    @classmethod
    def get_default_models(cls, provider: str) -> list[str]:
        """Get recommended models for provider."""
```

---

## Evaluation

### RAGASEvaluator

```python
from agentic_rag.evaluation import RAGASEvaluator

class RAGASEvaluator:
    """Complete RAGAS evaluation suite."""

    def __init__(self, generator: Generator):
        """Initialize RAGAS evaluator."""

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str = None,
    ) -> dict[str, EvaluationResult]:
        """Run all RAGAS evaluations."""

    def aggregate_scores(
        self,
        results: dict[str, EvaluationResult],
        weights: dict[str, float] = None,
    ) -> float:
        """Calculate weighted aggregate score."""
```

### SelfRAGEvaluator

```python
from agentic_rag.evaluation import SelfRAGEvaluator

class SelfRAGEvaluator:
    """Self-RAG reflection token evaluation."""

    def __init__(
        self,
        generator: Generator,
        regenerate_threshold: float = 0.5,
    ):
        """Initialize Self-RAG evaluator."""

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
    ) -> SelfRAGOutput:
        """Run Self-RAG evaluation."""

    def create_feedback(self, output: SelfRAGOutput) -> str:
        """Create human-readable feedback."""
```

---

## GraphRAG

### LLMEntityExtractor

```python
from agentic_rag.graph import LLMEntityExtractor

class LLMEntityExtractor:
    """LLM-based entity extraction."""

    def __init__(
        self,
        generator: Generator,
        entity_types: list[str] = None,
        max_entities_per_chunk: int = 50,
    ):
        """Initialize entity extractor."""

    async def extract(
        self,
        chunk: Chunk,
    ) -> ExtractionResult:
        """Extract entities and relationships."""
```

### LeidenCommunityDetector

```python
from agentic_rag.graph import LeidenCommunityDetector

class LeidenCommunityDetector:
    """Leiden algorithm community detection."""

    def __init__(
        self,
        resolution: float = 1.0,
        max_levels: int = 3,
        min_community_size: int = 2,
    ):
        """Initialize Leiden detector."""

    def detect(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> CommunityHierarchy:
        """Detect hierarchical communities."""
```

### GraphRAGRetriever

```python
from agentic_rag.graph import GraphRAGRetriever

class GraphRAGRetriever:
    """Graph-enhanced retrieval."""

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        graph_storage: GraphStorage,
    ):
        """Initialize GraphRAG retriever."""

    async def retrieve(
        self,
        query: str,
        mode: str = "auto",  # "local", "global", "auto"
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve using graph structure."""
```

---

## Data Models

### Document

```python
class Document(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any] = {}
    embedding: list[float] | None = None
    source: str | None = None
    created_at: datetime
```

### Chunk

```python
class Chunk(BaseModel):
    id: str
    content: str
    document_id: str
    metadata: dict[str, Any] = {}
    embedding: list[float] | None = None
    context_header: str | None = None
    position: int | None = None
    parent_id: str | None = None
    level: int = 0
```

### GenerationResult

```python
class GenerationResult(BaseModel):
    response: str
    sources: list[Chunk] = []
    confidence: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider: str = ""
    model: str = ""
    latency_ms: float | None = None
```

### RetrievalResult

```python
class RetrievalResult(BaseModel):
    chunks: list[Chunk]
    scores: list[float]
    retrieval_type: str
    query_embedding: list[float] | None = None
    retrieval_time_ms: float | None = None
```

