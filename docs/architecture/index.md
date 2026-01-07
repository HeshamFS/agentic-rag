# Core Architecture

> **Modular RAG Pipeline Design**
>
> This document covers the architectural foundations: protocols, models, pipeline builder, and how components interact.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Protocol-Based Architecture](#protocol-based-architecture)
3. [Core Data Models](#core-data-models)
4. [Pipeline Builder Pattern](#pipeline-builder-pattern)
5. [Query Flow](#query-flow)
6. [Ingestion Flow](#ingestion-flow)
7. [Agentic Mode](#agentic-mode)

---

## Design Philosophy

AgenticRAG follows the **2025 Modular RAG** paradigm, enabling flexible component arrangement for specific use cases.

### Key Principles

1. **Protocol-Based Components**: All major components implement runtime-checkable protocols
2. **Fluent Builder API**: Chainable configuration with sensible defaults
3. **Async-First**: All I/O operations are async for maximum concurrency
4. **Provider Agnostic**: Swap embedding models, LLMs, and vector databases
5. **Observability Built-in**: Tracing, metrics, and cost tracking

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PipelineBuilder                              │
│  (Fluent configuration API)                                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          RAGPipeline                                 │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Embedder │  │ Chunker  │  │Retriever │  │ Reranker │            │
│  │(Protocol)│  │(Protocol)│  │(Protocol)│  │(Protocol)│            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       │             │             │             │                   │
│  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐            │
│  │Compressor│  │ Cache    │  │Generator │  │Evaluator │            │
│  │(Protocol)│  │(Protocol)│  │(Protocol)│  │(Protocol)│            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                                                      │
│  ┌──────────┐  ┌──────────┐                                        │
│  │VectorDB  │  │  Graph   │                                        │
│  │(Protocol)│  │ Storage  │                                        │
│  └──────────┘  └──────────┘                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Protocol-Based Architecture

All components implement Python `Protocol` classes, enabling runtime type checking and easy component swapping.

### Embedder Protocol

```python
@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding models."""

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text into a vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into vectors."""
        ...
```

**Implementations**: `Qwen3Embedder`, `LateChunkingEmbedder`

### Chunker Protocol

```python
@runtime_checkable
class Chunker(Protocol):
    """Protocol for document chunking strategies."""

    @property
    def strategy_name(self) -> str:
        """Return the chunking strategy identifier."""
        ...

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        ...
```

**Implementations**: `SemanticChunker`, `HierarchicalChunker`, `RAPTORChunker`, `ContextualChunker`

### Retriever Protocol

```python
@runtime_checkable
class Retriever(Protocol):
    """Protocol for retrieval components."""

    @property
    def retriever_type(self) -> str:
        """Return the retriever type identifier."""
        ...

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        ...

    async def retrieve_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve using a pre-computed embedding vector."""
        ...
```

**Implementations**: `DenseRetriever`, `HybridRetriever`, `HyDERetriever`, `MultiQueryRetriever`, `RAPTORRetriever`

### Reranker Protocol

```python
@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking models."""

    @property
    def model_name(self) -> str:
        """Return the reranker model identifier."""
        ...

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Rerank chunks by relevance to query."""
        ...
```

**Implementations**: `ColBERTReranker`, `CrossEncoderReranker`

### Compressor Protocol

```python
@runtime_checkable
class Compressor(Protocol):
    """Protocol for context compression."""

    @property
    def compression_type(self) -> str:
        """Return the compression type identifier."""
        ...

    async def compress(
        self,
        query: str,
        chunks: list[Chunk],
        target_tokens: int | None = None,
    ) -> list[Chunk]:
        """Compress chunks to reduce context size."""
        ...
```

**Implementations**: `ExtractiveCompressor`, `LongLLMLinguaCompressor`

### Generator Protocol

```python
@runtime_checkable
class Generator(Protocol):
    """Protocol for response generation."""

    @property
    def provider(self) -> str:
        """Return the LLM provider (claude, openai, gemini, local)."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    async def generate(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response given query and context."""
        ...

    async def generate_text(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Simple text generation without RAG context."""
        ...
```

**Implementations**: `ClaudeGenerator`, `OpenAIGenerator`, `GeminiGenerator`, `LocalGenerator`

### VectorDB Protocol

```python
@runtime_checkable
class VectorDB(Protocol):
    """Protocol for vector database operations."""

    @property
    def db_type(self) -> str:
        """Return the database type identifier."""
        ...

    async def create_collection(
        self,
        name: str,
        dimension: int,
        **kwargs: Any,
    ) -> None:
        """Create a new collection."""
        ...

    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
    ) -> None:
        """Insert or update chunks in the collection."""
        ...

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar vectors."""
        ...

    async def hybrid_search(
        self,
        collection: str,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> list[tuple[Chunk, float]]:
        """Hybrid search combining dense and sparse retrieval."""
        ...
```

**Implementations**: `QdrantVectorDB`

---

## Core Data Models

All data structures are Pydantic models for validation and serialization.

### Document

```python
class Document(BaseModel):
    """A document in the RAG system."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    source: str | None = None  # File path, URL, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Chunk

```python
class Chunk(BaseModel):
    """A chunk of a document."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    document_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    # Contextual retrieval enhancement
    context_header: str | None = None

    # Position tracking for hierarchical chunking
    position: int | None = None
    parent_id: str | None = None
    level: int = 0  # 0 = leaf, higher = more abstract
```

### RetrievalResult

```python
class RetrievalResult(BaseModel):
    """Result from retrieval."""

    chunks: list[Chunk]
    scores: list[float]
    retrieval_type: str  # "dense", "sparse", "hybrid", "hyde", "rrf"
    query_embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Performance tracking
    retrieval_time_ms: float | None = None
    total_candidates: int | None = None
```

### GenerationResult

```python
class GenerationResult(BaseModel):
    """Result from LLM generation."""

    response: str
    sources: list[Chunk] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    # Token usage for cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Provider info
    provider: str = ""
    model: str = ""

    # Generation metadata
    finish_reason: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### EvaluationResult

```python
class EvaluationResult(BaseModel):
    """Comprehensive evaluation result."""

    # RAGAS metrics
    metrics: dict[str, float] = Field(default_factory=dict)

    # Self-RAG reflection tokens
    reflection_tokens: dict[str, bool] = Field(default_factory=dict)

    # Claim verification
    claims: list[ClaimVerification] = Field(default_factory=list)

    # Overall quality score
    overall_score: float = Field(ge=0.0, le=1.0, default=0.0)

    # Recommendations for improvement
    suggestions: list[str] = Field(default_factory=list)
```

---

## Pipeline Builder Pattern

The `PipelineBuilder` provides a fluent API for constructing pipelines.

### Basic Usage

```python
from agentic_rag.pipeline import PipelineBuilder

# Minimal pipeline (uses defaults)
pipeline = PipelineBuilder().build()

# Build with all features
pipeline = (
    PipelineBuilder()
    # Embedding
    .with_embedder(embedder="default")

    # Chunking
    .with_chunking(
        strategy="semantic",
        chunk_size=512,
        chunk_overlap=50,
    )

    # Retrieval
    .with_retrieval(
        strategy="hybrid",
        top_k=20,
        use_hyde=True,
    )

    # Reranking
    .with_reranker(reranker="jina", top_k=10)

    # Compression
    .with_compression(
        method="extractive",
        compression_ratio=0.5,
    )

    # Caching
    .with_cache(
        backend="memory",
        similarity_threshold=0.95,
        ttl_seconds=3600,
    )

    # Generation
    .with_generator(
        provider="claude",
        model="claude-sonnet-4-5-20250929",
        temperature=0.3,
    )

    # Evaluation
    .with_evaluation(enable_ragas=True)

    .build()
)
```

### Builder Methods

| Method | Purpose |
|--------|---------|
| `.with_embedder()` | Configure embedding model |
| `.with_vectordb()` | Configure vector database |
| `.with_chunking()` | Configure chunking strategy |
| `.with_retrieval()` | Configure retrieval strategy |
| `.with_reranker()` | Configure reranking |
| `.with_compression()` | Configure context compression |
| `.with_cache()` | Configure semantic caching |
| `.with_generator()` | Configure LLM provider |
| `.with_evaluation()` | Configure evaluation metrics |
| `.with_graphrag()` | Enable GraphRAG |
| `.with_contextual_chunking()` | Enable contextual headers |
| `.as_agentic()` | Enable agentic mode |
| `.build()` | Build the pipeline |

### Configuration Object

```python
class RAGConfig(BaseModel):
    """Configuration for RAG pipeline."""

    # Embedding configuration
    embedding_model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 32

    # Chunking configuration
    chunk_strategy: Literal["semantic", "hierarchical", "contextual", "recursive"] = "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 50
    add_context_headers: bool = True

    # Retrieval configuration
    retrieval_strategy: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = 10
    rerank_top_k: int = 5
    use_hyde: bool = True
    use_rrf: bool = True
    sparse_weight: float = 0.3

    # Generation configuration
    llm_provider: Literal["claude", "openai", "gemini", "local"] = "claude"
    llm_model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.3
    max_tokens: int = 4096

    # Agentic configuration
    enable_reflection: bool = True
    enable_planning: bool = True
    max_iterations: int = 3

    # CRAG configuration
    confidence_threshold: float = 0.7
    web_fallback: bool = False

    # Caching
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = 0.95
```

---

## Query Flow

The standard query pipeline follows this flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Query: "What is X?"                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      1. Semantic Cache        │
                    │    Check for similar query    │
                    └───────────────┬───────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                     HIT  ▼              MISS ▼
              ┌────────────────┐    ┌────────────────┐
              │ Return cached  │    │   Continue...  │
              │   response     │    │                │
              └────────────────┘    └───────┬────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────┐
                    │     2. Query Enhancement      │
                    │  - Multi-Query (4 variations) │
                    │  - OR HyDE (hypothetical doc) │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      3. Hybrid Retrieval      │
                    │  - Dense embedding search     │
                    │  - BM25 sparse search         │
                    │  - RRF fusion                 │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      4. ColBERT Reranking     │
                    │  - MaxSim scoring             │
                    │  - Top-K selection            │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     5. GraphRAG Enrichment    │
                    │   (if enabled)                │
                    │  - Entity context             │
                    │  - Relationship context       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    6. Context Compression     │
                    │   (if enabled)                │
                    │  - Reduce tokens by 50-70%    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      7. LLM Generation        │
                    │  - Build prompt with context  │
                    │  - Generate response          │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      8. Cache Response        │
                    │  - Store for future queries   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Return Response         │
                    │    with sources & metadata    │
                    └───────────────────────────────┘
```

### Pipeline Steps Tracking

Each step is tracked for observability:

```python
pipeline_steps = [
    {
        "name": "Semantic Cache",
        "duration_ms": 5.2,
        "details": {"hit": False}
    },
    {
        "name": "Multi-Query Retrieval",
        "duration_ms": 234.5,
        "details": {
            "query_variations": ["What is X?", "Explain X", ...],
            "chunks_retrieved": 30
        }
    },
    {
        "name": "ColBERT Reranking",
        "duration_ms": 45.3,
        "details": {"input_chunks": 30, "output_chunks": 10}
    },
    {
        "name": "Context Compression",
        "duration_ms": 12.1,
        "details": {
            "original_tokens": 2500,
            "compressed_tokens": 850,
            "reduction_percent": 66.0
        }
    },
    {
        "name": "Final LLM Generation",
        "duration_ms": 1234.5,
        "details": {"provider": "gemini", "context_chunks": 10}
    }
]
```

---

## Ingestion Flow

Document ingestion follows this flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Documents to Ingest                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   1. Collection Setup         │
                    │   - Create if not exists      │
                    │   - Configure HNSW index      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   2. Chunking                 │
                    │   - Semantic boundaries       │
                    │   - OR RAPTOR tree building   │
                    │   - OR Contextual headers     │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   3. Embedding                │
                    │   - Batch embed all chunks    │
                    │   - Qwen3 with instruct mode  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   4. Vector DB Upsert         │
                    │   - Store chunks + embeddings │
                    │   - Index for search          │
                    └───────────────┬───────────────┘
                                    │
                                    ▼ (if GraphRAG enabled)
                    ┌───────────────────────────────┐
                    │   5. Entity Extraction        │
                    │   - LLM extracts entities     │
                    │   - Extract relationships     │
                    │   - Merge duplicates          │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   6. Community Detection      │
                    │   - Leiden algorithm          │
                    │   - Hierarchical clustering   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   7. Community Summarization  │
                    │   - LLM generates summaries   │
                    │   - Store in graph storage    │
                    └───────────────────────────────┘
```

---

## Agentic Mode

When enabled via `.as_agentic()`, the pipeline uses multi-agent orchestration.

### Agent Types

| Agent | Role |
|-------|------|
| **OrchestratorAgent** | Coordinates all agents, manages workflow |
| **RouterAgent** | Classifies query intent, routes to strategy |
| **RetrieverAgent** | Decides retrieval strategy, handles CRAG |
| **GeneratorAgent** | Generates responses with reflection |
| **ValidatorAgent** | Validates response quality, triggers retry |

### Agentic Workflow

```
Query
  │
  ▼
┌─────────────────┐
│ OrchestratorAgent│
└───────┬─────────┘
        │
        ▼
┌─────────────────┐     ┌─────────────────┐
│  RouterAgent    │────▶│ Classify Intent │
└───────┬─────────┘     └─────────────────┘
        │
        ├─── Factual ────▶ Direct retrieval
        ├─── Analytical ──▶ Multi-step reasoning
        ├─── Comparative ─▶ Multi-doc retrieval
        └─── Exploratory ─▶ Broad retrieval
        │
        ▼
┌─────────────────┐
│ RetrieverAgent  │
└───────┬─────────┘
        │
        ├─── High Confidence ──▶ Use retrieved docs
        ├─── Medium ───────────▶ Transform query, retry
        └─── Low ──────────────▶ Fallback to web search
        │
        ▼
┌─────────────────┐
│ GeneratorAgent  │
└───────┬─────────┘
        │
        ▼
┌─────────────────┐     ┌─────────────────┐
│ ValidatorAgent  │────▶│  Self-RAG Check │
└───────┬─────────┘     │  ISREL, ISSUP   │
        │               │  ISUSE tokens   │
        │               └─────────────────┘
        │
        ├─── Pass ────▶ Return response
        └─── Fail ────▶ Retry (up to max_iterations)
```

### Self-RAG Reflection Tokens

```python
class ReflectionToken(BaseModel):
    """Self-RAG reflection token."""

    token_type: Literal["ISREL", "ISSUP", "ISUSE"]
    value: bool
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: str = ""

# Token meanings:
# ISREL: Is the retrieved content relevant to the query?
# ISSUP: Is the response supported by the retrieved content?
# ISUSE: Is the response useful for answering the query?
```

---

## Next Steps

- [Embeddings](../algorithms/embeddings.md) - Qwen3 and Late Chunking
- [Retrieval](../algorithms/retrieval.md) - Dense, Hybrid, HyDE
- [Reranking](../algorithms/reranking.md) - ColBERT MaxSim
- [API Reference](../api/index.md) - Complete API documentation

