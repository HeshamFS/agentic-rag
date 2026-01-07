"""
Pydantic models for the RAG pipeline.

Defines all data structures used throughout the system including:
- Documents and Chunks
- Retrieval and Generation results
- Self-RAG reflection tokens
- Pipeline configuration
- Tracing and observability
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# =============================================================================
# Core Document Models
# =============================================================================


class Document(BaseModel):
    """
    A document in the RAG system.

    Documents are the source material that gets chunked, embedded,
    and stored in the vector database for retrieval.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    source: str | None = None  # File path, URL, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """
    A chunk of a document.

    Chunks are the units of retrieval - documents are split into
    chunks using various strategies (semantic, hierarchical, etc.)
    and each chunk is independently embedded and indexed.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    document_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    # Contextual retrieval enhancement
    # Prepended context like "This chunk is from section X of document Y..."
    context_header: str | None = None

    # Position tracking for hierarchical chunking
    position: int | None = None
    parent_id: str | None = None
    level: int = 0  # 0 = leaf, higher = more abstract


# =============================================================================
# Query Classification
# =============================================================================


class QueryIntent(str, Enum):
    """
    Classified query intent.

    Used by the RouterAgent to determine the appropriate
    retrieval and generation strategy.
    """

    FACTUAL = "factual"  # Simple fact lookup
    ANALYTICAL = "analytical"  # Requires reasoning
    COMPARATIVE = "comparative"  # Compare multiple items
    PROCEDURAL = "procedural"  # Step-by-step instructions
    EXPLORATORY = "exploratory"  # Open-ended exploration


# =============================================================================
# Retrieval Models
# =============================================================================


class RetrievalDecision(BaseModel):
    """
    Agent's decision about retrieval strategy.

    The RetrieverAgent analyzes the query and context to decide
    the optimal retrieval approach based on CRAG principles.
    """

    should_retrieve: bool = True
    retrieval_strategy: Literal["dense", "sparse", "hybrid", "hyde", "multi_query"] = "hybrid"
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    reasoning: str = ""

    # CRAG: Fallback to web search if retrieval quality is low
    fallback_to_web: bool = False

    # Retrieval parameters
    top_k: int = 10
    use_reranking: bool = True


class RetrievalResult(BaseModel):
    """
    Result from retrieval.

    Contains the retrieved chunks along with their scores
    and metadata about the retrieval process.
    """

    chunks: list[Chunk]
    scores: list[float]
    retrieval_type: str  # "dense", "sparse", "hybrid", "hyde", "rrf"
    query_embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Performance tracking
    retrieval_time_ms: float | None = None
    total_candidates: int | None = None


# =============================================================================
# Generation Models
# =============================================================================


class GenerationResult(BaseModel):
    """
    Result from LLM generation.

    Includes the response, source attribution,
    and usage statistics.
    """

    response: str
    sources: list[Chunk] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    # Token usage for cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Provider info
    provider: str = ""  # "claude", "openai", "gemini", "local"
    model: str = ""

    # Generation metadata
    finish_reason: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Evaluation Models
# =============================================================================


class ReflectionToken(BaseModel):
    """
    Self-RAG reflection token.

    Part of the Self-RAG paper's approach to self-assessment.
    Enables agents to evaluate their own outputs.
    """

    token_type: Literal["ISREL", "ISSUP", "ISUSE"]
    value: bool
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: str = ""

    @property
    def description(self) -> str:
        """Human-readable description of the token."""
        descriptions = {
            "ISREL": "Is the retrieved content relevant to the query?",
            "ISSUP": "Is the response supported by the retrieved content?",
            "ISUSE": "Is the response useful for answering the query?",
        }
        return descriptions.get(self.token_type, "")


class ClaimVerification(BaseModel):
    """
    Result of NLI-based claim verification.

    Used for faithfulness evaluation - checking if
    claims in the response are supported by context.
    """

    claim: str
    is_supported: bool
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    contradiction_reason: str | None = None


class EvaluationResult(BaseModel):
    """
    Comprehensive evaluation result.

    Combines RAGAS metrics with Self-RAG reflection tokens
    and claim verification for complete quality assessment.
    """

    # RAGAS metrics
    metrics: dict[str, float] = Field(default_factory=dict)
    # Standard metrics: context_precision, context_recall, faithfulness, answer_relevancy

    # Self-RAG reflection tokens
    reflection_tokens: dict[str, bool] = Field(default_factory=dict)
    # Keys: ISREL, ISSUP, ISUSE

    # Claim verification
    claims: list[ClaimVerification] = Field(default_factory=list)

    # Overall quality score (weighted combination)
    overall_score: float = Field(ge=0.0, le=1.0, default=0.0)

    # Recommendations for improvement
    suggestions: list[str] = Field(default_factory=list)


# =============================================================================
# Pipeline Tracing
# =============================================================================


class PipelineStep(BaseModel):
    """
    A step in the RAG pipeline execution.

    Used for observability and debugging.
    """

    step_id: str = Field(default_factory=lambda: str(uuid4()))
    step_type: str  # "embed", "retrieve", "rerank", "compress", "generate", "evaluate"
    step_name: str = ""

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0

    # I/O
    input_summary: str = ""
    output_summary: str = ""

    # Resource usage
    tokens_used: int = 0
    cost_usd: float = 0.0

    # Error tracking
    error: str | None = None
    error_type: str | None = None


class PipelineTrace(BaseModel):
    """
    Full trace of a pipeline execution.

    Provides complete observability into the RAG pipeline
    for debugging, optimization, and monitoring.
    """

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    steps: list[PipelineStep] = Field(default_factory=list)

    # Final results
    final_result: GenerationResult | None = None
    evaluation: EvaluationResult | None = None

    # Aggregate metrics
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Metadata
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Configuration Model
# =============================================================================


class RAGConfig(BaseModel):
    """
    Configuration for RAG pipeline.

    Central configuration object that controls all aspects
    of the pipeline behavior.
    """

    # Embedding configuration
    embedding_model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 64  # Higher = faster but more memory

    # Chunking configuration
    # "semantic" gives +70% accuracy over fixed-size (RAG.md research)
    chunk_strategy: Literal["semantic", "hierarchical", "contextual", "recursive"] = "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 50
    add_context_headers: bool = True  # Contextual retrieval

    # Retrieval configuration
    retrieval_strategy: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = 10
    rerank_top_k: int = 5
    use_hyde: bool = True
    use_rrf: bool = True  # Reciprocal Rank Fusion
    sparse_weight: float = 0.3  # For hybrid retrieval

    # Generation configuration (Multi-Provider)
    llm_provider: Literal["claude", "openai", "gemini", "local"] = "claude"
    llm_model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.3
    max_tokens: int = 4096
    max_context_tokens: int = 8000

    # Agentic configuration
    enable_reflection: bool = True
    enable_planning: bool = True
    max_iterations: int = 3  # Max self-correction iterations

    # CRAG configuration
    confidence_threshold: float = 0.7
    web_fallback: bool = False

    # Evaluation configuration
    enable_self_rag: bool = True
    ragas_metrics: list[str] = Field(
        default_factory=lambda: [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]
    )

    # Observability
    enable_tracing: bool = True
    trace_export_url: str = ""

    # Caching
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = 0.95

    # Model suggestions per provider (for reference)
    @staticmethod
    def get_model_suggestions(provider: str) -> list[str]:
        """Get suggested models for a provider."""
        suggestions = {
            "claude": [
                "claude-sonnet-4-5-20250929",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
            ],
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "o1-preview",
            ],
            "gemini": [
                "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ],
            "local": [
                "qwen2.5:7b",
                "llama3.3:70b",
                "mistral:7b",
            ],
        }
        return suggestions.get(provider, [])
