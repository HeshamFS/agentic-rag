"""
Configuration management for RAG Optimizer.

Uses pydantic-settings for environment variable support.
All settings can be overridden via environment variables
prefixed with RAG_.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    RAG Optimizer configuration settings.

    Environment variables are loaded with the RAG_ prefix.
    Example: RAG_QDRANT_URL, RAG_ANTHROPIC_API_KEY, etc.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Vector Database (Qdrant)
    # ==========================================================================

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API key (for cloud deployments)",
    )
    qdrant_grpc_port: int = Field(
        default=6334,
        description="Qdrant gRPC port for faster operations",
    )
    qdrant_prefer_grpc: bool = Field(
        default=True,
        description="Use gRPC instead of HTTP when available",
    )

    # ==========================================================================
    # Embedding Model
    # ==========================================================================

    embedding_model: str = Field(
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        description="HuggingFace model ID for embeddings",
    )
    embedding_device: str = Field(
        default="cuda",
        description="Device for embedding model (cuda, cpu, mps). Falls back to cpu if cuda unavailable.",
    )
    embedding_batch_size: int = Field(
        default=64,
        description="Batch size for embedding operations. Higher = faster but more memory.",
    )
    embedding_max_length: int = Field(
        default=8192,
        description="Maximum sequence length for embeddings",
    )

    # ==========================================================================
    # Reranking Model
    # ==========================================================================

    reranker_model: str = Field(
        default="jinaai/jina-reranker-v2-base-multilingual",
        description="HuggingFace model ID for reranking",
    )
    reranker_device: str = Field(
        default="cuda",
        description="Device for reranker model",
    )

    # ==========================================================================
    # LLM Providers (Multi-Provider Support)
    # ==========================================================================

    # Default provider
    llm_provider: Literal["claude", "openai", "gemini", "local"] = Field(
        default="claude",
        description="Default LLM provider",
    )
    llm_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Default model ID for generation (latest Claude Sonnet 4.5)",
    )

    # Anthropic (Claude)
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key for Claude models",
    )
    anthropic_max_retries: int = Field(
        default=3,
        description="Max retries for Anthropic API calls",
    )

    # OpenAI
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for GPT models",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="OpenAI API base URL (for proxies or Azure)",
    )

    # Google (Gemini)
    google_api_key: SecretStr | None = Field(
        default=None,
        description="Google API key for Gemini models",
    )
    gemini_thinking_level: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Gemini 3 thinking level (low, medium, high)",
    )

    # Local (Ollama / vLLM)
    local_llm_url: str = Field(
        default="http://localhost:11434",
        description="Local LLM server URL (Ollama format)",
    )
    local_llm_model: str = Field(
        default="qwen2.5:7b",
        description="Default local model name",
    )

    # Groq (Ultra-fast inference for contextual chunking)
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Groq API key for ultra-fast inference",
    )
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model for fast tasks (llama-3.1-8b-instant, llama-3.3-70b-versatile)",
    )

    # Contextual Chunking Provider (separate from main LLM for speed)
    contextual_chunking_provider: Literal["groq", "gemini", "openai", "claude"] = Field(
        default="groq",
        description="LLM provider for contextual chunk headers (groq recommended for speed)",
    )
    contextual_chunking_model: str | None = Field(
        default=None,
        description="Model for contextual chunking (defaults to provider's fastest model)",
    )

    # ==========================================================================
    # Generation Parameters
    # ==========================================================================

    default_temperature: float = Field(
        default=0.3,
        description="Default temperature for generation",
    )
    default_max_tokens: int = Field(
        default=4096,
        description="Default max tokens for generation",
    )
    max_context_tokens: int = Field(
        default=8000,
        description="Max tokens for context window",
    )

    # ==========================================================================
    # Retrieval Parameters
    # ==========================================================================

    default_top_k: int = Field(
        default=10,
        description="Default number of chunks to retrieve",
    )
    default_rerank_top_k: int = Field(
        default=5,
        description="Number of chunks after reranking",
    )
    hybrid_sparse_weight: float = Field(
        default=0.3,
        description="Weight for sparse retrieval in hybrid mode (0-1)",
    )

    # ==========================================================================
    # Chunking Parameters
    # ==========================================================================

    default_chunk_size: int = Field(
        default=512,
        description="Default chunk size in tokens",
    )
    default_chunk_overlap: int = Field(
        default=50,
        description="Default chunk overlap in tokens",
    )

    # ==========================================================================
    # Agentic Parameters
    # ==========================================================================

    enable_reflection: bool = Field(
        default=True,
        description="Enable Self-RAG reflection",
    )
    enable_planning: bool = Field(
        default=True,
        description="Enable query planning for complex queries",
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum self-correction iterations",
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="CRAG confidence threshold for fallback",
    )

    # ==========================================================================
    # Caching
    # ==========================================================================

    enable_semantic_cache: bool = Field(
        default=True,
        description="Enable semantic similarity caching",
    )
    cache_backend: Literal["memory", "disk", "redis"] = Field(
        default="memory",
        description="Cache backend: memory (fast), disk (persistent), redis (distributed)",
    )
    cache_directory: Path = Field(
        default=Path(".cache/agentic_rag"),
        description="Directory for disk cache",
    )
    cache_similarity_threshold: float = Field(
        default=0.95,
        description="Similarity threshold for cache hits",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds",
    )

    # Redis Cache Settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL (redis://host:port/db)",
    )
    redis_password: SecretStr | None = Field(
        default=None,
        description="Redis password (if required)",
    )
    redis_max_connections: int = Field(
        default=10,
        description="Redis connection pool max connections",
    )
    redis_cache_prefix: str = Field(
        default="rag:cache:",
        description="Redis key prefix for cache entries",
    )

    # ==========================================================================
    # Context Compression
    # ==========================================================================

    enable_compression: bool = Field(
        default=False,
        description="Enable context compression to reduce token costs",
    )
    compression_type: Literal["extractive", "longllmlingua"] = Field(
        default="extractive",
        description="Compression method: extractive (fast) or longllmlingua (accurate)",
    )
    compression_ratio: float = Field(
        default=0.5,
        description="Target compression ratio (0.5 = keep 50% of tokens)",
    )
    compression_min_sentences: int = Field(
        default=3,
        description="Minimum sentences to keep after compression",
    )

    # ==========================================================================
    # RAPTOR (Hierarchical Chunking)
    # ==========================================================================

    raptor_max_levels: int = Field(
        default=3,
        description="Maximum tree depth for RAPTOR hierarchical chunking",
    )
    raptor_clustering: Literal["kmeans", "gmm"] = Field(
        default="gmm",
        description="Clustering algorithm for RAPTOR (gmm recommended)",
    )
    raptor_min_cluster_size: int = Field(
        default=2,
        description="Minimum nodes to form a cluster",
    )
    raptor_summary_tokens: int = Field(
        default=200,
        description="Maximum tokens per summary node",
    )

    # ==========================================================================
    # Observability
    # ==========================================================================

    enable_tracing: bool = Field(
        default=True,
        description="Enable OpenTelemetry tracing",
    )
    trace_export_url: str = Field(
        default="",
        description="OTLP exporter URL for traces",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a specific provider."""
        key_map = {
            "claude": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
            "groq": self.groq_api_key,
            "local": None,  # No API key needed
        }
        secret = key_map.get(provider)
        return secret.get_secret_value() if secret else None

    def validate_provider_config(self, provider: str) -> bool:
        """Check if a provider is properly configured."""
        if provider == "local":
            return True  # Local doesn't need API key

        api_key = self.get_api_key(provider)
        return api_key is not None and len(api_key) > 0


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to avoid reloading settings on every access.
    """
    return Settings()
