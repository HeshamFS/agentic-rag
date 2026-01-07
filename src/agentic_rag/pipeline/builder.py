"""
Fluent API for building RAG pipelines.

Provides a clean, chainable interface for constructing
RAG pipelines with all components configured.
"""

from typing import Any, Literal, Self

from agentic_rag.caching import DiskSemanticCache, RedisConfig, RedisSemanticCache, SemanticCache
from agentic_rag.compression import BaseCompressor, ExtractiveCompressor, LongLLMLinguaCompressor
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import RAGConfig
from agentic_rag.embeddings import Qwen3Embedder
from agentic_rag.generation import ProviderType, create_generator
from agentic_rag.graph.extractor import Entity, LLMEntityExtractor, Relationship

# GraphRAG imports
from agentic_rag.graph.storage import GraphStorage, NetworkXStorage
from agentic_rag.vectordb import QdrantVectorDB


class PipelineBuilder:
    """
    Fluent builder for constructing RAG pipelines.

    Example:
        pipeline = (
            PipelineBuilder()
            .with_embedder("default")  # or Qwen3Embedder instance
            .with_vectordb("qdrant")
            .with_chunking("semantic", chunk_size=512)
            .with_retrieval("hybrid", top_k=10, use_hyde=True)
            .with_reranker("jina")
            .with_generator(provider="claude", model="claude-sonnet-4-20250514")
            .with_evaluation(enable_ragas=True, enable_self_rag=True)
            .as_agentic(enable_reflection=True, enable_planning=True)
            .build()
        )

        # Use the pipeline
        result = await pipeline.query("What is X?", collection="my-docs")
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the pipeline builder.

        Args:
            settings: Settings instance. If None, loads from environment.
        """
        self._settings = settings or get_settings()
        self._config = RAGConfig()

        # Components (set during building)
        self._embedder: Qwen3Embedder | None = None
        self._vectordb: QdrantVectorDB | None = None
        self._generator: Any = None
        self._reranker: Any = None
        self._chunker: Any = None

        # Pipeline flags
        self._is_agentic: bool = False
        self._is_built: bool = False

        # Advanced features
        self._use_contextual_chunking: bool = False
        self._use_graphrag: bool = False
        self._graph_storage: GraphStorage | None = None
        self._use_multi_query: bool = True
        self._num_queries: int = 4

        # Caching
        self._cache: SemanticCache | None = None
        self._cache_backend: Literal["memory", "disk", "redis"] = "memory"

        # Compression
        self._compressor: BaseCompressor | None = None
        self._compression_config: dict[str, Any] | None = None

    # =========================================================================
    # Embedder Configuration
    # =========================================================================

    def with_embedder(
        self,
        embedder: str | Qwen3Embedder = "default",
        **kwargs: Any,
    ) -> Self:
        """
        Configure the embedding model.

        Args:
            embedder: Either a preset name ("default", "small", "large")
                     or a Qwen3Embedder instance.
            **kwargs: Additional arguments for Qwen3Embedder:
                device: Device to use (cuda, cpu, mps).
                batch_size: Batch size for embeddings.
                max_length: Maximum sequence length.

        Returns:
            Self for chaining.
        """
        if isinstance(embedder, str):
            from agentic_rag.embeddings import create_embedder

            self._embedder = create_embedder(embedder, settings=self._settings, **kwargs)
        else:
            self._embedder = embedder

        self._config.embedding_model = self._embedder.model_name
        self._config.embedding_dimension = self._embedder.dimension
        return self

    # =========================================================================
    # Vector Database Configuration
    # =========================================================================

    def with_vectordb(
        self,
        db_type: str = "qdrant",
        url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Configure the vector database.

        Args:
            db_type: Database type (currently only "qdrant" supported).
            url: Database URL. Defaults to settings.
            api_key: API key. Defaults to settings.
            **kwargs: Additional database arguments.

        Returns:
            Self for chaining.
        """
        if db_type != "qdrant":
            raise ValueError(f"Unsupported database type: {db_type}")

        self._vectordb = QdrantVectorDB(
            settings=self._settings,
            url=url,
            api_key=api_key,
        )
        return self

    # =========================================================================
    # Chunking Configuration
    # =========================================================================

    def with_chunking(
        self,
        strategy: str = "semantic",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        contextual: bool = True,
        raptor_levels: int = 3,
        raptor_clustering: str = "gmm",
        **kwargs: Any,
    ) -> Self:
        """
        Configure document chunking.

        Args:
            strategy: Chunking strategy (semantic, hierarchical, recursive, raptor).
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks.
            contextual: Add context headers for contextual retrieval.
            raptor_levels: Max tree depth for RAPTOR strategy.
            raptor_clustering: Clustering algorithm for RAPTOR (gmm, kmeans).
            **kwargs: Additional chunking arguments.

        Returns:
            Self for chaining.

        Note:
            RAPTOR strategy builds hierarchical trees with summaries.
            Best for long documents requiring multi-level understanding.
        """
        self._config.chunk_strategy = strategy  # type: ignore
        self._config.chunk_size = chunk_size
        self._config.chunk_overlap = chunk_overlap
        self._config.add_context_headers = contextual

        # Store RAPTOR config if using raptor strategy
        if strategy == "raptor":
            self._raptor_config = {
                "max_levels": raptor_levels,
                "clustering_algorithm": raptor_clustering,
                **kwargs,
            }
        return self

    # =========================================================================
    # Retrieval Configuration
    # =========================================================================

    def with_retrieval(
        self,
        strategy: str = "hybrid",
        top_k: int = 10,
        use_hyde: bool = False,
        use_multi_query: bool = True,
        num_queries: int = 4,
        use_rrf: bool = True,
        sparse_weight: float = 0.3,
        **kwargs: Any,
    ) -> Self:
        """
        Configure retrieval strategy.

        Args:
            strategy: Retrieval strategy (dense, sparse, hybrid).
            top_k: Number of chunks to retrieve.
            use_hyde: Enable HyDE (Hypothetical Document Embeddings).
            use_multi_query: Enable Multi-Query retrieval (generates query variations).
            num_queries: Number of query variations for multi-query (default 4).
            use_rrf: Enable RRF (Reciprocal Rank Fusion).
            sparse_weight: Weight for sparse retrieval in hybrid mode.
            **kwargs: Additional retrieval arguments.

        Returns:
            Self for chaining.
        """
        self._config.retrieval_strategy = strategy  # type: ignore
        self._config.top_k = top_k
        self._config.use_hyde = use_hyde
        self._config.use_rrf = use_rrf
        self._config.sparse_weight = sparse_weight
        # Store multi-query config
        self._use_multi_query = use_multi_query
        self._num_queries = num_queries
        return self

    # =========================================================================
    # Reranker Configuration
    # =========================================================================

    def with_reranker(
        self,
        reranker: str | Any = "colbert",
        top_k: int = 5,
        **kwargs: Any,
    ) -> Self:
        """
        Configure reranking.

        Args:
            reranker: Reranker name ("colbert", "cross-encoder") or instance.
            top_k: Number of chunks after reranking.
            **kwargs: Additional reranker arguments.

        Returns:
            Self for chaining.
        """
        self._config.rerank_top_k = top_k
        return self

    # =========================================================================
    # Generator Configuration
    # =========================================================================

    def with_generator(
        self,
        provider: ProviderType | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Configure the LLM generator.

        Args:
            provider: LLM provider (claude, openai, gemini, local).
            model: Model ID.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            **kwargs: Additional generator arguments.

        Returns:
            Self for chaining.
        """
        self._config.llm_provider = provider or self._settings.llm_provider
        if model:
            self._config.llm_model = model
        if temperature is not None:
            self._config.temperature = temperature
        if max_tokens is not None:
            self._config.max_tokens = max_tokens

        # Create generator
        self._generator = create_generator(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            settings=self._settings,
            **kwargs,
        )
        return self

    # =========================================================================
    # Evaluation Configuration
    # =========================================================================

    def with_evaluation(
        self,
        enable_ragas: bool = True,
        enable_self_rag: bool = True,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Configure evaluation.

        Args:
            enable_ragas: Enable RAGAS metrics.
            enable_self_rag: Enable Self-RAG reflection tokens.
            metrics: Specific RAGAS metrics to enable.
            **kwargs: Additional evaluation arguments.

        Returns:
            Self for chaining.
        """
        self._config.enable_self_rag = enable_self_rag
        if metrics:
            self._config.ragas_metrics = metrics
        return self

    # =========================================================================
    # Agentic Configuration
    # =========================================================================

    def as_agentic(
        self,
        enable_reflection: bool = True,
        enable_planning: bool = True,
        max_iterations: int = 3,
        confidence_threshold: float = 0.7,
        **kwargs: Any,
    ) -> Self:
        """
        Configure as an agentic pipeline.

        Args:
            enable_reflection: Enable Self-RAG reflection.
            enable_planning: Enable query planning for complex queries.
            max_iterations: Maximum self-correction iterations.
            confidence_threshold: CRAG confidence threshold.
            **kwargs: Additional agentic arguments.

        Returns:
            Self for chaining.
        """
        self._is_agentic = True
        self._config.enable_reflection = enable_reflection
        self._config.enable_planning = enable_planning
        self._config.max_iterations = max_iterations
        self._config.confidence_threshold = confidence_threshold
        return self

    def as_standard(self) -> Self:
        """Configure as a standard (non-agentic) pipeline."""
        self._is_agentic = False
        self._config.enable_reflection = False
        self._config.enable_planning = False
        return self

    # =========================================================================
    # Advanced Features: Contextual Chunking & GraphRAG
    # =========================================================================

    def with_contextual_chunking(self, enabled: bool = True) -> Self:
        """
        Enable Anthropic's contextual chunking.

        Each chunk gets an LLM-generated context header explaining its place
        in the document. Reduces failed retrievals by 67%.

        Note: Adds latency during ingestion (1 LLM call per chunk).

        Args:
            enabled: Whether to enable contextual chunking.

        Returns:
            Self for chaining.
        """
        self._use_contextual_chunking = enabled
        return self

    def with_graphrag(self, enabled: bool = True, graph_path: str | None = None) -> Self:
        """
        Enable Microsoft's GraphRAG.

        Extracts entities and relationships from documents to build a
        knowledge graph. Enables:
        - Local search: Find specific entities and their neighborhoods
        - Global search: Answer questions about overall themes/patterns

        Best for: Research papers, legal documents, interconnected content.

        Note: Adds latency during ingestion (LLM calls for entity extraction).

        Args:
            enabled: Whether to enable GraphRAG.
            graph_path: Optional path to persist the graph.

        Returns:
            Self for chaining.
        """
        self._use_graphrag = enabled
        if enabled:
            self._graph_storage = NetworkXStorage()
        return self

    # =========================================================================
    # Caching Configuration
    # =========================================================================

    def with_cache(
        self,
        backend: Literal["memory", "disk", "redis"] = "memory",
        redis_url: str | None = None,
        redis_config: RedisConfig | None = None,
        cache_dir: str | None = None,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
    ) -> Self:
        """
        Configure semantic caching for query responses.

        Caches query-response pairs with semantic similarity matching.
        Similar queries (above threshold) return cached responses instantly.

        Backends:
        - memory: Fast, in-process (default, lost on restart)
        - disk: Persistent file-based cache
        - redis: Distributed cache for production (requires Redis server)

        Args:
            backend: Cache backend type.
            redis_url: Redis connection URL (redis://host:port/db).
            redis_config: Full Redis configuration (alternative to redis_url).
            cache_dir: Directory for disk cache.
            similarity_threshold: Min similarity for cache hit (0.0-1.0).
            ttl_seconds: Cache entry TTL in seconds.

        Returns:
            Self for chaining.

        Example:
            # Redis backend for production
            .with_cache(backend="redis", redis_url="redis://localhost:6379/0")

            # Disk cache for persistence
            .with_cache(backend="disk", cache_dir=".cache/rag")
        """
        self._cache_backend = backend
        # Store config for later initialization (needs embedder first)
        self._cache_config = {
            "backend": backend,
            "redis_url": redis_url,
            "redis_config": redis_config,
            "cache_dir": cache_dir,
            "similarity_threshold": similarity_threshold,
            "ttl_seconds": ttl_seconds,
        }
        return self

    # =========================================================================
    # Compression Configuration
    # =========================================================================

    def with_compression(
        self,
        method: Literal["extractive", "longllmlingua"] = "extractive",
        compression_ratio: float = 0.5,
        target_tokens: int | None = None,
        min_sentences: int = 3,
    ) -> Self:
        """
        Configure context compression to reduce token costs.

        Compression happens after retrieval/reranking, before generation.
        Typical savings: 50-70% tokens with minimal quality impact.

        Methods:
        - extractive: Fast, uses reranker to select top sentences
        - longllmlingua: Slower, uses LLM for importance scoring

        Args:
            method: Compression method.
            compression_ratio: Target ratio (0.5 = keep 50% of tokens).
            target_tokens: Absolute token target (overrides ratio).
            min_sentences: Minimum sentences to preserve.

        Returns:
            Self for chaining.

        Example:
            # Keep 30% of tokens (70% reduction)
            .with_compression(method="extractive", compression_ratio=0.3)

            # Limit to 2000 tokens
            .with_compression(target_tokens=2000)
        """
        self._compression_config = {
            "method": method,
            "compression_ratio": compression_ratio,
            "target_tokens": target_tokens,
            "min_sentences": min_sentences,
        }
        return self

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _create_chunking_generator(self) -> Any:
        """
        Create a dedicated fast generator for contextual chunking.

        Uses Groq by default for ultra-fast inference (~560 tok/s),
        falling back to other providers if Groq is not configured.

        Returns:
            Generator instance optimized for fast context header generation.
        """
        import logging

        logger = logging.getLogger("agentic_rag.pipeline")

        provider = self._settings.contextual_chunking_provider
        model = self._settings.contextual_chunking_model

        # Try to create the configured chunking generator
        if provider == "groq":
            try:
                from agentic_rag.generation.groq_generator import GroqGenerator

                groq_model = model or self._settings.groq_model or "llama-3.1-8b-instant"
                generator = GroqGenerator(
                    model=groq_model,
                    max_tokens=150,  # Short context headers
                    temperature=0.3,  # Consistent output
                    settings=self._settings,
                )
                logger.info(f"BUILD: Created Groq generator for chunking ({groq_model})")
                return generator
            except (ValueError, ImportError) as e:
                logger.warning(f"BUILD: Groq not available ({e}), falling back to main generator")

        elif provider == "gemini":
            try:
                from agentic_rag.generation.gemini_generator import GeminiGenerator

                gemini_model = model or "gemini-2.5-flash"  # Fast Gemini model
                generator = GeminiGenerator(
                    model=gemini_model,
                    max_tokens=150,
                    temperature=0.3,
                    settings=self._settings,
                )
                logger.info(f"BUILD: Created Gemini generator for chunking ({gemini_model})")
                return generator
            except (ValueError, ImportError) as e:
                logger.warning(f"BUILD: Gemini not available ({e}), falling back to main generator")

        elif provider == "openai":
            try:
                from agentic_rag.generation.openai_generator import OpenAIGenerator

                openai_model = model or "gpt-4o-mini"  # Fast OpenAI model
                generator = OpenAIGenerator(
                    model=openai_model,
                    max_tokens=150,
                    temperature=0.3,
                    settings=self._settings,
                )
                logger.info(f"BUILD: Created OpenAI generator for chunking ({openai_model})")
                return generator
            except (ValueError, ImportError) as e:
                logger.warning(f"BUILD: OpenAI not available ({e}), falling back to main generator")

        # Fallback to main generator
        logger.info("BUILD: Using main generator for contextual chunking")
        return self._generator

    # =========================================================================
    # Build
    # =========================================================================

    def build(self) -> "RAGPipeline":
        """
        Build the configured pipeline.

        Returns:
            Configured RAGPipeline instance.

        Raises:
            ValueError: If required components are not configured.
        """
        # Validate required components
        if self._embedder is None:
            # Use default embedder
            from agentic_rag.embeddings import create_embedder

            self._embedder = create_embedder("default", settings=self._settings)
            self._config.embedding_model = self._embedder.model_name
            self._config.embedding_dimension = self._embedder.dimension

        if self._vectordb is None:
            # Use default vectordb
            self._vectordb = QdrantVectorDB(settings=self._settings)

        if self._generator is None:
            # Use default generator
            self._generator = create_generator(settings=self._settings)

        self._is_built = True

        # Create chunker based on config
        chunker = None
        import logging

        logger = logging.getLogger("agentic_rag.pipeline")

        if self._config.chunk_strategy == "raptor" and self._embedder is not None:
            # Use RAPTOR Chunker (hierarchical tree with summaries)
            from agentic_rag.chunking.raptor import RAPTORChunker
            from agentic_rag.chunking.semantic import SemanticChunker

            base_chunker = SemanticChunker(
                embedder=self._embedder,
                chunk_size=self._config.chunk_size,
            )

            raptor_cfg = getattr(self, "_raptor_config", {})
            chunker = RAPTORChunker(
                embedder=self._embedder,
                generator=self._generator,
                base_chunker=base_chunker,
                max_levels=raptor_cfg.get("max_levels", 3),
                clustering_algorithm=raptor_cfg.get("clustering_algorithm", "gmm"),
            )
            logger.info(
                f"BUILD: Using RAPTORChunker (max_levels={raptor_cfg.get('max_levels', 3)})"
            )
        elif self._use_contextual_chunking and self._generator is not None:
            # Use Contextual Chunker (Anthropic's approach) with FAST Groq inference
            # Key: Use SentenceChunker as base (fast!) not SemanticChunker (slow embedding)
            from agentic_rag.chunking.base import SentenceChunker
            from agentic_rag.chunking.contextual import BatchContextualChunker

            # SentenceChunker is FAST - just splits by sentences, no embedding
            # SemanticChunker would embed every sentence which is very slow
            base_chunker = SentenceChunker(
                chunk_size=self._config.chunk_size,
                chunk_overlap=1,  # 1 sentence overlap
            )

            # Use dedicated fast generator for contextual chunking (Groq by default)
            chunking_generator = self._create_chunking_generator()

            chunker = BatchContextualChunker(
                generator=chunking_generator,
                base_chunker=base_chunker,
                chunk_size=self._config.chunk_size,
            )
            chunking_provider = self._settings.contextual_chunking_provider
            logger.info(
                f"BUILD: Using ContextualChunker with {chunking_provider.upper()} (-67% failed retrievals, FAST)"
            )
        elif self._config.chunk_strategy == "recursive":
            # RecursiveChunker: Fast but basic (no semantic boundaries)
            # Use for: quick prototyping, CPU-only environments, large batch processing
            from agentic_rag.chunking.recursive import RecursiveChunker

            chunker = RecursiveChunker(
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )
            logger.info("BUILD: Using RecursiveChunker (fast, basic boundaries)")
        elif self._config.chunk_strategy == "semantic" and self._embedder is not None:
            # SemanticChunker: +70% accuracy over fixed-size (RAG.md research)
            # Trade-off: embeds every sentence to find intelligent breakpoints
            # This is what makes Agentic RAG better than basic RAG!
            from agentic_rag.chunking.semantic import SemanticChunker

            chunker = SemanticChunker(
                embedder=self._embedder,
                chunk_size=self._config.chunk_size,
            )
            logger.info("BUILD: Using SemanticChunker (+70% accuracy, intelligent breakpoints)")

        # Create entity extractor for GraphRAG
        entity_extractor = None
        if self._use_graphrag and self._generator is not None:
            entity_extractor = LLMEntityExtractor(generator=self._generator)
            logger.info("BUILD: GraphRAG enabled - entity extraction active")

        # Create reranker - ColBERT for high-quality reranking
        reranker = None
        try:
            import logging

            from agentic_rag.reranking.colbert import ColBERTReranker

            logger = logging.getLogger("agentic_rag.pipeline")
            logger.info("BUILD: Initializing ColBERT reranker (jinaai/jina-colbert-v2)...")
            reranker = ColBERTReranker()
            logger.info("BUILD: ColBERT reranker loaded successfully")
        except Exception as e:
            import logging

            logging.getLogger("agentic_rag.pipeline").warning(
                f"BUILD: ColBERT reranker unavailable: {e}"
            )

        # Create semantic cache based on configuration
        cache: SemanticCache | None = None
        if hasattr(self, "_cache_config") and self._cache_config:
            cfg = self._cache_config
            backend = cfg["backend"]
            threshold = cfg["similarity_threshold"]
            ttl = cfg["ttl_seconds"]

            if backend == "redis":
                # Redis cache for distributed/production deployments
                cache = RedisSemanticCache(
                    embedder=self._embedder,
                    redis_url=cfg.get("redis_url"),
                    redis_config=cfg.get("redis_config"),
                    similarity_threshold=threshold,
                    ttl_seconds=ttl,
                    settings=self._settings,
                )
                logger.info("BUILD: Redis semantic cache configured")
            elif backend == "disk":
                # Disk cache for persistence
                from pathlib import Path

                cache_dir = Path(cfg.get("cache_dir") or ".cache/agentic_rag")
                cache = DiskSemanticCache(
                    embedder=self._embedder,
                    cache_dir=cache_dir,
                    similarity_threshold=threshold,
                    ttl_seconds=ttl,
                    settings=self._settings,
                )
                logger.info(f"BUILD: Disk semantic cache at {cache_dir}")
            else:
                # Default: in-memory cache
                cache = SemanticCache(
                    embedder=self._embedder,
                    similarity_threshold=threshold,
                    ttl_seconds=ttl,
                    settings=self._settings,
                )
                logger.info("BUILD: In-memory semantic cache configured")

        # Create compressor based on configuration
        compressor: BaseCompressor | None = None
        if self._compression_config:
            cfg = self._compression_config
            method = cfg["method"]
            ratio = cfg["compression_ratio"]
            target = cfg.get("target_tokens")
            min_sent = cfg["min_sentences"]

            if method == "longllmlingua":
                # LongLLMLingua uses LLM for scoring
                compressor = LongLLMLinguaCompressor(
                    generator=self._generator,
                    target_tokens=target,
                    compression_ratio=ratio,
                    min_sentences=min_sent,
                )
                logger.info(f"BUILD: LongLLMLingua compressor configured (ratio={ratio})")
            else:
                # Extractive uses reranker (faster)
                if reranker is not None:
                    compressor = ExtractiveCompressor(
                        reranker=reranker,
                        target_tokens=target,
                        compression_ratio=ratio,
                        min_sentences=min_sent,
                    )
                    logger.info(f"BUILD: Extractive compressor configured (ratio={ratio})")
                else:
                    logger.warning("BUILD: No reranker available for extractive compression")

        return RAGPipeline(
            config=self._config,
            embedder=self._embedder,
            vectordb=self._vectordb,
            generator=self._generator,
            chunker=chunker,
            reranker=reranker,
            is_agentic=self._is_agentic,
            settings=self._settings,
            # Advanced features
            graph_storage=self._graph_storage,
            entity_extractor=entity_extractor,
            use_contextual_chunking=self._use_contextual_chunking,
            use_graphrag=self._use_graphrag,
            use_multi_query=self._use_multi_query,
            num_queries=self._num_queries,
            # Caching
            cache=cache,
            # Compression
            compressor=compressor,
        )

    @property
    def config(self) -> RAGConfig:
        """Get the current configuration."""
        return self._config


class RAGPipeline:
    """
    The main RAG pipeline that orchestrates all components.

    Created via PipelineBuilder.build().
    """

    def __init__(
        self,
        config: RAGConfig,
        embedder: Qwen3Embedder,
        vectordb: QdrantVectorDB,
        generator: Any,
        chunker: Any = None,
        reranker: Any = None,
        is_agentic: bool = False,
        settings: Settings | None = None,
        # Advanced features
        graph_storage: GraphStorage | None = None,
        entity_extractor: Any = None,
        use_contextual_chunking: bool = False,
        use_graphrag: bool = False,
        use_multi_query: bool = True,
        num_queries: int = 4,
        # Caching
        cache: SemanticCache | None = None,
        # Compression
        compressor: BaseCompressor | None = None,
    ):
        """Initialize the pipeline with configured components."""
        self.config = config
        self.embedder = embedder
        self.vectordb = vectordb
        self.generator = generator
        self.chunker = chunker
        self.reranker = reranker
        self.is_agentic = is_agentic
        self._settings = settings or get_settings()

        # Advanced features
        self.graph_storage = graph_storage
        self.entity_extractor = entity_extractor
        self.use_contextual_chunking = use_contextual_chunking
        self.use_graphrag = use_graphrag
        self.use_multi_query = use_multi_query
        self.num_queries = num_queries

        # Caching
        self.cache = cache

        # Compression
        self.compressor = compressor

    async def query(
        self,
        question: str,
        collection: str,
        **kwargs: Any,
    ) -> Any:  # Returns GenerationResult
        """
        Execute a RAG query.

        Args:
            question: The user's question.
            collection: Vector DB collection to search.
            **kwargs: Additional query parameters:
                top_k: Override default retrieval count.
                temperature: Override generation temperature.
                max_tokens: Override max output tokens.

        Returns:
            GenerationResult with response and sources.
        """

        if self.is_agentic:
            # Use orchestrator for agentic pipeline
            return await self._agentic_query(question, collection, **kwargs)

        # Standard pipeline
        return await self._standard_query(question, collection, **kwargs)

    async def _standard_query(
        self,
        question: str,
        collection: str,
        **kwargs: Any,
    ) -> Any:
        """Standard retrieve-then-generate pipeline with full tracking."""
        import logging
        import time

        logger = logging.getLogger("agentic_rag.pipeline")

        from agentic_rag.core.models import GenerationResult
        from agentic_rag.retrieval import HybridRetriever, HyDERetriever
        from agentic_rag.retrieval.multi_query import MultiQueryRetriever

        # Track pipeline steps
        pipeline_steps = []
        query_variations = [question]  # Original query

        # Check cache first (if enabled)
        if self.cache is not None:
            step_start = time.time()
            try:
                # Connect if Redis cache
                if hasattr(self.cache, "connect") and hasattr(self.cache, "is_connected"):
                    if not self.cache.is_connected:
                        await self.cache.connect()

                cached = await self.cache.get(question)
                cache_time = (time.time() - step_start) * 1000

                if cached is not None:
                    logger.info(
                        f"QUERY: Cache HIT - returning cached response ({cache_time:.0f}ms)"
                    )
                    pipeline_steps.append(
                        {
                            "name": "Semantic Cache",
                            "duration_ms": round(cache_time, 1),
                            "details": {
                                "hit": True,
                                "similarity": getattr(cached, "similarity", 0.95),
                            },
                        }
                    )

                    # Return cached response
                    return GenerationResult(
                        response=cached.response,
                        sources=[],  # Original sources not stored in cache entry
                        model=getattr(self.generator, "model", "unknown"),
                        provider=self.generator.provider,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=cache_time,
                        metadata={
                            "cache_hit": True,
                            "pipeline_steps": pipeline_steps,
                            "cached_query": cached.query,
                        },
                    )
                else:
                    logger.info(
                        f"QUERY: Cache MISS - proceeding with retrieval ({cache_time:.0f}ms)"
                    )
                    pipeline_steps.append(
                        {
                            "name": "Semantic Cache",
                            "duration_ms": round(cache_time, 1),
                            "details": {"hit": False},
                        }
                    )
            except Exception as e:
                logger.warning(f"QUERY: Cache check failed: {e}")

        # Create base retriever (hybrid)
        base_retriever = HybridRetriever(
            embedder=self.embedder,
            vectordb=self.vectordb,
            settings=self._settings,
        )

        # Wrap with Multi-Query or HyDE based on config
        step_start = time.time()
        if self.use_multi_query:
            logger.info(
                f"QUERY: Using Multi-Query retrieval ({self.num_queries} query variations)..."
            )
            retriever = MultiQueryRetriever(
                base_retriever=base_retriever,
                generator=self.generator,
                num_queries=self.num_queries,
                settings=self._settings,
            )
        elif self.config.use_hyde:
            retriever = HyDERetriever(
                generator=self.generator,
                embedder=self.embedder,
                vectordb=self.vectordb,
                settings=self._settings,
            )
        else:
            retriever = base_retriever

        # Check if reranking is enabled (from kwargs or pipeline config)
        use_reranking = kwargs.get("use_reranking")
        if use_reranking is None:
            use_reranking = self.reranker is not None
        should_rerank = use_reranking and self.reranker is not None

        # Retrieve with extra candidates for reranking
        top_k = kwargs.get("top_k", self.config.top_k)
        retrieve_k = top_k * 3 if should_rerank else top_k  # Get more candidates for reranking

        retrieval_result = await retriever.retrieve(
            query=question,
            collection=collection,
            top_k=retrieve_k,
        )
        chunks = retrieval_result.chunks

        # Extract query variations from multi-query metadata
        if hasattr(retrieval_result, "metadata") and retrieval_result.metadata:
            query_variations = retrieval_result.metadata.get("queries", [question])

        retrieval_time = (time.time() - step_start) * 1000
        pipeline_steps.append(
            {
                "name": "Multi-Query Retrieval"
                if self.use_multi_query
                else ("HyDE Retrieval" if self.config.use_hyde else "Hybrid Retrieval"),
                "duration_ms": round(retrieval_time, 1),
                "details": {
                    "query_variations": query_variations,
                    "chunks_retrieved": len(chunks),
                },
            }
        )
        logger.info(f"QUERY: Retrieved {len(chunks)} chunks in {retrieval_time:.0f}ms")

        # Rerank if enabled and reranker is available
        if should_rerank and len(chunks) > top_k:
            try:
                step_start = time.time()
                logger.info(f"QUERY: Reranking {len(chunks)} chunks with ColBERT...")

                # Rerank chunks - returns RerankResult with .chunks and .scores
                result = await self.reranker.rerank(
                    query=question,
                    chunks=chunks,
                    top_k=top_k,
                )
                rerank_time = (time.time() - step_start) * 1000

                # Store reranking scores in chunk metadata for downstream use
                for chunk, score in zip(result.chunks, result.scores, strict=False):
                    chunk.metadata = chunk.metadata or {}
                    chunk.metadata["rerank_score"] = score

                chunks = result.chunks

                pipeline_steps.append(
                    {
                        "name": "ColBERT Reranking",
                        "duration_ms": round(rerank_time, 1),
                        "details": {
                            "input_chunks": len(retrieval_result.chunks),
                            "output_chunks": len(chunks),
                        },
                    }
                )
                logger.info(f"QUERY: Reranked to top {len(chunks)} chunks in {rerank_time:.0f}ms")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
                # Fall back to original chunks
                chunks = chunks[:top_k]
        else:
            chunks = chunks[:top_k]

        # GraphRAG: Enrich context with graph knowledge
        graph_context = ""
        if self.use_graphrag and self.graph_storage:
            try:
                step_start = time.time()
                from agentic_rag.graph.retriever import GraphRAGRetriever

                graph_retriever = GraphRAGRetriever(
                    storage=self.graph_storage,
                    generator=self.generator,
                    embedder=self.embedder,
                )

                graph_result = await graph_retriever.retrieve(question, top_k=5)
                graph_time = (time.time() - step_start) * 1000

                if graph_result.context:
                    graph_context = f"\n\n[Knowledge Graph Context]\n{graph_result.context}"
                    pipeline_steps.append(
                        {
                            "name": "GraphRAG Enrichment",
                            "duration_ms": round(graph_time, 1),
                            "details": {
                                "entities": len(graph_result.entities),
                                "relationships": len(graph_result.relationships),
                            },
                        }
                    )
                    logger.info(
                        f"QUERY: GraphRAG added {len(graph_result.entities)} entities in {graph_time:.0f}ms"
                    )

            except Exception as e:
                logger.warning(f"QUERY: GraphRAG retrieval failed: {e}")

        # Context compression (if enabled)
        if self.compressor is not None and chunks:
            try:
                step_start = time.time()
                len(chunks)
                sum(len(c.content) // 4 for c in chunks)

                compression_result = await self.compressor.compress(
                    query=question,
                    chunks=chunks,
                )

                chunks = compression_result.compressed_chunks
                compression_time = (time.time() - step_start) * 1000

                pipeline_steps.append(
                    {
                        "name": "Context Compression",
                        "duration_ms": round(compression_time, 1),
                        "details": {
                            "original_tokens": compression_result.original_tokens,
                            "compressed_tokens": compression_result.compressed_tokens,
                            "reduction_percent": round(compression_result.savings_percent, 1),
                            "chunks_kept": len(chunks),
                        },
                    }
                )
                logger.info(
                    f"QUERY: Compressed context {compression_result.original_tokens} -> "
                    f"{compression_result.compressed_tokens} tokens "
                    f"({compression_result.savings_percent:.1f}% reduction) in {compression_time:.0f}ms"
                )
            except Exception as e:
                logger.warning(f"QUERY: Compression failed: {e}")

        # Generate intermediate answers for each query variation (if multi-query)
        intermediate_answers = []
        if self.use_multi_query and len(query_variations) > 1:
            step_start = time.time()
            logger.info(
                f"QUERY: Generating intermediate answers for {len(query_variations)} query variations..."
            )

            import asyncio

            async def generate_intermediate(q: str, idx: int) -> dict:
                """Generate answer for a single query variation."""
                try:
                    # Use fewer chunks for intermediate answers (faster)
                    intermediate_result = await self.generator.generate(
                        query=q,
                        context=chunks[:3],  # Use top 3 chunks for speed
                    )
                    return {
                        "query": q,
                        "answer": intermediate_result.response[:500],  # Truncate for display
                        "index": idx,
                    }
                except Exception as e:
                    return {"query": q, "answer": f"Error: {e}", "index": idx}

            # Generate all intermediate answers in parallel
            tasks = [generate_intermediate(q, i) for i, q in enumerate(query_variations)]
            intermediate_answers = await asyncio.gather(*tasks)
            intermediate_answers = sorted(intermediate_answers, key=lambda x: x["index"])

            intermediate_time = (time.time() - step_start) * 1000
            pipeline_steps.append(
                {
                    "name": "Intermediate Answers",
                    "duration_ms": round(intermediate_time, 1),
                    "details": {
                        "num_answers": len(intermediate_answers),
                    },
                }
            )
            logger.info(
                f"QUERY: Generated {len(intermediate_answers)} intermediate answers in {intermediate_time:.0f}ms"
            )

        # Generate final response with full context
        step_start = time.time()
        result = await self.generator.generate(
            query=question,
            context=chunks,
            additional_context=graph_context if graph_context else None,
        )
        generation_time = (time.time() - step_start) * 1000

        pipeline_steps.append(
            {
                "name": "Final LLM Generation",
                "duration_ms": round(generation_time, 1),
                "details": {
                    "provider": self.generator.provider,
                    "context_chunks": len(chunks),
                },
            }
        )
        logger.info(f"QUERY: Generated final response in {generation_time:.0f}ms")

        # Add pipeline metadata to result
        result.metadata = result.metadata or {}
        result.metadata["pipeline_steps"] = pipeline_steps
        result.metadata["query_variations"] = query_variations
        result.metadata["intermediate_answers"] = intermediate_answers

        # Cache the result (if caching enabled)
        if self.cache is not None:
            try:
                # Convert chunks to serializable format for cache
                sources_for_cache = [
                    {"content": c.content[:500], "document_id": c.document_id}
                    for c in chunks[:5]  # Store top 5 chunks summary
                ]
                await self.cache.set(
                    query=question,
                    response=result.response,
                    sources=sources_for_cache,
                    metadata={"model": result.model, "provider": result.provider},
                )
                logger.debug("QUERY: Response cached for future queries")
            except Exception as e:
                logger.warning(f"QUERY: Failed to cache response: {e}")

        return result

    async def _agentic_query(
        self,
        question: str,
        collection: str,
        **kwargs: Any,
    ) -> Any:
        """Agentic pipeline with multi-agent orchestration."""
        from agentic_rag.agents import AgentState, OrchestratorAgent

        # Create orchestrator
        orchestrator = OrchestratorAgent(
            generator=self.generator,
            embedder=self.embedder,
            vectordb=self.vectordb,
            settings=self._settings,
            max_iterations=self.config.max_iterations,
        )

        # Execute agentic workflow
        state = AgentState(
            query=question,
            context={"collection": collection},
        )

        result = await orchestrator.execute(state)

        # Convert to GenerationResult
        from agentic_rag.core.models import GenerationResult

        return GenerationResult(
            response=result.response,
            sources=result.sources,
            model=getattr(self.generator, "model", "unknown"),
            provider=self.generator.provider,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=0,
            metadata={"workflow_history": result.workflow_history},
        )

    async def ingest(
        self,
        documents: list[Any],  # list[Document]
        collection: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Ingest documents into the vector database.

        Args:
            documents: Documents to ingest.
            collection: Target collection.
            **kwargs: Additional ingestion parameters.

        Returns:
            Ingestion statistics.
        """
        import logging
        import time

        logger = logging.getLogger("agentic_rag.pipeline")

        from agentic_rag.core.models import Chunk

        logger.info(f"INGEST: Starting ingestion of {len(documents)} documents")
        total_start = time.time()

        # Ensure collection exists
        step_start = time.time()
        exists = await self.vectordb.collection_exists(collection)
        if not exists:
            await self.vectordb.create_collection(
                name=collection,
                dimension=self.embedder.dimension,
            )
        logger.info(f"INGEST: Collection check/create in {time.time() - step_start:.3f}s")

        # Process documents into chunks
        step_start = time.time()
        all_chunks: list[Chunk] = []

        if self.chunker is not None:
            chunker_name = type(self.chunker).__name__
            logger.info(f"INGEST: Using {chunker_name} for chunking...")
            for doc in documents:
                # Ensure we have a proper Document object
                if hasattr(doc, "content"):
                    # Support both chunk() and chunk_async() methods
                    if hasattr(self.chunker, "chunk_async"):
                        doc_chunks = await self.chunker.chunk_async(doc)
                    elif hasattr(self.chunker, "chunk"):
                        doc_chunks = await self.chunker.chunk(doc)
                    else:
                        doc_chunks = []
                    # Propagate document metadata to chunks (for caching, filtering, etc.)
                    for chunk in doc_chunks:
                        chunk.metadata.update(doc.metadata)
                    all_chunks.extend(doc_chunks)
            logger.info(
                f"INGEST: {chunker_name} created {len(all_chunks)} chunks from {len(documents)} docs in {time.time() - step_start:.3f}s"
            )
        else:
            # Fallback to simple chunking
            logger.info("INGEST: Using fallback character-based chunking...")
            MAX_CHUNK_SIZE = 1500
            OVERLAP = 150

            for doc in documents:
                content = doc.content
                doc_len = len(content)

                if doc_len <= MAX_CHUNK_SIZE:
                    chunk = Chunk(
                        content=content,
                        document_id=doc.id,
                        metadata={**doc.metadata, "chunk_index": 0, "total_chunks": 1},
                    )
                    all_chunks.append(chunk)
                else:
                    chunk_index = 0
                    start = 0
                    while start < doc_len:
                        end = min(start + MAX_CHUNK_SIZE, doc_len)
                        if end < doc_len:
                            for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                                last_sep = content[max(start, end - 150) : end].rfind(sep)
                                if last_sep != -1:
                                    end = max(start, end - 150) + last_sep + len(sep)
                                    break

                        chunk_content = content[start:end].strip()
                        if chunk_content:
                            chunk = Chunk(
                                content=chunk_content,
                                document_id=doc.id,
                                metadata={**doc.metadata, "chunk_index": chunk_index},
                            )
                            all_chunks.append(chunk)
                            chunk_index += 1

                        start = end - OVERLAP if end < doc_len else doc_len

                    for chunk in all_chunks[-chunk_index:]:
                        chunk.metadata["total_chunks"] = chunk_index

            logger.info(
                f"INGEST: Created {len(all_chunks)} chunks from {len(documents)} docs in {time.time() - step_start:.3f}s"
            )

        # Embed chunks
        step_start = time.time()
        texts = [c.content for c in all_chunks]
        total_chars = sum(len(t) for t in texts)
        logger.info(f"INGEST: Embedding {len(texts)} texts ({total_chars} total chars)...")
        embeddings = await self.embedder.embed_batch(texts)
        logger.info(f"INGEST: Embedding completed in {time.time() - step_start:.2f}s")

        for chunk, embedding in zip(all_chunks, embeddings, strict=False):
            chunk.embedding = embedding

        # Upsert to vector DB
        step_start = time.time()
        logger.info(f"INGEST: Upserting {len(all_chunks)} chunks to Qdrant...")
        await self.vectordb.upsert(collection, all_chunks)
        logger.info(f"INGEST: Upsert completed in {time.time() - step_start:.2f}s")

        # GraphRAG: Extract entities and relationships
        entities_count = 0
        relationships_count = 0
        if self.use_graphrag and self.entity_extractor and self.graph_storage:
            step_start = time.time()
            logger.info(f"INGEST: GraphRAG - Extracting entities from {len(all_chunks)} chunks...")

            import asyncio

            from agentic_rag.graph.extractor import merge_entities, merge_relationships

            all_entities: list[Entity] = []
            all_relationships: list[Relationship] = []

            # Process in concurrent batches to speed up extraction
            BATCH_SIZE = 10  # Process 10 chunks concurrently (respect rate limits)

            for batch_start in range(0, len(all_chunks), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(all_chunks))
                batch = all_chunks[batch_start:batch_end]

                logger.info(
                    f"INGEST: GraphRAG - Processing chunks {batch_start + 1}-{batch_end}/{len(all_chunks)} (concurrent)..."
                )

                # Run extractions concurrently
                tasks = [self.entity_extractor.extract(chunk) for chunk in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"INGEST: GraphRAG - Extraction failed: {result}")
                        continue
                    all_entities.extend(result.entities)
                    all_relationships.extend(result.relationships)

            # Merge duplicates
            merged_entities = merge_entities(all_entities)
            merged_relationships = merge_relationships(all_relationships)

            # Store in graph
            for entity in merged_entities:
                self.graph_storage.add_entity(entity)
            for rel in merged_relationships:
                self.graph_storage.add_relationship(rel)

            entities_count = len(merged_entities)
            relationships_count = len(merged_relationships)

            logger.info(
                f"INGEST: GraphRAG - Extracted {entities_count} entities, {relationships_count} relationships in {time.time() - step_start:.2f}s"
            )

            # Detect communities
            try:
                from agentic_rag.graph.community import (
                    CommunitySummarizer,
                    LeidenCommunityDetector,
                )

                detector = LeidenCommunityDetector()
                hierarchy = detector.detect(merged_entities, merged_relationships)

                # Generate summaries for communities
                if hierarchy.communities:
                    summarizer = CommunitySummarizer(generator=self.generator)
                    hierarchy = await summarizer.summarize_hierarchy(
                        hierarchy, merged_entities, merged_relationships
                    )
                    self.graph_storage.set_communities(hierarchy)
                    logger.info(
                        f"INGEST: GraphRAG - Detected {len(hierarchy.communities)} communities"
                    )
            except Exception as e:
                logger.warning(f"INGEST: GraphRAG - Community detection failed: {e}")

        logger.info(f"INGEST: Total ingestion time: {time.time() - total_start:.2f}s")

        return {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "collection": collection,
            "entities": entities_count,
            "relationships": relationships_count,
        }

    async def close(self) -> None:
        """Close all connections."""
        await self.vectordb.close()

        # Close cache connection if Redis
        if self.cache is not None and hasattr(self.cache, "close"):
            await self.cache.close()
