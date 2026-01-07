"""
Agentic RAG pipeline.

Multi-agent orchestrated pipeline with reflection, planning, and
self-correction capabilities.
"""

from typing import Any

from agentic_rag.agents import (
    EvaluatorAgent,
    GeneratorAgent,
    OrchestratorAgent,
    RetrieverAgent,
    RouterAgent,
)
from agentic_rag.chunking import BaseChunker, ContextualChunker
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, Document
from agentic_rag.embeddings import BaseEmbedder, Qwen3Embedder
from agentic_rag.generation import BaseGenerator, GeneratorFactory
from agentic_rag.pipeline.base import BasePipeline, IngestResult, PipelineResult
from agentic_rag.reranking import BaseReranker, LostInMiddleReorderer
from agentic_rag.retrieval import BaseRetriever, HybridRetriever
from agentic_rag.vectordb import QdrantVectorDB


class AgenticPipeline(BasePipeline):
    """
    Agentic RAG pipeline with multi-agent orchestration.

    Features:
    - RouterAgent: Classifies query intent and selects retrieval strategy
    - RetrieverAgent: CRAG-style confidence-based retrieval
    - EvaluatorAgent: Self-RAG reflection tokens (ISREL, ISSUP, ISUSE)
    - GeneratorAgent: Context-aware response generation
    - OrchestratorAgent: Coordinates all agents with planning

    Flow:
    1. Route query → classify intent, plan approach
    2. Retrieve → execute with confidence assessment
    3. Evaluate → check relevance, support, usefulness
    4. Generate → produce response
    5. Reflect → iterate if confidence < threshold
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        retriever: BaseRetriever | None = None,
        reranker: BaseReranker | None = None,
        generator: BaseGenerator | None = None,
        chunker: BaseChunker | None = None,
        vectordb: QdrantVectorDB | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize agentic pipeline.

        Args:
            embedder: Embedding model.
            retriever: Base retriever.
            reranker: Reranker (wrapped with lost-in-middle fix).
            generator: LLM generator.
            chunker: Document chunker.
            vectordb: Vector database client.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()

        # Initialize core components
        self._embedder = embedder or Qwen3Embedder(settings=self._settings)
        self._vectordb = vectordb or QdrantVectorDB(settings=self._settings)
        self._base_retriever = retriever or HybridRetriever(
            embedder=self._embedder,
            vectordb=self._vectordb,
            settings=self._settings,
        )

        # Wrap reranker with lost-in-middle fix
        if reranker:
            self._reranker = LostInMiddleReorderer(base_reranker=reranker)
        else:
            self._reranker = None

        self._generator = generator or GeneratorFactory.create(settings=self._settings)
        self._chunker = chunker or ContextualChunker(
            embedder=self._embedder,
            generator=self._generator,
            settings=self._settings,
        )

        # Initialize agents
        self._router = RouterAgent(generator=self._generator, settings=self._settings)
        self._retriever_agent = RetrieverAgent(
            retriever=self._base_retriever,
            generator=self._generator,
            settings=self._settings,
        )
        self._evaluator = EvaluatorAgent(
            generator=self._generator,
            settings=self._settings,
        )
        self._generator_agent = GeneratorAgent(
            generator=self._generator,
            settings=self._settings,
        )
        self._orchestrator = OrchestratorAgent(
            router=self._router,
            retriever=self._retriever_agent,
            evaluator=self._evaluator,
            generator=self._generator_agent,
            settings=self._settings,
        )

    async def query(
        self,
        question: str,
        collection: str,
        max_iterations: int | None = None,
        enable_reflection: bool | None = None,
        enable_planning: bool | None = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        Query the agentic pipeline.

        Args:
            question: User question.
            collection: Collection to search.
            max_iterations: Max self-correction iterations.
            enable_reflection: Enable Self-RAG reflection.
            enable_planning: Enable query planning.
            **kwargs: Additional parameters.

        Returns:
            PipelineResult with response.
        """
        max_iterations = max_iterations or self._settings.max_iterations
        enable_reflection = (
            enable_reflection if enable_reflection is not None else self._settings.enable_reflection
        )
        enable_planning = (
            enable_planning if enable_planning is not None else self._settings.enable_planning
        )

        # Run orchestrator
        result = await self._orchestrator.run(
            query=question,
            collection=collection,
            max_iterations=max_iterations,
            enable_reflection=enable_reflection,
            enable_planning=enable_planning,
            reranker=self._reranker,
            **kwargs,
        )

        return PipelineResult(
            response=result.response,
            sources=result.sources,
            confidence=result.confidence,
            provider=self._generator.provider,
            model=self._generator.model_name,
            input_tokens=result.metadata.get("input_tokens", 0),
            output_tokens=result.metadata.get("output_tokens", 0),
            total_tokens=result.metadata.get("total_tokens", 0),
            latency_ms=result.metadata.get("latency_ms", 0),
            metadata={
                "iterations": result.metadata.get("iterations", 1),
                "query_type": result.metadata.get("query_type", "unknown"),
                "retrieval_strategy": result.metadata.get("retrieval_strategy", "hybrid"),
                "reflection_tokens": result.metadata.get("reflection_tokens", {}),
                "planning_steps": result.metadata.get("planning_steps", []),
            },
        )

    async def ingest(
        self,
        documents: list[Document],
        collection: str,
        use_contextual: bool = True,
        **kwargs: Any,
    ) -> IngestResult:
        """
        Ingest documents with contextual chunking.

        Args:
            documents: Documents to ingest.
            collection: Target collection.
            use_contextual: Use contextual headers (Anthropic technique).
            **kwargs: Additional parameters.

        Returns:
            IngestResult with statistics.
        """
        all_chunks: list[Chunk] = []

        # Chunk each document (with context headers if enabled)
        for doc in documents:
            chunks = await self._chunker.chunk(doc, use_contextual=use_contextual)
            all_chunks.extend(chunks)

        # Generate embeddings
        texts = [chunk.content for chunk in all_chunks]
        embeddings = await self._embedder.embed_batch(texts)

        for chunk, embedding in zip(all_chunks, embeddings, strict=False):
            chunk.embedding = embedding

        # Store in vector DB
        await self._vectordb.upsert(
            collection=collection,
            chunks=all_chunks,
        )

        return IngestResult(
            documents=len(documents),
            chunks=len(all_chunks),
            collection=collection,
            metadata={
                "contextual": use_contextual,
                "chunker": self._chunker.__class__.__name__,
            },
        )

    async def close(self) -> None:
        """Close pipeline resources."""
        if hasattr(self._vectordb, "close"):
            await self._vectordb.close()
