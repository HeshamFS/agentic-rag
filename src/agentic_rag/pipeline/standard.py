"""
Standard RAG pipeline.

Simple retrieve-then-generate pipeline without agentic features.
Good baseline for comparison and simple use cases.
"""

from typing import Any

from agentic_rag.chunking import BaseChunker, SemanticChunker
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, Document
from agentic_rag.embeddings import BaseEmbedder, Qwen3Embedder
from agentic_rag.generation import BaseGenerator, GeneratorFactory
from agentic_rag.pipeline.base import BasePipeline, IngestResult, PipelineResult
from agentic_rag.reranking import BaseReranker
from agentic_rag.retrieval import BaseRetriever, HybridRetriever
from agentic_rag.vectordb import QdrantVectorDB


class StandardPipeline(BasePipeline):
    """
    Standard retrieve-then-generate RAG pipeline.

    Flow:
    1. Retrieve relevant chunks from vector DB
    2. Optionally rerank results
    3. Generate response using LLM

    No reflection, planning, or iterative refinement.
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
        Initialize standard pipeline.

        Args:
            embedder: Embedding model. Defaults to Qwen3Embedder.
            retriever: Retriever. Defaults to HybridRetriever.
            reranker: Optional reranker for result refinement.
            generator: LLM generator. Defaults to settings provider.
            chunker: Document chunker. Defaults to SemanticChunker.
            vectordb: Vector database client.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()

        # Initialize components
        self._embedder = embedder or Qwen3Embedder(settings=self._settings)
        self._vectordb = vectordb or QdrantVectorDB(settings=self._settings)
        self._retriever = retriever or HybridRetriever(
            embedder=self._embedder,
            vectordb=self._vectordb,
            settings=self._settings,
        )
        self._reranker = reranker
        self._generator = generator or GeneratorFactory.create(settings=self._settings)
        self._chunker = chunker or SemanticChunker(
            embedder=self._embedder,
            settings=self._settings,
        )

    async def query(
        self,
        question: str,
        collection: str,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        Query the standard pipeline.

        Args:
            question: User question.
            collection: Collection to search.
            top_k: Number of chunks to retrieve.
            rerank_top_k: Number of chunks after reranking.
            **kwargs: Additional parameters.

        Returns:
            PipelineResult with response.
        """
        top_k = top_k or self._settings.default_top_k
        rerank_top_k = rerank_top_k or self._settings.default_rerank_top_k

        # Step 1: Retrieve
        retrieval_result = await self._retriever.retrieve(
            query=question,
            collection=collection,
            top_k=top_k,
        )
        chunks = retrieval_result.chunks

        # Step 2: Rerank (optional)
        if self._reranker and chunks:
            rerank_result = await self._reranker.rerank(
                query=question,
                chunks=chunks,
                top_k=rerank_top_k,
            )
            chunks = rerank_result.chunks

        # Step 3: Generate
        generation_result = await self._generator.generate(
            query=question,
            context=chunks,
            **kwargs,
        )

        return PipelineResult(
            response=generation_result.response,
            sources=chunks,
            confidence=generation_result.confidence,
            provider=generation_result.provider,
            model=generation_result.model,
            input_tokens=generation_result.input_tokens,
            output_tokens=generation_result.output_tokens,
            total_tokens=generation_result.total_tokens,
            latency_ms=generation_result.latency_ms,
            metadata={
                "retrieval_type": retrieval_result.retrieval_type,
                "chunks_retrieved": len(retrieval_result.chunks),
                "chunks_used": len(chunks),
            },
        )

    async def ingest(
        self,
        documents: list[Document],
        collection: str,
        **kwargs: Any,
    ) -> IngestResult:
        """
        Ingest documents into the pipeline.

        Args:
            documents: Documents to ingest.
            collection: Target collection.
            **kwargs: Chunking parameters.

        Returns:
            IngestResult with statistics.
        """
        all_chunks: list[Chunk] = []

        # Chunk each document
        for doc in documents:
            chunks = await self._chunker.chunk(doc)
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
        )

    async def close(self) -> None:
        """Close pipeline resources."""
        if hasattr(self._vectordb, "close"):
            await self._vectordb.close()
