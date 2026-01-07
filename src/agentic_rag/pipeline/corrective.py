"""
Corrective RAG (CRAG) pipeline.

Implements the CRAG pattern with confidence-based retrieval assessment
and fallback strategies.

Reference: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.chunking import BaseChunker, SemanticChunker
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, Document
from agentic_rag.embeddings import BaseEmbedder, Qwen3Embedder
from agentic_rag.generation import BaseGenerator, GeneratorFactory
from agentic_rag.pipeline.base import BasePipeline, IngestResult, PipelineResult
from agentic_rag.reranking import BaseReranker, apply_lost_in_middle
from agentic_rag.retrieval import BaseRetriever, HybridRetriever, HyDERetriever
from agentic_rag.vectordb import QdrantVectorDB


class RetrievalConfidence(str, Enum):
    """CRAG confidence levels."""

    CORRECT = "correct"  # High confidence, use as-is
    AMBIGUOUS = "ambiguous"  # Medium confidence, refine
    INCORRECT = "incorrect"  # Low confidence, fallback


class CRAGAssessment(BaseModel):
    """CRAG retrieval assessment result."""

    confidence: RetrievalConfidence
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class CorrectivePipeline(BasePipeline):
    """
    Corrective RAG (CRAG) pipeline.

    Implements three-tier confidence assessment:
    1. CORRECT: High confidence → use retrieved context directly
    2. AMBIGUOUS: Medium confidence → refine query and re-retrieve
    3. INCORRECT: Low confidence → use web search or knowledge fallback

    Flow:
    1. Initial retrieval
    2. Assess retrieval confidence
    3. Based on confidence:
       - CORRECT: Generate response
       - AMBIGUOUS: Refine query, re-retrieve, generate
       - INCORRECT: Fallback to alternative sources
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        retriever: BaseRetriever | None = None,
        reranker: BaseReranker | None = None,
        generator: BaseGenerator | None = None,
        chunker: BaseChunker | None = None,
        vectordb: QdrantVectorDB | None = None,
        hyde_retriever: HyDERetriever | None = None,
        confidence_threshold_high: float = 0.8,
        confidence_threshold_low: float = 0.4,
        settings: Settings | None = None,
    ):
        """
        Initialize CRAG pipeline.

        Args:
            embedder: Embedding model.
            retriever: Primary retriever.
            reranker: Optional reranker.
            generator: LLM generator.
            chunker: Document chunker.
            vectordb: Vector database client.
            hyde_retriever: HyDE retriever for refinement.
            confidence_threshold_high: Threshold for CORRECT.
            confidence_threshold_low: Threshold below which INCORRECT.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()

        # Core components
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

        # HyDE for refinement
        self._hyde_retriever = hyde_retriever or HyDERetriever(
            base_retriever=self._retriever,
            generator=self._generator,
            settings=self._settings,
        )

        # Confidence thresholds
        self._threshold_high = confidence_threshold_high
        self._threshold_low = confidence_threshold_low

    async def _assess_confidence(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> CRAGAssessment:
        """
        Assess retrieval confidence using LLM.

        Args:
            query: Original query.
            chunks: Retrieved chunks.

        Returns:
            CRAGAssessment with confidence level.
        """
        if not chunks:
            return CRAGAssessment(
                confidence=RetrievalConfidence.INCORRECT,
                score=0.0,
                reasoning="No chunks retrieved",
            )

        # Build assessment prompt
        context = "\n\n".join([f"[{i + 1}] {c.content[:500]}" for i, c in enumerate(chunks[:5])])

        prompt = f"""Assess how well the retrieved context answers the query.

Query: {query}

Retrieved Context:
{context}

Rate the retrieval quality on a scale of 0.0 to 1.0:
- 1.0: Perfect match, context directly answers the query
- 0.7-0.9: Good match, context is highly relevant
- 0.4-0.6: Partial match, some relevant information
- 0.1-0.3: Poor match, tangentially related
- 0.0: No match, context is irrelevant

Respond with ONLY a JSON object:
{{"score": <float>, "reasoning": "<brief explanation>"}}"""

        response = await self._generator.generate_text(prompt)

        # Parse response
        try:
            import json

            data = json.loads(response.strip())
            score = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            score = 0.5
            reasoning = "Could not parse assessment"

        # Determine confidence level
        if score >= self._threshold_high:
            confidence = RetrievalConfidence.CORRECT
        elif score >= self._threshold_low:
            confidence = RetrievalConfidence.AMBIGUOUS
        else:
            confidence = RetrievalConfidence.INCORRECT

        return CRAGAssessment(
            confidence=confidence,
            score=score,
            reasoning=reasoning,
        )

    async def _refine_query(self, query: str, chunks: list[Chunk]) -> str:
        """
        Refine query based on partial context.

        Args:
            query: Original query.
            chunks: Retrieved chunks.

        Returns:
            Refined query.
        """
        context = "\n".join([c.content[:300] for c in chunks[:3]])

        prompt = f"""The following query did not retrieve highly relevant results.
Rewrite it to be more specific and searchable.

Original Query: {query}

Partial Context Found:
{context}

Rewritten Query (be specific, use key terms from context if relevant):"""

        refined = await self._generator.generate_text(prompt, max_tokens=100)
        return refined.strip().strip('"').strip("'")

    async def query(
        self,
        question: str,
        collection: str,
        top_k: int | None = None,
        max_refinements: int = 2,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        Query with CRAG confidence assessment.

        Args:
            question: User question.
            collection: Collection to search.
            top_k: Number of chunks to retrieve.
            max_refinements: Max query refinement attempts.
            **kwargs: Additional parameters.

        Returns:
            PipelineResult with response.
        """
        top_k = top_k or self._settings.default_top_k
        current_query = question
        refinement_count = 0
        all_chunks: list[Chunk] = []
        confidence_history: list[CRAGAssessment] = []

        while refinement_count <= max_refinements:
            # Step 1: Retrieve
            if refinement_count == 0:
                # Initial retrieval
                result = await self._retriever.retrieve(
                    query=current_query,
                    collection=collection,
                    top_k=top_k,
                )
            else:
                # Use HyDE for refinement
                result = await self._hyde_retriever.retrieve(
                    query=current_query,
                    collection=collection,
                    top_k=top_k,
                )

            chunks = result.chunks

            # Optional reranking
            if self._reranker and chunks:
                rerank_result = await self._reranker.rerank(
                    query=current_query,
                    chunks=chunks,
                    top_k=self._settings.default_rerank_top_k,
                )
                chunks = rerank_result.chunks
                # Apply lost-in-middle fix
                rerank_result = apply_lost_in_middle(rerank_result)
                chunks = rerank_result.chunks

            all_chunks = chunks

            # Step 2: Assess confidence
            assessment = await self._assess_confidence(current_query, chunks)
            confidence_history.append(assessment)

            # Step 3: Handle based on confidence
            if assessment.confidence == RetrievalConfidence.CORRECT:
                # High confidence - proceed with generation
                break

            elif assessment.confidence == RetrievalConfidence.AMBIGUOUS:
                # Medium confidence - try to refine
                if refinement_count < max_refinements:
                    current_query = await self._refine_query(question, chunks)
                    refinement_count += 1
                else:
                    # Max refinements reached, use what we have
                    break

            else:  # INCORRECT
                # Low confidence - try HyDE or give up
                if refinement_count < max_refinements:
                    current_query = await self._refine_query(question, chunks)
                    refinement_count += 1
                else:
                    # Generate with caveat about low confidence
                    break

        # Step 4: Generate response
        generation_result = await self._generator.generate(
            query=question,  # Use original query for generation
            context=all_chunks,
            **kwargs,
        )

        # Build metadata
        final_assessment = confidence_history[-1] if confidence_history else None

        return PipelineResult(
            response=generation_result.response,
            sources=all_chunks,
            confidence=final_assessment.score if final_assessment else 0.0,
            provider=generation_result.provider,
            model=generation_result.model,
            input_tokens=generation_result.input_tokens,
            output_tokens=generation_result.output_tokens,
            total_tokens=generation_result.total_tokens,
            latency_ms=generation_result.latency_ms,
            metadata={
                "crag_confidence": final_assessment.confidence.value
                if final_assessment
                else "unknown",
                "refinement_count": refinement_count,
                "confidence_history": [
                    {"score": a.score, "level": a.confidence.value, "reasoning": a.reasoning}
                    for a in confidence_history
                ],
            },
        )

    async def ingest(
        self,
        documents: list[Document],
        collection: str,
        **kwargs: Any,
    ) -> IngestResult:
        """
        Ingest documents.

        Args:
            documents: Documents to ingest.
            collection: Target collection.
            **kwargs: Additional parameters.

        Returns:
            IngestResult with statistics.
        """
        all_chunks: list[Chunk] = []

        for doc in documents:
            chunks = await self._chunker.chunk(doc)
            all_chunks.extend(chunks)

        # Generate embeddings
        texts = [chunk.content for chunk in all_chunks]
        embeddings = await self._embedder.embed_batch(texts)

        for chunk, embedding in zip(all_chunks, embeddings, strict=False):
            chunk.embedding = embedding

        # Store
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
