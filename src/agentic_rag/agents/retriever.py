"""
Retriever Agent implementing CRAG (Corrective RAG).

Makes intelligent decisions about retrieval quality and takes
corrective actions when initial retrieval is insufficient.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.agents.base import AgentState, BaseAgent
from agentic_rag.config import Settings
from agentic_rag.core.models import Chunk, RetrievalResult
from agentic_rag.core.protocols import Generator


class RetrievalQuality(str, Enum):
    """Quality assessment of retrieved documents."""

    EXCELLENT = "excellent"  # High relevance, sufficient coverage
    GOOD = "good"  # Relevant but may need supplementing
    PARTIAL = "partial"  # Some relevant, some irrelevant
    POOR = "poor"  # Mostly irrelevant
    EMPTY = "empty"  # No results


class CorrectionAction(str, Enum):
    """Corrective actions for retrieval."""

    PROCEED = "proceed"  # Use current results
    REFINE_QUERY = "refine_query"  # Reformulate query
    EXPAND_SEARCH = "expand_search"  # Retrieve more documents
    WEB_SEARCH = "web_search"  # Fallback to web search
    DECOMPOSE = "decompose"  # Break into sub-queries
    ABORT = "abort"  # Cannot answer from knowledge base


class RetrieverOutput(BaseModel):
    """Output from the retriever agent."""

    quality: RetrievalQuality = Field(description="Quality assessment")
    action: CorrectionAction = Field(description="Recommended action")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assessment")
    relevant_chunks: list[int] = Field(
        default_factory=list, description="Indices of relevant chunks"
    )
    refined_query: str = Field(default="", description="Refined query if action is REFINE_QUERY")
    reasoning: str = Field(default="", description="Explanation of assessment")
    knowledge_gaps: list[str] = Field(default_factory=list, description="Identified knowledge gaps")


class RetrieverAgent(BaseAgent[RetrieverOutput]):
    """
    Retriever Agent with CRAG (Corrective RAG) capabilities.

    Responsibilities:
    1. Evaluate retrieval quality
    2. Filter irrelevant documents
    3. Identify knowledge gaps
    4. Decide on corrective actions
    5. Refine queries when needed
    """

    def __init__(
        self,
        generator: Generator,
        settings: Settings | None = None,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize retriever agent.

        Args:
            generator: LLM for evaluation.
            settings: Configuration settings.
            confidence_threshold: Minimum confidence for "good" quality.
        """
        super().__init__(
            generator=generator,
            settings=settings,
            name="RetrieverAgent",
        )
        self.confidence_threshold = confidence_threshold

    def _get_default_system_prompt(self) -> str:
        """Get retriever-specific system prompt."""
        return """You are a retrieval quality evaluator for a RAG system.
Your job is to assess whether retrieved documents are relevant and sufficient
to answer the user's query.

You must be critical - if documents don't directly address the query, mark them as irrelevant.
Consider:
- Does the document contain information that answers the query?
- Is the information accurate and up-to-date?
- Are there any knowledge gaps?

Always respond with a JSON object containing your assessment."""

    async def execute(self, state: AgentState) -> RetrieverOutput:
        """
        Evaluate the quality of retrieved documents and decide on corrective actions (CRAG).

        The assessment follows these steps:
        1. Individual Assessment: Each retrieved chunk is evaluated for relevance using the LLM.
        2. Aggregation: Individual assessments are aggregated into an overall quality score.
        3. Decision: Based on the quality, a corrective action is chosen:
           - EXCELLENT/GOOD: PROCEED to generation.
           - PARTIAL: EXPAND_SEARCH (e.g., using HyDE or more results).
           - POOR: REFINE_QUERY and try retrieval again.
           - EMPTY: Fallback to WEB_SEARCH if enabled.

        Args:
            state: AgentState containing the query and initial retrieval results.

        Returns:
            RetrieverOutput with quality assessment, recommended action, and relevant chunk indices.
        """
        query = state.query
        retrieval_result = state.context.get("retrieval_result")

        if not retrieval_result or not retrieval_result.chunks:
            return RetrieverOutput(
                quality=RetrievalQuality.EMPTY,
                action=CorrectionAction.WEB_SEARCH,
                confidence=1.0,
                reasoning="No documents retrieved",
            )

        # Evaluate each chunk
        chunks = retrieval_result.chunks
        chunk_assessments = await self._assess_chunks(query, chunks)

        # Aggregate assessment
        return self._aggregate_assessment(query, chunks, chunk_assessments)

    async def _assess_chunks(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> list[dict[str, Any]]:
        """
        Assess relevance of each chunk.

        Args:
            query: User query.
            chunks: Retrieved chunks.

        Returns:
            List of chunk assessments.
        """
        assessments = []

        # Build batch assessment prompt for efficiency
        chunks_text = ""
        for i, chunk in enumerate(chunks[:10]):  # Limit to top 10
            chunks_text += f"\n[Document {i}]\n{chunk.content[:500]}\n"

        prompt = f"""Query: "{query}"

Retrieved Documents:
{chunks_text}

For each document, assess its relevance to answering the query.
Output a JSON object with the following structure:
{{
    "assessments": [
        {{"doc_id": 0, "relevant": true/false, "score": 0.0-1.0, "reason": "brief reason"}},
        ...
    ],
    "overall_coverage": "full|partial|minimal|none",
    "knowledge_gaps": ["gap1", "gap2"]
}}"""

        response = await self.think(prompt, temperature=0.2)

        # Parse response
        try:
            import json
            import re

            json_match = re.search(r"\{[\s\S]*\}", str(response))
            if json_match:
                data = json.loads(json_match.group())
                return data.get("assessments", [])
        except Exception:
            pass

        # Fallback: simple keyword matching
        for i, chunk in enumerate(chunks):
            query_terms = set(query.lower().split())
            chunk_terms = set(chunk.content.lower().split())
            overlap = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0

            assessments.append(
                {
                    "doc_id": i,
                    "relevant": overlap > 0.3,
                    "score": min(overlap * 2, 1.0),
                    "reason": "keyword overlap assessment",
                }
            )

        return assessments

    def _aggregate_assessment(
        self,
        query: str,
        chunks: list[Chunk],
        assessments: list[dict[str, Any]],
    ) -> RetrieverOutput:
        """
        Aggregate chunk assessments into overall quality.

        Args:
            query: User query.
            chunks: Retrieved chunks.
            assessments: Individual chunk assessments.

        Returns:
            Overall retrieval assessment.
        """
        if not assessments:
            return RetrieverOutput(
                quality=RetrievalQuality.POOR,
                action=CorrectionAction.REFINE_QUERY,
                confidence=0.5,
                reasoning="Could not assess retrieval quality",
            )

        # Calculate statistics
        relevant_count = sum(1 for a in assessments if a.get("relevant", False))
        total_count = len(assessments)
        avg_score = sum(a.get("score", 0) for a in assessments) / total_count if total_count else 0
        relevant_indices = [a["doc_id"] for a in assessments if a.get("relevant", False)]

        # Determine quality
        relevance_ratio = relevant_count / total_count if total_count else 0

        if relevance_ratio >= 0.7 and avg_score >= 0.7:
            quality = RetrievalQuality.EXCELLENT
            action = CorrectionAction.PROCEED
            confidence = avg_score
        elif relevance_ratio >= 0.5 or avg_score >= 0.5:
            quality = RetrievalQuality.GOOD
            action = CorrectionAction.PROCEED
            confidence = avg_score
        elif relevance_ratio >= 0.3 or avg_score >= 0.3:
            quality = RetrievalQuality.PARTIAL
            action = CorrectionAction.EXPAND_SEARCH
            confidence = avg_score
        else:
            quality = RetrievalQuality.POOR
            action = CorrectionAction.REFINE_QUERY
            confidence = avg_score

        return RetrieverOutput(
            quality=quality,
            action=action,
            confidence=confidence,
            relevant_chunks=relevant_indices,
            reasoning=f"{relevant_count}/{total_count} documents relevant, avg score {avg_score:.2f}",
        )

    async def refine_query(
        self,
        original_query: str,
        failed_retrieval: RetrievalResult,
    ) -> str:
        """
        Refine a query that produced poor results.

        Args:
            original_query: Original query that failed.
            failed_retrieval: The poor retrieval result.

        Returns:
            Refined query.
        """
        # Get some context from failed retrieval
        sample_content = ""
        if failed_retrieval.chunks:
            sample_content = "\n".join(c.content[:200] for c in failed_retrieval.chunks[:3])

        prompt = f"""The following query produced poor retrieval results:

Query: "{original_query}"

Sample of retrieved content (not relevant enough):
{sample_content}

Generate a refined query that might retrieve more relevant documents.
Consider:
- Using different terminology
- Being more specific
- Adding context

Output only the refined query, nothing else."""

        refined = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
        )

        return refined.strip().strip('"')

    def filter_relevant_chunks(
        self,
        chunks: list[Chunk],
        relevant_indices: list[int],
    ) -> list[Chunk]:
        """
        Filter to only relevant chunks.

        Args:
            chunks: All retrieved chunks.
            relevant_indices: Indices of relevant chunks.

        Returns:
            Filtered list of relevant chunks.
        """
        return [chunks[i] for i in relevant_indices if i < len(chunks)]


class AdaptiveRetrieverAgent(RetrieverAgent):
    """
    Adaptive retriever that adjusts strategy based on query characteristics.

    Learns from successful retrievals to improve future performance.
    """

    def __init__(
        self,
        generator: Generator,
        settings: Settings | None = None,
    ):
        """
        Initialize adaptive retriever.

        Args:
            generator: LLM for evaluation.
            settings: Configuration settings.
        """
        super().__init__(generator, settings)
        self._success_patterns: list[dict[str, Any]] = []
        self._failure_patterns: list[dict[str, Any]] = []

    def record_outcome(
        self,
        query: str,
        strategy: str,
        success: bool,
        metrics: dict[str, Any],
    ) -> None:
        """
        Record retrieval outcome for learning.

        Args:
            query: The query.
            strategy: Strategy used.
            success: Whether retrieval was successful.
            metrics: Performance metrics.
        """
        pattern = {
            "query_features": self._extract_query_features(query),
            "strategy": strategy,
            "metrics": metrics,
        }

        if success:
            self._success_patterns.append(pattern)
        else:
            self._failure_patterns.append(pattern)

        # Limit memory
        self._success_patterns = self._success_patterns[-100:]
        self._failure_patterns = self._failure_patterns[-100:]

    def _extract_query_features(self, query: str) -> dict[str, Any]:
        """Extract features from a query for pattern matching."""
        words = query.lower().split()
        return {
            "length": len(words),
            "has_question_word": any(
                w in words for w in ["what", "how", "why", "when", "where", "who"]
            ),
            "has_comparison": any(w in words for w in ["compare", "difference", "versus", "vs"]),
            "has_temporal": any(w in words for w in ["latest", "recent", "current", "new"]),
        }

    def suggest_strategy(self, query: str) -> str:
        """
        Suggest retrieval strategy based on learned patterns.

        Args:
            query: Query to suggest strategy for.

        Returns:
            Suggested strategy name.
        """
        features = self._extract_query_features(query)

        # Find similar successful patterns
        best_strategy = "hybrid"  # Default
        best_score = 0

        for pattern in self._success_patterns:
            similarity = self._feature_similarity(features, pattern["query_features"])
            if similarity > best_score:
                best_score = similarity
                best_strategy = pattern["strategy"]

        return best_strategy

    def _feature_similarity(
        self,
        f1: dict[str, Any],
        f2: dict[str, Any],
    ) -> float:
        """Calculate similarity between feature sets."""
        matches = sum(1 for k in f1 if f1.get(k) == f2.get(k))
        total = len(f1)
        return matches / total if total else 0
