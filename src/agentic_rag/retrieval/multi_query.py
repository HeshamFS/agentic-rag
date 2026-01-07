"""
Multi-Query Retrieval with query expansion.

Generates multiple query variations to improve recall,
then combines results using RRF fusion.
"""

import asyncio
from typing import Any

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import RetrievalResult
from agentic_rag.generation import BaseGenerator
from agentic_rag.retrieval.base import BaseRetriever
from agentic_rag.retrieval.fusion import reciprocal_rank_fusion


class MultiQueryRetriever(BaseRetriever):
    """
    Multi-query retriever with LLM-based query expansion.

    Generates multiple perspectives of the original query,
    retrieves for each, and fuses results with RRF.

    Benefits:
    - Improves recall by capturing different phrasings
    - Reduces sensitivity to exact query wording
    - Finds documents that match query intent, not just keywords
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        generator: BaseGenerator,
        num_queries: int = 3,
        settings: Settings | None = None,
    ):
        """
        Initialize multi-query retriever.

        Args:
            base_retriever: Underlying retriever for each query.
            generator: LLM for generating query variations.
            num_queries: Number of query variations to generate.
            settings: Settings instance.
        """
        self._base_retriever = base_retriever
        self._generator = generator
        self._num_queries = num_queries
        self._settings = settings or get_settings()

    async def _generate_query_variations(self, query: str) -> list[str]:
        """
        Generate query variations using LLM.

        Args:
            query: Original query.

        Returns:
            List of query variations including original.
        """
        import logging

        logger = logging.getLogger("agentic_rag.retrieval")

        num_to_generate = self._num_queries - 1  # Original + N-1 variations

        prompt = f"""Rewrite this search query in {num_to_generate} different ways.
Each rewrite should search for the same information using different words.

Query: {query}

Output exactly {num_to_generate} rewrites, one per line. No explanations, no numbering, no bullets.
Just the {num_to_generate} alternative queries:"""

        response = await self._generator.generate_text(prompt, max_tokens=300)
        logger.info(f"Multi-Query LLM response:\n{response}")

        # Parse variations
        variations = [query]  # Include original first
        for line in response.strip().split("\n"):
            # Clean up the line
            line = line.strip()
            if not line:
                continue
            # Remove common prefixes
            for prefix in ["-", "•", "*", "1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)"]:
                if line.startswith(prefix):
                    line = line[len(prefix) :].strip()
            # Remove "Query:" or similar prefixes
            for prefix in ["Query:", "Rewrite:", "Alternative:"]:
                if line.lower().startswith(prefix.lower()):
                    line = line[len(prefix) :].strip()
            # Remove quotes if present
            if len(line) > 2 and line[0] in "\"'`" and line[-1] in "\"'`":
                line = line[1:-1]
            # Skip if too short or looks like explanation
            if len(line) < 5:
                continue
            if any(
                skip in line.lower() for skip in ["here are", "alternative", "rewrite", "variation"]
            ):
                continue
            # Add if valid and not duplicate
            if line and line.lower() != query.lower() and line not in variations:
                variations.append(line)
                if len(variations) >= self._num_queries:
                    break

        # If we still don't have enough, log warning
        if len(variations) < self._num_queries:
            logger.warning(
                f"Multi-Query: Only generated {len(variations)} variations (wanted {self._num_queries})"
            )
            # Add simple variations as fallback
            fallback_prefixes = ["What is", "Define", "Explain"]
            for prefix in fallback_prefixes:
                if len(variations) >= self._num_queries:
                    break
                fallback = (
                    f"{prefix} {query.lower().replace('what is', '').replace('?', '').strip()}?"
                )
                if fallback not in variations:
                    variations.append(fallback)

        logger.info(f"Multi-Query: Generated {len(variations)} variations: {variations}")
        return variations[: self._num_queries]

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Execute multi-query retrieval with parallel execution.

        This method:
        1. Generates multiple variations of the original query using an LLM.
        2. Executes retrieval for all variations in parallel using the base retriever.
        3. Combines and ranks all retrieved chunks using Reciprocal Rank Fusion (RRF).
        4. Deduplicates the final result set.

        Args:
            query: The user's original search query.
            collection: Vector DB collection to search.
            top_k: Number of final results to return.
            **kwargs: Additional parameters for the underlying retriever.

        Returns:
            RetrievalResult containing the fused and deduplicated chunks.
        """
        top_k = top_k or self._settings.default_top_k

        # Generate query variations
        queries = await self._generate_query_variations(query)

        # Retrieve for each query in parallel
        async def retrieve_one(q: str) -> RetrievalResult:
            return await self._base_retriever.retrieve(
                query=q,
                collection=collection,
                top_k=top_k,
                **kwargs,
            )

        results = await asyncio.gather(*[retrieve_one(q) for q in queries])

        # Fuse with RRF (pass RetrievalResult objects directly)
        fused_result = reciprocal_rank_fusion(list(results), k=60, top_k=top_k)

        return RetrievalResult(
            chunks=fused_result.chunks,
            scores=fused_result.scores,
            retrieval_type="multi_query",
            metadata={
                "queries": queries,
                "num_queries": len(queries),
            },
        )

    async def batch_retrieve(
        self,
        queries: list[str],
        collection: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """
        Batch retrieve for multiple queries.

        Args:
            queries: List of queries.
            collection: Collection to search.
            top_k: Number of results per query.
            **kwargs: Additional parameters.

        Returns:
            List of RetrievalResults.
        """
        results = await asyncio.gather(
            *[self.retrieve(q, collection, top_k, **kwargs) for q in queries]
        )
        return list(results)


class QueryDecomposer:
    """
    Decomposes complex queries into sub-queries.

    Useful for multi-hop questions that require information
    from multiple sources.
    """

    def __init__(self, generator: BaseGenerator):
        """
        Initialize query decomposer.

        Args:
            generator: LLM for decomposition.
        """
        self._generator = generator

    async def decompose(self, query: str) -> list[str]:
        """
        Decompose a complex query into sub-queries.

        Args:
            query: Complex query.

        Returns:
            List of simpler sub-queries.
        """
        prompt = f"""Analyze this question and break it down into simpler sub-questions
that can be answered independently.

Question: {query}

If the question is already simple, return it as-is.
If it requires multiple pieces of information, list each sub-question.

Output ONLY the sub-questions, one per line, no numbering:"""

        response = await self._generator.generate_text(prompt, max_tokens=300)

        sub_queries = []
        for line in response.strip().split("\n"):
            line = line.strip().strip("-").strip("•").strip()
            if line:
                sub_queries.append(line)

        return sub_queries if sub_queries else [query]


class StepBackRetriever(BaseRetriever):
    """
    Step-Back Prompting retriever.

    First generates a more abstract "step-back" question,
    retrieves for both original and step-back, then combines.

    Reference: "Take a Step Back: Evoking Reasoning via Abstraction" (2023)
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        generator: BaseGenerator,
        settings: Settings | None = None,
    ):
        """
        Initialize step-back retriever.

        Args:
            base_retriever: Underlying retriever.
            generator: LLM for step-back generation.
            settings: Settings instance.
        """
        self._base_retriever = base_retriever
        self._generator = generator
        self._settings = settings or get_settings()

    async def _generate_step_back(self, query: str) -> str:
        """
        Generate a step-back (more abstract) version of the query.

        Args:
            query: Original specific query.

        Returns:
            More abstract step-back query.
        """
        prompt = f"""Generate a more abstract, higher-level question that would help answer this specific question.

Specific question: {query}

The step-back question should:
- Ask about underlying concepts or principles
- Be more general but still relevant
- Help provide context for the specific question

Step-back question:"""

        response = await self._generator.generate_text(prompt, max_tokens=100)
        return response.strip()

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieve using step-back prompting.

        Args:
            query: Original query.
            collection: Collection to search.
            top_k: Number of final results.
            **kwargs: Additional parameters.

        Returns:
            RetrievalResult with combined results.
        """
        top_k = top_k or self._settings.default_top_k

        # Generate step-back query
        step_back_query = await self._generate_step_back(query)

        # Retrieve for both queries in parallel
        original_result, step_back_result = await asyncio.gather(
            self._base_retriever.retrieve(query, collection, top_k=top_k, **kwargs),
            self._base_retriever.retrieve(step_back_query, collection, top_k=top_k, **kwargs),
        )

        # Fuse results
        chunk_lists = [original_result.chunks, step_back_result.chunks]
        fused_chunks, fused_scores = reciprocal_rank_fusion(chunk_lists, k=60)

        # Take top_k
        fused_chunks = fused_chunks[:top_k]
        fused_scores = fused_scores[:top_k]

        return RetrievalResult(
            chunks=fused_chunks,
            scores=fused_scores,
            retrieval_type="step_back",
            metadata={
                "original_query": query,
                "step_back_query": step_back_query,
            },
        )
