"""
HyDE (Hypothetical Document Embeddings) retrieval.

Generates a hypothetical answer to the query, then retrieves
documents similar to that hypothetical document rather than the query.

This bridges the asymmetry between queries (short, question-form)
and documents (longer, declarative form).
"""

from typing import Any

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import RetrievalResult
from agentic_rag.core.protocols import Embedder, Generator, VectorDB
from agentic_rag.retrieval.base import BaseRetriever
from agentic_rag.retrieval.dense import DenseRetriever
from agentic_rag.retrieval.hybrid import HybridRetriever

# Default HyDE prompt template
HYDE_PROMPT_TEMPLATE = """Given the following question, write a short passage that would be a perfect answer.
Do not say "the answer is" or similar. Just write the content directly as if it were from a document.

Question: {query}

Passage:"""


class HyDERetriever(BaseRetriever):
    """
    HyDE (Hypothetical Document Embeddings) retriever.

    Process:
    1. Generate a hypothetical document that would answer the query
    2. Embed the hypothetical document (not the query)
    3. Retrieve documents similar to the hypothetical document

    Benefits:
    - Bridges query-document asymmetry
    - Better semantic matching for complex queries
    - Works well with any underlying retriever
    """

    def __init__(
        self,
        generator: Generator,
        embedder: Embedder,
        vectordb: VectorDB,
        settings: Settings | None = None,
        use_hybrid: bool = True,
        prompt_template: str | None = None,
    ):
        """
        Initialize HyDE retriever.

        Args:
            generator: LLM for generating hypothetical documents.
            embedder: Embedding model.
            vectordb: Vector database.
            settings: Configuration settings.
            use_hybrid: Use hybrid retrieval (dense + sparse).
            prompt_template: Custom HyDE prompt template.
        """
        self._generator = generator
        self._embedder = embedder
        self._vectordb = vectordb
        self._settings = settings or get_settings()
        self._use_hybrid = use_hybrid
        self._prompt_template = prompt_template or HYDE_PROMPT_TEMPLATE

        # Initialize underlying retriever
        if use_hybrid:
            self._retriever: BaseRetriever = HybridRetriever(
                embedder=embedder,
                vectordb=vectordb,
                settings=self._settings,
            )
        else:
            self._retriever = DenseRetriever(
                embedder=embedder,
                vectordb=vectordb,
                settings=self._settings,
            )

    async def generate_hypothetical_document(
        self,
        query: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a hypothetical document for the query.

        Args:
            query: User query.
            temperature: Generation temperature.

        Returns:
            Hypothetical document text.
        """
        prompt = self._prompt_template.format(query=query)

        hypothetical_doc = await self._generator.generate_text(
            prompt=prompt,
            temperature=temperature,
            max_tokens=256,
        )

        return hypothetical_doc.strip()

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        temperature: float = 0.7,
        num_hypotheticals: int = 1,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Execute HyDE (Hypothetical Document Embeddings) retrieval.

        HyDE addresses query-document asymmetry by generating an "ideal" answer
        and using its embedding for retrieval. This is particularly effective
        for short queries that match long, declarative document content.

        Args:
            query: The user's search query.
            collection: Vector DB collection to search.
            top_k: Number of final results to return.
            temperature: Sampling temperature for hypothetical document generation.
            num_hypotheticals: Number of hypothetical documents to generate and fuse.
            **kwargs: Additional parameters for the underlying retriever.

        Returns:
            RetrievalResult containing chunks matched against the hypothetical document(s).
        """
        if num_hypotheticals == 1:
            # Single hypothetical document
            hypothetical = await self.generate_hypothetical_document(
                query=query,
                temperature=temperature,
            )

            # Retrieve using hypothetical as query
            result = await self._retriever.retrieve(
                query=hypothetical,
                collection=collection,
                top_k=top_k,
                **kwargs,
            )

            result.retrieval_type = "hyde"
            result.metadata.update(
                {
                    "original_query": query,
                    "hypothetical_document": hypothetical,
                }
            )

            return result

        else:
            # Multiple hypotheticals with fusion
            from agentic_rag.retrieval.fusion import RRFFusion

            hypotheticals = []
            for _ in range(num_hypotheticals):
                hypo = await self.generate_hypothetical_document(
                    query=query,
                    temperature=temperature,
                )
                hypotheticals.append(hypo)

            # Retrieve for each hypothetical
            results = []
            for hypothetical in hypotheticals:
                result = await self._retriever.retrieve(
                    query=hypothetical,
                    collection=collection,
                    top_k=top_k,
                    **kwargs,
                )
                results.append(result)

            # Fuse results
            fusion = RRFFusion(k=60)
            fused = fusion.fuse(results, top_k=top_k)

            fused.retrieval_type = "hyde_multi"
            fused.metadata.update(
                {
                    "original_query": query,
                    "hypothetical_documents": hypotheticals,
                    "num_hypotheticals": num_hypotheticals,
                }
            )

            return fused

    async def batch_retrieve(
        self,
        queries: list[str],
        collection: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """
        Retrieve for multiple queries using HyDE.

        Args:
            queries: List of queries.
            collection: Collection to search.
            top_k: Number of results per query.
            **kwargs: Additional arguments.

        Returns:
            List of RetrievalResults.
        """
        results = []
        for query in queries:
            result = await self.retrieve(
                query=query,
                collection=collection,
                top_k=top_k,
                **kwargs,
            )
            results.append(result)
        return results

    async def close(self) -> None:
        """Clean up resources."""
        await self._retriever.close()


class AdaptiveHyDERetriever(BaseRetriever):
    """
    Adaptive HyDE that decides whether to use HyDE based on query complexity.

    Simple factual queries go directly to retrieval.
    Complex or abstract queries use HyDE.
    """

    def __init__(
        self,
        generator: Generator,
        embedder: Embedder,
        vectordb: VectorDB,
        settings: Settings | None = None,
    ):
        """
        Initialize adaptive HyDE retriever.

        Args:
            generator: LLM for query analysis and hypothetical generation.
            embedder: Embedding model.
            vectordb: Vector database.
            settings: Configuration settings.
        """
        self._generator = generator
        self._settings = settings or get_settings()

        # Initialize both retrievers
        self._hyde = HyDERetriever(
            generator=generator,
            embedder=embedder,
            vectordb=vectordb,
            settings=self._settings,
        )
        self._direct = HybridRetriever(
            embedder=embedder,
            vectordb=vectordb,
            settings=self._settings,
        )

    async def should_use_hyde(self, query: str) -> bool:
        """
        Determine if HyDE would benefit this query.

        Args:
            query: User query.

        Returns:
            True if HyDE should be used.
        """
        # Simple heuristics - in production, use LLM classification
        query_lower = query.lower()

        # Short, specific queries don't need HyDE
        if len(query.split()) <= 3:
            return False

        # Questions starting with "what is" or "who is" are usually factual
        factual_prefixes = ["what is", "who is", "when was", "where is"]
        if any(query_lower.startswith(p) for p in factual_prefixes):
            return False

        # Complex "how" and "why" questions benefit from HyDE
        complex_prefixes = ["how do", "how can", "why does", "explain", "describe"]
        if any(query_lower.startswith(p) for p in complex_prefixes):
            return True

        # Default: use HyDE for longer queries
        return len(query.split()) > 5

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        force_hyde: bool | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieve with adaptive HyDE decision.

        Args:
            query: Search query.
            collection: Collection to search.
            top_k: Number of results.
            force_hyde: Force HyDE on/off (None = auto).
            **kwargs: Additional arguments.

        Returns:
            RetrievalResult.
        """
        use_hyde = force_hyde if force_hyde is not None else await self.should_use_hyde(query)

        if use_hyde:
            result = await self._hyde.retrieve(
                query=query,
                collection=collection,
                top_k=top_k,
                **kwargs,
            )
        else:
            result = await self._direct.retrieve(
                query=query,
                collection=collection,
                top_k=top_k,
                **kwargs,
            )
            result.retrieval_type = "hybrid_direct"

        result.metadata["used_hyde"] = use_hyde
        return result

    async def batch_retrieve(
        self,
        queries: list[str],
        collection: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of queries.
            collection: Collection to search.
            top_k: Number of results per query.
            **kwargs: Additional arguments.

        Returns:
            List of RetrievalResults.
        """
        results = []
        for query in queries:
            result = await self.retrieve(
                query=query,
                collection=collection,
                top_k=top_k,
                **kwargs,
            )
            results.append(result)
        return results

    async def close(self) -> None:
        """Clean up resources."""
        await self._hyde.close()
        await self._direct.close()
