"""
Sparse retriever using BM25.

Provides keyword-based retrieval using BM25 algorithm.
"""

import math
from collections import Counter
from typing import Any

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, RetrievalResult
from agentic_rag.core.protocols import VectorDB
from agentic_rag.retrieval.base import BaseRetriever


class BM25Index:
    """
    In-memory BM25 index for sparse retrieval.

    Implements the Okapi BM25 algorithm for keyword matching.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter.
            b: Document length normalization parameter.
        """
        self.k1 = k1
        self.b = b

        # Index data
        self._docs: list[Chunk] = []
        self._doc_freqs: Counter[str] = Counter()
        self._doc_lens: list[int] = []
        self._avg_doc_len: float = 0.0
        self._doc_term_freqs: list[Counter[str]] = []
        self._n_docs: int = 0

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        # Simple whitespace tokenization with lowercasing
        # In production, use a proper tokenizer
        import re

        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def index(self, chunks: list[Chunk]) -> None:
        """
        Index a list of chunks.

        Args:
            chunks: Chunks to index.
        """
        self._docs = chunks
        self._n_docs = len(chunks)
        self._doc_term_freqs = []
        self._doc_lens = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            term_freq = Counter(tokens)
            self._doc_term_freqs.append(term_freq)
            self._doc_lens.append(len(tokens))

            # Update document frequencies
            for term in set(tokens):
                self._doc_freqs[term] += 1

        # Calculate average document length
        if self._n_docs > 0:
            self._avg_doc_len = sum(self._doc_lens) / self._n_docs
        else:
            self._avg_doc_len = 0.0

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """
        Search the index.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of (chunk, score) tuples.
        """
        if not self._docs:
            return []

        query_tokens = self._tokenize(query)
        scores: list[float] = []

        for doc_idx in range(self._n_docs):
            score = self._score_document(query_tokens, doc_idx)
            scores.append(score)

        # Get top-k results
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append((self._docs[doc_idx], score))

        return results

    def _score_document(
        self,
        query_tokens: list[str],
        doc_idx: int,
    ) -> float:
        """
        Calculate BM25 score for a document.

        Args:
            query_tokens: Tokenized query.
            doc_idx: Document index.

        Returns:
            BM25 score.
        """
        score = 0.0
        doc_len = self._doc_lens[doc_idx]
        term_freqs = self._doc_term_freqs[doc_idx]

        for term in query_tokens:
            if term not in self._doc_freqs:
                continue

            # Document frequency
            df = self._doc_freqs[term]

            # Inverse document frequency
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            # Term frequency in document
            tf = term_freqs.get(term, 0)

            # BM25 score component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1))

            score += idf * (numerator / denominator)

        return score


class SparseRetriever(BaseRetriever):
    """
    Sparse retriever using BM25 algorithm.

    Provides keyword-based retrieval complementary to dense retrieval.
    """

    def __init__(
        self,
        vectordb: VectorDB,
        settings: Settings | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize sparse retriever.

        Args:
            vectordb: Vector database for fetching documents.
            settings: Configuration settings.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.
        """
        self._vectordb = vectordb
        self._settings = settings or get_settings()
        self._k1 = k1
        self._b = b

        # Collection-specific indices
        self._indices: dict[str, BM25Index] = {}

    async def _ensure_index(self, collection: str) -> BM25Index:
        """
        Ensure BM25 index exists for collection.

        Args:
            collection: Collection name.

        Returns:
            BM25Index for collection.
        """
        if collection not in self._indices:
            # Fetch all chunks from collection
            chunks = await self._vectordb.get_all(collection)

            # Build index
            index = BM25Index(k1=self._k1, b=self._b)
            index.index(chunks)
            self._indices[collection] = index

        return self._indices[collection]

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieve chunks using sparse BM25 keyword matching.

        1. Ensures a BM25 index is built for the specified collection.
        2. Performs a keyword-based search on the index.
        3. Returns chunks ranked by their BM25 relevance score.

        Args:
            query: The user search query.
            collection: Vector DB collection name.
            top_k: Number of results to return.
            **kwargs: Additional parameters.

        Returns:
            RetrievalResult containing the ranked chunks and their BM25 scores.
        """
        index = await self._ensure_index(collection)
        results = index.search(query, top_k=top_k)

        chunks = []
        scores = []
        for chunk, score in results:
            chunks.append(chunk)
            scores.append(score)

        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            retrieval_type="sparse",
            metadata={"query": query, "top_k": top_k, "method": "bm25"},
        )

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
        index = await self._ensure_index(collection)

        results = []
        for query in queries:
            search_results = index.search(query, top_k=top_k)

            chunks = []
            scores = []
            for chunk, score in search_results:
                chunks.append(chunk)
                scores.append(score)

            results.append(
                RetrievalResult(
                    chunks=chunks,
                    scores=scores,
                    retrieval_type="sparse",
                    metadata={"query": query, "top_k": top_k, "method": "bm25"},
                )
            )

        return results

    def invalidate_cache(self, collection: str | None = None) -> None:
        """
        Invalidate BM25 index cache.

        Args:
            collection: Collection to invalidate, or None for all.
        """
        if collection:
            self._indices.pop(collection, None)
        else:
            self._indices.clear()

    async def close(self) -> None:
        """Clean up resources."""
        self._indices.clear()
        await self._vectordb.close()
