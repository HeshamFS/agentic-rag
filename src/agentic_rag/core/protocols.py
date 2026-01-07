"""
Core protocol definitions for the RAG pipeline.

All components implement these protocols for maximum swappability and testability.
This follows the 2025 Modular RAG paradigm where modules can be rearranged for specific contexts.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentic_rag.core.models import (
        Chunk,
        Document,
        EvaluationResult,
        GenerationResult,
        RetrievalResult,
    )


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol for embedding models.

    Supports both single text and batch embedding operations.
    Used by dense retrieval, semantic chunking, and contextual retrieval.
    """

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text into a vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts into vectors.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """
    Protocol for document chunking strategies.

    Supports various strategies:
    - Semantic: Embedding similarity for boundaries
    - Hierarchical: Multi-level representations
    - Contextual: Add context headers before embedding
    - Late: Embed entire doc first, split after
    """

    @property
    def strategy_name(self) -> str:
        """Return the chunking strategy identifier."""
        ...

    def chunk(self, document: "Document") -> list["Chunk"]:
        """
        Split a document into chunks.

        Args:
            document: The document to chunk.

        Returns:
            List of chunks with metadata.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """
    Protocol for retrieval components.

    Supports dense, sparse, and hybrid retrieval with optional
    HyDE (Hypothetical Document Embeddings) and multi-query expansion.
    """

    @property
    def retriever_type(self) -> str:
        """Return the retriever type identifier."""
        ...

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> "RetrievalResult":
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Number of results to return.
            **kwargs: Additional retriever-specific parameters.

        Returns:
            RetrievalResult with chunks and scores.
        """
        ...

    async def retrieve_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 10,
        **kwargs: Any,
    ) -> "RetrievalResult":
        """
        Retrieve using a pre-computed embedding vector.

        Used by HyDE retrieval where we embed the hypothetical document.

        Args:
            embedding: Pre-computed embedding vector.
            top_k: Number of results to return.
            **kwargs: Additional parameters.

        Returns:
            RetrievalResult with chunks and scores.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """
    Protocol for reranking models.

    Rerankers improve retrieval accuracy by 15-40% vs semantic search alone.
    Supports cross-encoders, ColBERT late interaction, and lost-in-the-middle fixes.
    """

    @property
    def model_name(self) -> str:
        """Return the reranker model identifier."""
        ...

    async def rerank(
        self,
        query: str,
        chunks: list["Chunk"],
        top_k: int = 5,
    ) -> list[tuple["Chunk", float]]:
        """
        Rerank chunks by relevance to query.

        Args:
            query: The search query.
            chunks: Chunks to rerank.
            top_k: Number of top results to return.

        Returns:
            List of (chunk, score) tuples sorted by relevance.
        """
        ...


@runtime_checkable
class Compressor(Protocol):
    """
    Protocol for context compression.

    Addresses the "lost in the middle" problem and reduces token costs.
    Supports extractive compression, LongLLMLingua, and abstractive summarization.
    """

    @property
    def compression_type(self) -> str:
        """Return the compression type identifier."""
        ...

    async def compress(
        self,
        query: str,
        chunks: list["Chunk"],
        target_tokens: int | None = None,
    ) -> list["Chunk"]:
        """
        Compress chunks to reduce context size.

        Args:
            query: The query for relevance-aware compression.
            chunks: Chunks to compress.
            target_tokens: Optional target token count.

        Returns:
            Compressed chunks.
        """
        ...


@runtime_checkable
class Generator(Protocol):
    """
    Protocol for response generation.

    Multi-provider support: Claude, OpenAI, Gemini, Local (Ollama/vLLM).
    Includes structured output and streaming capabilities.
    """

    @property
    def provider(self) -> str:
        """Return the LLM provider (claude, openai, gemini, local)."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    async def generate(
        self,
        query: str,
        context: list["Chunk"],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> "GenerationResult":
        """
        Generate a response given query and context.

        Args:
            query: The user query.
            context: Retrieved chunks for grounding.
            system_prompt: Optional system prompt override.
            **kwargs: Additional generation parameters.

        Returns:
            GenerationResult with response and metadata.
        """
        ...

    async def generate_text(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """
        Simple text generation without RAG context.

        Used for HyDE hypothetical document generation,
        query expansion, and agent reasoning.

        Args:
            prompt: The prompt to complete.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text string.
        """
        ...


@runtime_checkable
class Evaluator(Protocol):
    """
    Protocol for RAG evaluation.

    Implements RAGAS metrics and Self-RAG reflection tokens.
    Supports NLI-based claim verification for faithfulness.
    """

    async def evaluate(
        self,
        query: str,
        response: str,
        context: list["Chunk"],
        ground_truth: str | None = None,
    ) -> "EvaluationResult":
        """
        Evaluate RAG response quality.

        Args:
            query: The original query.
            response: Generated response.
            context: Retrieved chunks used for generation.
            ground_truth: Optional ground truth answer.

        Returns:
            EvaluationResult with metrics and reflection tokens.
        """
        ...


@runtime_checkable
class VectorDB(Protocol):
    """
    Protocol for vector database operations.

    Supports Qdrant, with abstraction for other backends.
    Includes collection management and hybrid search support.
    """

    @property
    def db_type(self) -> str:
        """Return the database type identifier."""
        ...

    async def create_collection(
        self,
        name: str,
        dimension: int,
        **kwargs: Any,
    ) -> None:
        """Create a new collection."""
        ...

    async def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        ...

    async def upsert(
        self,
        collection: str,
        chunks: list["Chunk"],
    ) -> None:
        """Insert or update chunks in the collection."""
        ...

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple["Chunk", float]]:
        """
        Search for similar vectors.

        Args:
            collection: Collection name.
            query_vector: Query embedding vector.
            top_k: Number of results.
            filters: Optional metadata filters.

        Returns:
            List of (chunk, score) tuples.
        """
        ...

    async def hybrid_search(
        self,
        collection: str,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> list[tuple["Chunk", float]]:
        """
        Hybrid search combining dense and sparse retrieval.

        Args:
            collection: Collection name.
            query_vector: Dense embedding vector.
            query_text: Text for sparse matching.
            top_k: Number of results.
            alpha: Weight for dense vs sparse (0=sparse, 1=dense).

        Returns:
            List of (chunk, score) tuples.
        """
        ...


class BaseAgent(ABC):
    """
    Abstract base class for agentic components.

    Implements the 2025 Agentic RAG patterns:
    - Reflection: Agents evaluate their own decisions
    - Planning: Breaking complex tasks into steps
    - Tool Use: Dynamic API/DB integration
    - Multi-Agent: Specialized sub-agents
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return the agent identifier."""
        ...

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute the agent's main task."""
        ...

    def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Emit an observability event.

        Override for custom event handling (tracing, metrics, etc.).
        """
        pass
