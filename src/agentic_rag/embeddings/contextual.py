"""
Contextual embeddings for RAG.

Embeds chunks with their surrounding context for
better semantic representation.
"""

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk
from agentic_rag.embeddings.base import BaseEmbedder


class ContextualEmbedder(BaseEmbedder):
    """
    Contextual embedder that includes surrounding context.

    When embedding a chunk, includes text from before and
    after the chunk to provide better context.
    """

    def __init__(
        self,
        base_embedder: BaseEmbedder,
        context_window: int = 200,
        settings: Settings | None = None,
    ):
        """
        Initialize contextual embedder.

        Args:
            base_embedder: Underlying embedding model.
            context_window: Characters of context to include.
            settings: Settings instance.
        """
        self._base_embedder = base_embedder
        self._context_window = context_window
        self._settings = settings or get_settings()

    @property
    def model_name(self) -> str:
        return f"contextual_{self._base_embedder.model_name}"

    @property
    def dimension(self) -> int:
        return self._base_embedder.dimension

    async def embed(self, text: str) -> list[float]:
        """Embed text (without context)."""
        return await self._base_embedder.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch (without context)."""
        return await self._base_embedder.embed_batch(texts)

    async def embed_with_context(
        self,
        chunk: Chunk,
        full_document: str,
    ) -> list[float]:
        """
        Embed chunk with surrounding context.

        Args:
            chunk: Chunk to embed.
            full_document: Full document text.

        Returns:
            Contextual embedding.
        """
        # Find chunk position
        chunk_start = full_document.find(chunk.content)
        if chunk_start == -1:
            # Fallback to direct embedding
            return await self.embed(chunk.content)

        chunk_end = chunk_start + len(chunk.content)

        # Extract context
        context_start = max(0, chunk_start - self._context_window)
        context_end = min(len(full_document), chunk_end + self._context_window)

        # Build contextual text
        before = full_document[context_start:chunk_start]
        after = full_document[chunk_end:context_end]

        contextual_text = f"{before}[CHUNK]{chunk.content}[/CHUNK]{after}"

        return await self.embed(contextual_text)

    async def embed_chunks_with_context(
        self,
        chunks: list[Chunk],
        full_document: str,
    ) -> list[list[float]]:
        """
        Embed multiple chunks with context.

        Args:
            chunks: Chunks to embed.
            full_document: Full document text.

        Returns:
            List of contextual embeddings.
        """
        import asyncio

        tasks = [self.embed_with_context(chunk, full_document) for chunk in chunks]
        return await asyncio.gather(*tasks)


class InstructionEmbedder(BaseEmbedder):
    """
    Instruction-tuned embedder.

    Prepends task-specific instructions to improve
    embedding quality for different use cases.
    """

    INSTRUCTIONS = {
        "query": "Represent this query for retrieving relevant documents: ",
        "document": "Represent this document for retrieval: ",
        "similarity": "Represent this text for semantic similarity: ",
        "classification": "Represent this text for classification: ",
        "clustering": "Represent this text for clustering: ",
    }

    def __init__(
        self,
        base_embedder: BaseEmbedder,
        default_task: str = "document",
        settings: Settings | None = None,
    ):
        """
        Initialize instruction embedder.

        Args:
            base_embedder: Underlying embedding model.
            default_task: Default task instruction.
            settings: Settings instance.
        """
        self._base_embedder = base_embedder
        self._default_task = default_task
        self._settings = settings or get_settings()

    @property
    def model_name(self) -> str:
        return f"instruction_{self._base_embedder.model_name}"

    @property
    def dimension(self) -> int:
        return self._base_embedder.dimension

    async def embed(self, text: str, task: str | None = None) -> list[float]:
        """
        Embed with instruction.

        Args:
            text: Text to embed.
            task: Task type (query, document, similarity, etc.).

        Returns:
            Embedding vector.
        """
        task = task or self._default_task
        instruction = self.INSTRUCTIONS.get(task, "")
        return await self._base_embedder.embed(f"{instruction}{text}")

    async def embed_batch(
        self,
        texts: list[str],
        task: str | None = None,
    ) -> list[list[float]]:
        """Embed batch with instruction."""
        task = task or self._default_task
        instruction = self.INSTRUCTIONS.get(task, "")
        instructed_texts = [f"{instruction}{text}" for text in texts]
        return await self._base_embedder.embed_batch(instructed_texts)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query."""
        return await self.embed(query, task="query")

    async def embed_document(self, document: str) -> list[float]:
        """Embed a document for indexing."""
        return await self.embed(document, task="document")
