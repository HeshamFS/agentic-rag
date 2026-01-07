"""
Contextual chunking with context headers.

Anthropic's technique that reduces failed retrievals by 67%.
Each chunk gets a context header explaining its place in the document.
"""

from typing import Any

from agentic_rag.chunking.base import BaseChunker, SentenceChunker
from agentic_rag.core.models import Chunk, Document
from agentic_rag.core.protocols import Generator

# Prompt for generating context headers
CONTEXT_PROMPT_TEMPLATE = """<document>
{doc_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context (1-2 sentences) to situate this chunk within the overall document.
Focus on what comes before and after this chunk, and how it relates to the main topic.
Answer only with the context, nothing else."""


class ContextualChunker(BaseChunker):
    """
    Contextual chunker that adds context headers to chunks.

    Process:
    1. Chunk document using base chunker
    2. For each chunk, generate a context header using LLM
    3. Prepend context header to chunk content

    Benefits:
    - -67% failed retrievals (Anthropic research)
    - Chunks retain document context
    - Better semantic understanding
    """

    def __init__(
        self,
        generator: Generator,
        base_chunker: BaseChunker | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        context_template: str | None = None,
        max_doc_context: int = 8000,
    ):
        """
        Initialize contextual chunker.

        Args:
            generator: LLM for generating context headers.
            base_chunker: Underlying chunker (default: SentenceChunker).
            chunk_size: Chunk size for base chunker.
            chunk_overlap: Overlap for base chunker.
            context_template: Custom context prompt template.
            max_doc_context: Max document chars for context generation.
        """
        super().__init__(chunk_size, chunk_overlap)
        self._generator = generator
        self._base_chunker = base_chunker or SentenceChunker(
            chunk_size=chunk_size,
            chunk_overlap=1,  # 1 sentence overlap
        )
        self._context_template = context_template or CONTEXT_PROMPT_TEMPLATE
        self._max_doc_context = max_doc_context

    async def generate_context(
        self,
        document: Document,
        chunk: Chunk,
    ) -> str:
        """
        Generate context header for a chunk.

        Args:
            document: Source document.
            chunk: Chunk to contextualize.

        Returns:
            Context header string.
        """
        # Truncate document if too long
        doc_content = document.content
        if len(doc_content) > self._max_doc_context:
            # Keep beginning and end for context
            half = self._max_doc_context // 2
            doc_content = doc_content[:half] + "\n...[truncated]...\n" + doc_content[-half:]

        prompt = self._context_template.format(
            doc_content=doc_content,
            chunk_content=chunk.content,
        )

        context = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.3,  # Low temp for consistency
            max_tokens=150,
        )

        return context.strip()

    async def chunk_async(self, document: Document) -> list[Chunk]:
        """
        Chunk document with context headers (async).

        Args:
            document: Document to chunk.

        Returns:
            Chunks with context headers.
        """
        # First, chunk using base chunker
        base_chunks = self._base_chunker.chunk(document)

        if not base_chunks:
            return []

        # Generate context for each chunk
        contextualized_chunks = []
        for chunk in base_chunks:
            # Generate context header
            context = await self.generate_context(document, chunk)

            # Create new chunk with context prepended
            contextualized_content = f"[Context: {context}]\n\n{chunk.content}"

            new_chunk = Chunk(
                id=chunk.id,
                content=contextualized_content,
                document_id=chunk.document_id,
                metadata={
                    **chunk.metadata,
                    "context_header": context,
                    "original_content": chunk.content,
                    "contextualized": True,
                },
                embedding=None,
            )
            contextualized_chunks.append(new_chunk)

        return contextualized_chunks

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Synchronous wrapper for contextual chunking.

        Args:
            document: Document to chunk.

        Returns:
            Contextualized chunks.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.chunk_async(document))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.chunk_async(document))


class CachedContextualChunker(ContextualChunker):
    """
    Contextual chunker with caching to reduce LLM calls.

    Caches generated contexts based on chunk content hash.
    """

    def __init__(
        self,
        generator: Generator,
        base_chunker: BaseChunker | None = None,
        chunk_size: int = 512,
        cache_size: int = 1000,
        **kwargs: Any,
    ):
        """
        Initialize cached contextual chunker.

        Args:
            generator: LLM for context generation.
            base_chunker: Base chunker.
            chunk_size: Chunk size.
            cache_size: Maximum cache entries.
            **kwargs: Additional arguments.
        """
        super().__init__(
            generator=generator,
            base_chunker=base_chunker,
            chunk_size=chunk_size,
            **kwargs,
        )
        self._cache: dict[str, str] = {}
        self._cache_size = cache_size

    def _get_cache_key(self, document: Document, chunk: Chunk) -> str:
        """
        Generate cache key for a chunk.

        Args:
            document: Source document.
            chunk: Chunk.

        Returns:
            Cache key string.
        """
        import hashlib

        content = f"{document.source}:{chunk.content[:200]}"
        return hashlib.md5(content.encode()).hexdigest()

    async def generate_context(
        self,
        document: Document,
        chunk: Chunk,
    ) -> str:
        """
        Generate context with caching.

        Args:
            document: Source document.
            chunk: Chunk.

        Returns:
            Context header.
        """
        cache_key = self._get_cache_key(document, chunk)

        if cache_key in self._cache:
            return self._cache[cache_key]

        context = await super().generate_context(document, chunk)

        # Add to cache
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = context
        return context

    def clear_cache(self) -> None:
        """Clear the context cache."""
        self._cache.clear()


class BatchContextualChunker(ContextualChunker):
    """
    Contextual chunker that generates all contexts in a single LLM call.

    More efficient for large documents with many chunks.
    """

    async def chunk_async(self, document: Document) -> list[Chunk]:
        """
        Chunk with batch context generation.

        Args:
            document: Document to chunk.

        Returns:
            Contextualized chunks.
        """
        base_chunks = self._base_chunker.chunk(document)

        if not base_chunks:
            return []

        # Generate all contexts in one call
        contexts = await self._generate_batch_contexts(document, base_chunks)

        # Create contextualized chunks
        contextualized_chunks = []
        for chunk, context in zip(base_chunks, contexts, strict=False):
            contextualized_content = f"[Context: {context}]\n\n{chunk.content}"

            new_chunk = Chunk(
                id=chunk.id,
                content=contextualized_content,
                document_id=chunk.document_id,
                metadata={
                    **chunk.metadata,
                    "context_header": context,
                    "original_content": chunk.content,
                    "contextualized": True,
                },
                embedding=None,
            )
            contextualized_chunks.append(new_chunk)

        return contextualized_chunks

    async def _generate_batch_contexts(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> list[str]:
        """
        Generate contexts for chunks in small batches to avoid payload limits.

        Processes 10 chunks at a time to stay within API limits while
        maintaining speed with fast providers like Groq.

        Args:
            document: Source document.
            chunks: All chunks.

        Returns:
            List of context strings.
        """
        import asyncio
        import logging

        logger = logging.getLogger("agentic_rag.chunking")

        # Truncate document for context
        doc_content = document.content
        if len(doc_content) > self._max_doc_context:
            half = self._max_doc_context // 2
            doc_content = doc_content[:half] + "\n...[truncated]...\n" + doc_content[-half:]

        # Process in batches to avoid payload limits
        # Groq free tier: 30 RPM = very strict sliding window
        # Need 6+ seconds between requests to avoid 429 retries
        BATCH_SIZE = 20  # 20 chunks per batch
        RATE_LIMIT_DELAY = 6.0  # 6 seconds to avoid 429 retries
        all_contexts: list[str] = []

        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(
            f"CONTEXTUAL: Generating contexts for {len(chunks)} chunks in {total_batches} batches"
        )

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            # Rate limit delay FIRST (skip on first request)
            if batch_start > 0:
                logger.info(f"CONTEXTUAL: Rate limit delay ({RATE_LIMIT_DELAY}s)...")
                await asyncio.sleep(RATE_LIMIT_DELAY)

            batch_end = min(batch_start + BATCH_SIZE, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_num = batch_start // BATCH_SIZE + 1

            logger.info(
                f"CONTEXTUAL: Batch {batch_num}/{total_batches} (chunks {batch_start + 1}-{batch_end})"
            )

            # Build prompt for this batch
            chunks_text = ""
            for i, chunk in enumerate(batch_chunks):
                chunks_text += f'\n<chunk id="{i}">\n{chunk.content[:400]}\n</chunk>\n'

            prompt = f"""<document>
{doc_content}
</document>

Here are the chunks we want to situate within the document:
{chunks_text}

For each chunk, provide a 1-2 sentence context that situates it within the document.
Format your response as:
CHUNK_0: [context for chunk 0]
CHUNK_1: [context for chunk 1]
...

Only output the contexts, nothing else."""

            try:
                response = await self._generator.generate_text(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=100 * len(batch_chunks),  # ~100 tokens per context
                )

                # Parse response
                lines = response.strip().split("\n")

                for i, chunk in enumerate(batch_chunks):
                    context = ""
                    prefix = f"CHUNK_{i}:"

                    for line in lines:
                        if line.startswith(prefix):
                            context = line[len(prefix) :].strip()
                            break

                    if not context:
                        # Use a simple fallback context
                        context = f"This section discusses content from {document.source or 'the document'}."

                    all_contexts.append(context)

            except Exception as e:
                logger.warning(f"CONTEXTUAL: Batch failed ({e}), using fallback contexts")
                # Fallback: use simple context for this batch
                for chunk in batch_chunks:
                    all_contexts.append(
                        f"This section is from {document.source or 'the document'}."
                    )

        return all_contexts
