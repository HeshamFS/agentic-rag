"""
Late Chunking implementation.

Late chunking reverses the traditional order: embed the entire document first
using a long-context model, then split into chunks afterward. Each chunk
embedding retains semantic information from its position in the whole document.

This solves the "context loss" problem where pronouns and references lose
their antecedents when chunking before embedding.

Reference: "Late Chunking" (Jina AI, 2025)
"""

from typing import Any

import numpy as np

from agentic_rag.chunking.base import BaseChunker
from agentic_rag.core.models import Chunk, Document
from agentic_rag.core.protocols import Embedder


class LateChunker(BaseChunker):
    """
    Late chunking: embed first, chunk later.

    Process:
    1. Embed the entire document using a long-context model
    2. Get token-level or sentence-level embeddings
    3. Split the document into chunks
    4. Pool the embeddings for each chunk span

    Benefits:
    - Each chunk retains document-level context
    - Pronouns and references maintain their antecedents
    - Significantly higher similarity scores for contextual references

    Requirements:
    - Long-context embedding model (8K+ tokens)
    - Model that provides token-level embeddings
    """

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        pooling_strategy: str = "mean",
        sentence_level: bool = True,
    ):
        """
        Initialize late chunker.

        Args:
            embedder: Long-context embedding model.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks.
            pooling_strategy: How to pool embeddings ("mean", "max", "first", "last").
            sentence_level: If True, chunk at sentence boundaries.
        """
        super().__init__(chunk_size, chunk_overlap)
        self._embedder = embedder
        self._pooling_strategy = pooling_strategy
        self._sentence_level = sentence_level

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunk document using late chunking.

        This is a synchronous wrapper that calls the async implementation.
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

    async def chunk_async(self, document: Document) -> list[Chunk]:
        """
        Chunk document using late chunking (async).

        Args:
            document: Document to chunk.

        Returns:
            List of chunks with context-aware embeddings.
        """
        if not document.content.strip():
            return []

        # Step 1: Split document into sentences/segments
        segments = self._segment_document(document.content)

        if not segments:
            return []

        # Step 2: Get embeddings for all segments
        # For true late chunking, we'd need token-level embeddings
        # Here we approximate with sentence-level embeddings that
        # are computed with full document context

        # Create chunks from segments
        chunks = self._create_chunks_from_segments(document, segments)

        # Step 3: Embed chunks with document context
        # We embed each chunk with surrounding context
        chunk_embeddings = await self._embed_with_context(
            document.content,
            chunks,
        )

        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, chunk_embeddings, strict=False):
            chunk.embedding = embedding

        return chunks

    def _segment_document(self, text: str) -> list[dict[str, Any]]:
        """
        Segment document into sentences or fixed segments.

        Returns list of dicts with 'text', 'start', 'end' positions.
        """
        import re

        segments = []

        if self._sentence_level:
            # Split into sentences
            sentence_pattern = r"(?<=[.!?])\s+"
            sentences = re.split(sentence_pattern, text)

            position = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    start = text.find(sentence, position)
                    if start == -1:
                        start = position
                    end = start + len(sentence)

                    segments.append(
                        {
                            "text": sentence,
                            "start": start,
                            "end": end,
                        }
                    )
                    position = end
        else:
            # Fixed-size segments
            words = text.split()
            words_per_segment = 50  # ~50 words per segment

            position = 0
            for i in range(0, len(words), words_per_segment):
                segment_words = words[i : i + words_per_segment]
                segment_text = " ".join(segment_words)

                start = text.find(segment_text, position)
                if start == -1:
                    start = position
                end = start + len(segment_text)

                segments.append(
                    {
                        "text": segment_text,
                        "start": start,
                        "end": end,
                    }
                )
                position = end

        return segments

    def _create_chunks_from_segments(
        self,
        document: Document,
        segments: list[dict[str, Any]],
    ) -> list[Chunk]:
        """
        Group segments into chunks based on size.
        """
        from uuid import uuid4

        chunks = []
        current_segments = []
        current_size = 0

        for segment in segments:
            segment_size = len(segment["text"])

            if current_size + segment_size > self.chunk_size and current_segments:
                # Create chunk from current segments
                chunk_text = " ".join(s["text"] for s in current_segments)
                chunk = Chunk(
                    id=str(uuid4()),
                    content=chunk_text,
                    document_id=document.id,
                    metadata={
                        **document.metadata,
                        "start_pos": current_segments[0]["start"],
                        "end_pos": current_segments[-1]["end"],
                        "chunking_method": "late_chunking",
                    },
                )
                chunks.append(chunk)

                # Handle overlap
                overlap_size = 0
                overlap_segments = []
                for s in reversed(current_segments):
                    if overlap_size < self.chunk_overlap:
                        overlap_segments.insert(0, s)
                        overlap_size += len(s["text"])
                    else:
                        break

                current_segments = overlap_segments
                current_size = sum(len(s["text"]) for s in current_segments)

            current_segments.append(segment)
            current_size += segment_size

        # Don't forget the last chunk
        if current_segments:
            chunk_text = " ".join(s["text"] for s in current_segments)
            chunk = Chunk(
                id=str(uuid4()),
                content=chunk_text,
                document_id=document.id,
                metadata={
                    **document.metadata,
                    "start_pos": current_segments[0]["start"],
                    "end_pos": current_segments[-1]["end"],
                    "chunking_method": "late_chunking",
                },
            )
            chunks.append(chunk)

        return chunks

    async def _embed_with_context(
        self,
        full_document: str,
        chunks: list[Chunk],
    ) -> list[list[float]]:
        """
        Embed chunks with document context.

        For true late chunking, we'd get token embeddings for the full doc
        and pool them for each chunk span. Here we approximate by:
        1. Including surrounding context in the embedding input
        2. Using the context-aware nature of transformer models
        """
        embeddings = []

        # Option 1: Embed full document and compute chunk embeddings
        # This requires a model that returns token-level embeddings
        # Most embedding models only return [CLS] embeddings

        # Option 2: Embed each chunk with surrounding context
        # This is an approximation that still preserves some context
        context_window = 500  # chars of context on each side

        for chunk in chunks:
            start_pos = chunk.metadata.get("start_pos", 0)
            end_pos = chunk.metadata.get("end_pos", len(chunk.content))

            # Get surrounding context
            context_start = max(0, start_pos - context_window)
            context_end = min(len(full_document), end_pos + context_window)

            # Build contextualized input
            # Format: [CONTEXT BEFORE] [CHUNK] [CONTEXT AFTER]
            # The model will attend to all parts

            before_context = full_document[context_start:start_pos].strip()
            after_context = full_document[end_pos:context_end].strip()

            # For embedding, we use the full contextualized text
            # but focus on the main chunk
            contextualized = f"{before_context} {chunk.content} {after_context}"

            # Embed the contextualized text
            embedding = await self._embedder.embed_text(contextualized)
            embeddings.append(embedding)

        return embeddings

    def _pool_embeddings(
        self,
        token_embeddings: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> list[float]:
        """
        Pool token embeddings for a span.

        Args:
            token_embeddings: Array of shape (num_tokens, embedding_dim)
            start_idx: Start token index.
            end_idx: End token index.

        Returns:
            Pooled embedding vector.
        """
        span_embeddings = token_embeddings[start_idx:end_idx]

        if self._pooling_strategy == "mean":
            pooled = np.mean(span_embeddings, axis=0)
        elif self._pooling_strategy == "max":
            pooled = np.max(span_embeddings, axis=0)
        elif self._pooling_strategy == "first":
            pooled = span_embeddings[0]
        elif self._pooling_strategy == "last":
            pooled = span_embeddings[-1]
        else:
            # Default to mean
            pooled = np.mean(span_embeddings, axis=0)

        return pooled.tolist()


class TrueLateChucker(LateChunker):
    """
    True late chunking using token-level embeddings.

    Requires a model that returns token-level embeddings
    (not just the [CLS] embedding).

    This is the "real" late chunking as described in Jina AI's paper.
    """

    async def chunk_async(self, document: Document) -> list[Chunk]:
        """
        Chunk using true late chunking with token-level embeddings.
        """
        if not document.content.strip():
            return []

        # Check if embedder supports token-level embeddings
        if not hasattr(self._embedder, "embed_tokens"):
            # Fall back to approximation
            return await super().chunk_async(document)

        # Step 1: Get token-level embeddings for full document
        token_embeddings, tokens = await self._embedder.embed_tokens(document.content)

        # Step 2: Segment document and map to token indices
        segments = self._segment_document(document.content)

        # Step 3: Create chunks with pooled embeddings
        chunks = []
        current_segments = []
        current_size = 0

        for segment in segments:
            segment_size = len(segment["text"])

            if current_size + segment_size > self.chunk_size and current_segments:
                # Create chunk
                chunk = self._create_chunk_with_pooled_embedding(
                    document,
                    current_segments,
                    token_embeddings,
                    tokens,
                )
                chunks.append(chunk)

                # Handle overlap (simplified)
                current_segments = current_segments[-2:] if len(current_segments) > 2 else []
                current_size = sum(len(s["text"]) for s in current_segments)

            current_segments.append(segment)
            current_size += segment_size

        # Last chunk
        if current_segments:
            chunk = self._create_chunk_with_pooled_embedding(
                document,
                current_segments,
                token_embeddings,
                tokens,
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk_with_pooled_embedding(
        self,
        document: Document,
        segments: list[dict[str, Any]],
        token_embeddings: np.ndarray,
        tokens: list[str],
    ) -> Chunk:
        """Create chunk with embedding pooled from token embeddings."""
        from uuid import uuid4

        chunk_text = " ".join(s["text"] for s in segments)
        start_pos = segments[0]["start"]
        end_pos = segments[-1]["end"]

        # Find token indices for this span
        # This is simplified - real implementation would need proper tokenizer
        char_to_token = self._map_chars_to_tokens(document.content, tokens)

        start_token = char_to_token.get(start_pos, 0)
        end_token = char_to_token.get(end_pos, len(tokens))

        # Pool embeddings
        embedding = self._pool_embeddings(token_embeddings, start_token, end_token)

        return Chunk(
            id=str(uuid4()),
            content=chunk_text,
            document_id=document.id,
            embedding=embedding,
            metadata={
                **document.metadata,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "start_token": start_token,
                "end_token": end_token,
                "chunking_method": "true_late_chunking",
            },
        )

    def _map_chars_to_tokens(
        self,
        text: str,
        tokens: list[str],
    ) -> dict[int, int]:
        """
        Map character positions to token indices.

        This is a simplified version - production would use tokenizer offsets.
        """
        char_to_token = {}
        char_pos = 0

        for token_idx, token in enumerate(tokens):
            # Find where this token appears in text
            token_text = token.replace("##", "").replace("▁", "")  # Handle subword tokens
            found_pos = text.find(token_text, char_pos)

            if found_pos != -1:
                for i in range(found_pos, found_pos + len(token_text)):
                    char_to_token[i] = token_idx
                char_pos = found_pos + len(token_text)

        return char_to_token
