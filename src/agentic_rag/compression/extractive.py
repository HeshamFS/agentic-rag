"""
Extractive context compression using reranker-based sentence selection.

This is the simplest and fastest compression approach:
1. Split chunks into sentences
2. Score each sentence with reranker
3. Select top sentences until target token count
"""

import logging
import re
from typing import Any

from agentic_rag.compression.base import BaseCompressor, CompressionResult
from agentic_rag.core.models import Chunk

logger = logging.getLogger(__name__)


class ExtractiveCompressor(BaseCompressor):
    """
    Extractive compressor using reranker for sentence scoring.

    Fast and effective compression that:
    - Splits chunks into sentences
    - Scores sentences against query using reranker
    - Selects top-scoring sentences up to token limit

    Example:
        compressor = ExtractiveCompressor(reranker=my_reranker)
        result = await compressor.compress(
            query="What is RAG?",
            chunks=retrieved_chunks,
            compression_ratio=0.5,
        )
        # Uses 50% fewer tokens
    """

    def __init__(
        self,
        reranker: Any,
        target_tokens: int | None = None,
        compression_ratio: float = 0.5,
        min_sentences: int = 3,
        batch_size: int = 32,
    ):
        """
        Initialize extractive compressor.

        Args:
            reranker: Reranker for sentence scoring.
            target_tokens: Target token count.
            compression_ratio: Target compression ratio.
            min_sentences: Minimum sentences to keep.
            batch_size: Batch size for reranker.
        """
        super().__init__(
            target_tokens=target_tokens,
            compression_ratio=compression_ratio,
            min_chunks=1,
        )
        self._reranker = reranker
        self._min_sentences = min_sentences
        self._batch_size = batch_size

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting (handles common cases)
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern, text)
        # Filter empty and very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    async def compress(
        self,
        query: str,
        chunks: list[Chunk],
        target_tokens: int | None = None,
        compression_ratio: float | None = None,
    ) -> CompressionResult:
        """
        Compress chunks using extractive sentence selection.

        Args:
            query: Query for relevance scoring.
            chunks: Chunks to compress.
            target_tokens: Override target tokens.
            compression_ratio: Override compression ratio.

        Returns:
            CompressionResult with selected sentences.
        """
        if not chunks:
            return CompressionResult()

        ratio = compression_ratio or self._compression_ratio
        target = target_tokens or self._target_tokens

        # Calculate original tokens
        original_tokens = self._chunks_to_tokens(chunks)

        # Calculate target if not specified
        if target is None:
            target = int(original_tokens * ratio)

        logger.info(f"Compressing {len(chunks)} chunks: {original_tokens} -> ~{target} tokens")

        # Extract all sentences with source tracking
        all_sentences: list[tuple[str, int, int]] = []  # (sentence, chunk_idx, sent_idx)

        for chunk_idx, chunk in enumerate(chunks):
            sentences = self._split_sentences(chunk.content)
            for sent_idx, sentence in enumerate(sentences):
                all_sentences.append((sentence, chunk_idx, sent_idx))

        if not all_sentences:
            # No sentences found, return original chunks
            return CompressionResult(
                compressed_chunks=chunks,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
            )

        # Score sentences with reranker
        sentence_texts = [s[0] for s in all_sentences]

        # Create pseudo-chunks for reranker
        sentence_chunks = [
            Chunk(content=text, document_id=f"sent_{i}") for i, text in enumerate(sentence_texts)
        ]

        # Rerank sentences
        try:
            rerank_result = await self._reranker.rerank(
                query=query,
                chunks=sentence_chunks,
                top_k=len(sentence_chunks),  # Get all scores
            )

            # Map scores back to sentences
            scored_sentences = []
            for ranked_chunk in rerank_result.chunks:
                # Find original sentence index
                idx = int(ranked_chunk.document_id.split("_")[1])
                sentence, chunk_idx, sent_idx = all_sentences[idx]
                score = ranked_chunk.metadata.get("rerank_score", 0.5)
                scored_sentences.append((sentence, chunk_idx, sent_idx, score))

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using position-based selection")
            # Fallback: use original order with decreasing scores
            scored_sentences = [
                (sent, cidx, sidx, 1.0 - (i / len(all_sentences)))
                for i, (sent, cidx, sidx) in enumerate(all_sentences)
            ]

        # Sort by score descending
        scored_sentences.sort(key=lambda x: x[3], reverse=True)

        # Select sentences until target tokens
        selected_sentences: list[tuple[str, int, int, float]] = []
        current_tokens = 0

        for sentence, chunk_idx, sent_idx, score in scored_sentences:
            sent_tokens = self._estimate_tokens(sentence)

            if (
                current_tokens + sent_tokens <= target
                or len(selected_sentences) < self._min_sentences
            ):
                selected_sentences.append((sentence, chunk_idx, sent_idx, score))
                current_tokens += sent_tokens

                if current_tokens >= target and len(selected_sentences) >= self._min_sentences:
                    break

        # Group selected sentences by chunk
        chunk_sentences: dict[int, list[tuple[int, str]]] = {}
        for sentence, chunk_idx, sent_idx, _ in selected_sentences:
            if chunk_idx not in chunk_sentences:
                chunk_sentences[chunk_idx] = []
            chunk_sentences[chunk_idx].append((sent_idx, sentence))

        # Reconstruct compressed chunks (preserve sentence order within chunk)
        compressed_chunks = []
        for chunk_idx in sorted(chunk_sentences.keys()):
            original_chunk = chunks[chunk_idx]
            sentences = chunk_sentences[chunk_idx]
            sentences.sort(key=lambda x: x[0])  # Sort by original position

            compressed_content = " ".join(s[1] for s in sentences)

            compressed_chunk = Chunk(
                id=original_chunk.id,
                content=compressed_content,
                document_id=original_chunk.document_id,
                metadata={
                    **original_chunk.metadata,
                    "compressed": True,
                    "original_length": len(original_chunk.content),
                    "sentences_kept": len(sentences),
                },
                embedding=original_chunk.embedding,
            )
            compressed_chunks.append(compressed_chunk)

        compressed_tokens = self._chunks_to_tokens(compressed_chunks)
        actual_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        logger.info(
            f"Compression complete: {original_tokens} -> {compressed_tokens} tokens "
            f"({(1 - actual_ratio) * 100:.1f}% reduction)"
        )

        return CompressionResult(
            compressed_chunks=compressed_chunks,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=actual_ratio,
            metadata={
                "sentences_total": len(all_sentences),
                "sentences_selected": len(selected_sentences),
                "chunks_with_content": len(compressed_chunks),
            },
        )
