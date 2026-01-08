"""
LongLLMLingua-style context compression using perplexity scoring.

Based on the LongLLMLingua paper's approach:
1. Use a small LM to compute per-token perplexity given the query
2. Tokens with low perplexity (predictable) are less informative
3. Keep high-perplexity (surprising/informative) tokens

This implementation uses sentence-level perplexity for efficiency.
"""

import logging
from typing import Any

from agentic_rag.compression.base import BaseCompressor, CompressionResult
from agentic_rag.core.models import Chunk

logger = logging.getLogger(__name__)


class LongLLMLinguaCompressor(BaseCompressor):
    """
    LongLLMLingua-style compressor using query-conditioned perplexity.

    This compressor:
    1. Conditions a small LM on the query
    2. Scores sentences by how "surprising" they are given the query
    3. High perplexity = informative for answering the query
    4. Selects highest-perplexity sentences

    Uses sentence-level scoring for efficiency (vs token-level).

    Example:
        compressor = LongLLMLinguaCompressor(generator=fast_llm)
        result = await compressor.compress(
            query="What causes climate change?",
            chunks=chunks,
            compression_ratio=0.3,  # Keep 30%
        )
    """

    def __init__(
        self,
        generator: Any,
        target_tokens: int | None = None,
        compression_ratio: float = 0.5,
        min_sentences: int = 3,
        use_query_conditioning: bool = True,
    ):
        """
        Initialize LongLLMLingua compressor.

        Args:
            generator: LLM for perplexity estimation.
            target_tokens: Target token count.
            compression_ratio: Target compression ratio.
            min_sentences: Minimum sentences to keep.
            use_query_conditioning: Score relative to query.
        """
        super().__init__(
            target_tokens=target_tokens,
            compression_ratio=compression_ratio,
            min_chunks=1,
        )
        self._generator = generator
        self._min_sentences = min_sentences
        self._use_query_conditioning = use_query_conditioning

    async def _score_sentences(
        self,
        query: str,
        sentences: list[str],
    ) -> list[float]:
        """
        Score sentences by query-conditioned importance.

        Uses LLM to estimate how informative each sentence is
        for answering the query (proxy for perplexity).

        Args:
            query: User query.
            sentences: Sentences to score.

        Returns:
            List of importance scores (higher = more important).
        """
        if not sentences:
            return []

        # Batch score sentences
        prompt = f"""You are scoring sentences for relevance to a query.

Query: {query}

For each sentence below, rate its importance for answering the query.
Score from 0 to 10 where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = directly answers the query

Output ONLY the scores as a comma-separated list of numbers, one per sentence.

Sentences:
"""
        for i, sent in enumerate(sentences[:50]):  # Limit to 50 for context length
            prompt += f"{i + 1}. {sent[:200]}\n"  # Truncate long sentences

        prompt += "\nScores (comma-separated):"

        try:
            # Use fast generation
            result = await self._generator.generate(
                query=prompt,
                context=[],
                max_tokens=200,
                temperature=0.0,
            )

            # Parse scores
            response = result.response.strip()
            scores = []

            # Try to extract numbers
            import re

            numbers = re.findall(r"\d+(?:\.\d+)?", response)

            for i, num in enumerate(numbers[: len(sentences)]):
                try:
                    score = float(num)
                    scores.append(min(10, max(0, score)))  # Clamp to 0-10
                except ValueError:
                    scores.append(5.0)  # Default middle score

            # Pad if needed
            while len(scores) < len(sentences):
                scores.append(5.0)

            return scores

        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}, using fallback")
            # Fallback: position-based (earlier = more important for RAG)
            return [10 - (i * 10 / len(sentences)) for i in range(len(sentences))]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    async def compress(
        self,
        query: str,
        chunks: list[Chunk],
        target_tokens: int | None = None,
        compression_ratio: float | None = None,
    ) -> CompressionResult:
        """
        Compress using LongLLMLingua-style perplexity scoring.

        Args:
            query: Query for importance scoring.
            chunks: Chunks to compress.
            target_tokens: Override target tokens.
            compression_ratio: Override compression ratio.

        Returns:
            CompressionResult with compressed chunks.
        """
        if not chunks:
            return CompressionResult()

        ratio = compression_ratio or self._compression_ratio
        target = target_tokens or self._target_tokens

        original_tokens = self._chunks_to_tokens(chunks)

        if target is None:
            target = int(original_tokens * ratio)

        logger.info(
            f"LongLLMLingua compressing {len(chunks)} chunks: {original_tokens} -> ~{target} tokens"
        )

        # Extract sentences with source tracking
        all_sentences: list[tuple[str, int, int]] = []  # (sentence, chunk_idx, sent_idx)

        for chunk_idx, chunk in enumerate(chunks):
            sentences = self._split_sentences(chunk.content)
            for sent_idx, sentence in enumerate(sentences):
                all_sentences.append((sentence, chunk_idx, sent_idx))

        if not all_sentences:
            return CompressionResult(
                compressed_chunks=chunks,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
            )

        # Score sentences with LLM
        sentence_texts = [s[0] for s in all_sentences]
        scores = await self._score_sentences(query, sentence_texts)

        # Combine with source info
        scored_sentences = [
            (sent, cidx, sidx, score)
            for (sent, cidx, sidx), score in zip(all_sentences, scores, strict=False)
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

        # Group by chunk and reconstruct
        chunk_sentences: dict[int, list[tuple[int, str]]] = {}
        for sentence, chunk_idx, sent_idx, _ in selected_sentences:
            if chunk_idx not in chunk_sentences:
                chunk_sentences[chunk_idx] = []
            chunk_sentences[chunk_idx].append((sent_idx, sentence))

        compressed_chunks = []
        for chunk_idx in sorted(chunk_sentences.keys()):
            original_chunk = chunks[chunk_idx]
            sentences = chunk_sentences[chunk_idx]
            sentences.sort(key=lambda x: x[0])

            compressed_content = " ".join(s[1] for s in sentences)

            compressed_chunk = Chunk(
                id=original_chunk.id,
                content=compressed_content,
                document_id=original_chunk.document_id,
                metadata={
                    **original_chunk.metadata,
                    "compressed": True,
                    "compression_method": "longllmlingua",
                    "original_length": len(original_chunk.content),
                    "sentences_kept": len(sentences),
                },
                embedding=original_chunk.embedding,
            )
            compressed_chunks.append(compressed_chunk)

        compressed_tokens = self._chunks_to_tokens(compressed_chunks)
        actual_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        logger.info(
            f"LongLLMLingua compression: {original_tokens} -> {compressed_tokens} tokens "
            f"({(1 - actual_ratio) * 100:.1f}% reduction)"
        )

        return CompressionResult(
            compressed_chunks=compressed_chunks,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=actual_ratio,
            metadata={
                "method": "longllmlingua",
                "sentences_total": len(all_sentences),
                "sentences_selected": len(selected_sentences),
                "chunks_with_content": len(compressed_chunks),
            },
        )
