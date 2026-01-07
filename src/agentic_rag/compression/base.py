"""
Base classes for context compression.

Context compression reduces token costs by selecting/filtering
the most relevant parts of retrieved chunks before generation.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.core.models import Chunk


class CompressionResult(BaseModel):
    """Result of context compression."""

    compressed_chunks: list[Chunk] = Field(
        default_factory=list,
        description="Compressed chunks with selected content",
    )
    original_tokens: int = Field(
        default=0,
        description="Token count before compression",
    )
    compressed_tokens: int = Field(
        default=0,
        description="Token count after compression",
    )
    compression_ratio: float = Field(
        default=1.0,
        description="Ratio of compressed/original tokens",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional compression metadata",
    )

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return self.original_tokens - self.compressed_tokens

    @property
    def savings_percent(self) -> float:
        """Percentage of tokens saved."""
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.compression_ratio) * 100


class BaseCompressor(ABC):
    """
    Abstract base class for context compressors.

    Compressors reduce the size of retrieved context while preserving
    the most relevant information for answering the query.

    Compression strategies:
    - Extractive: Select most relevant sentences/passages
    - Sentence: Score and filter at sentence level
    - LongLLMLingua: Perplexity-based token importance scoring
    """

    def __init__(
        self,
        target_tokens: int | None = None,
        compression_ratio: float = 0.5,
        min_chunks: int = 1,
    ):
        """
        Initialize compressor.

        Args:
            target_tokens: Target token count after compression.
            compression_ratio: Target ratio (0.5 = 50% reduction).
            min_chunks: Minimum chunks to preserve.
        """
        self._target_tokens = target_tokens
        self._compression_ratio = compression_ratio
        self._min_chunks = min_chunks

    @abstractmethod
    async def compress(
        self,
        query: str,
        chunks: list[Chunk],
        target_tokens: int | None = None,
        compression_ratio: float | None = None,
    ) -> CompressionResult:
        """
        Compress context chunks.

        Args:
            query: User query for relevance scoring.
            chunks: Retrieved chunks to compress.
            target_tokens: Override target token count.
            compression_ratio: Override compression ratio.

        Returns:
            CompressionResult with compressed chunks and stats.
        """
        pass

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: 4 chars per token)."""
        return len(text) // 4

    def _chunks_to_tokens(self, chunks: list[Chunk]) -> int:
        """Estimate total tokens in chunks."""
        return sum(self._estimate_tokens(c.content) for c in chunks)
