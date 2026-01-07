"""
Context compression for RAG pipelines.

Reduces token costs by selecting the most relevant parts of
retrieved context before passing to the LLM for generation.

Compression methods:
- ExtractiveCompressor: Fast reranker-based sentence selection
- LongLLMLinguaCompressor: LLM-based importance scoring

Typical token savings: 50-70% with minimal quality impact.
"""

from agentic_rag.compression.base import BaseCompressor, CompressionResult
from agentic_rag.compression.extractive import ExtractiveCompressor
from agentic_rag.compression.longllmlingua import LongLLMLinguaCompressor

__all__ = [
    # Base
    "BaseCompressor",
    "CompressionResult",
    # Implementations
    "ExtractiveCompressor",
    "LongLLMLinguaCompressor",
]
