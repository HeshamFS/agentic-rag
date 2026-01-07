"""
Reranking module for RAG Optimizer.

Provides cross-encoder rerankers, ColBERT late interaction, and attention-aware reordering.
"""

from agentic_rag.reranking.base import BaseReranker, RerankResult
from agentic_rag.reranking.colbert import (
    CachedColBERTReranker,
    ColBERTReranker,
    ColBERTScore,
    LightweightColBERT,
)
from agentic_rag.reranking.cross_encoder import (
    BGEReranker,
    CrossEncoderReranker,
    MxbaiReranker,
)
from agentic_rag.reranking.jina_reranker import JinaReranker, JinaRerankerV3
from agentic_rag.reranking.lost_middle import (
    InterleavedReorderer,
    LostInMiddleReorderer,
    apply_lost_in_middle,
    reorder_for_attention,
)

__all__ = [
    # Base
    "BaseReranker",
    "RerankResult",
    # Jina
    "JinaReranker",
    "JinaRerankerV3",
    # Cross-Encoder
    "CrossEncoderReranker",
    "BGEReranker",
    "MxbaiReranker",
    # ColBERT (Late Interaction)
    "ColBERTReranker",
    "CachedColBERTReranker",
    "LightweightColBERT",
    "ColBERTScore",
    # Lost-in-Middle
    "LostInMiddleReorderer",
    "InterleavedReorderer",
    "reorder_for_attention",
    "apply_lost_in_middle",
]
