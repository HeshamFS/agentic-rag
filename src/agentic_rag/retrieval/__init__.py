"""Retrieval components including dense, sparse, hybrid, HyDE, and RRF fusion."""

from agentic_rag.retrieval.base import (
    BaseRetriever,
    deduplicate_chunks,
    normalize_scores,
)
from agentic_rag.retrieval.dense import DenseRetriever
from agentic_rag.retrieval.fusion import (
    RRFFusion,
    linear_combination_fusion,
    reciprocal_rank_fusion,
)
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.hyde import (
    AdaptiveHyDERetriever,
    HyDERetriever,
)
from agentic_rag.retrieval.multi_query import (
    MultiQueryRetriever,
    QueryDecomposer,
    StepBackRetriever,
)
from agentic_rag.retrieval.sparse import BM25Index, SparseRetriever

__all__ = [
    # Base
    "BaseRetriever",
    "normalize_scores",
    "deduplicate_chunks",
    # Dense
    "DenseRetriever",
    # Sparse
    "SparseRetriever",
    "BM25Index",
    # Hybrid
    "HybridRetriever",
    # Fusion
    "RRFFusion",
    "linear_combination_fusion",
    "reciprocal_rank_fusion",
    # HyDE
    "HyDERetriever",
    "AdaptiveHyDERetriever",
    # Multi-Query
    "MultiQueryRetriever",
    "QueryDecomposer",
    "StepBackRetriever",
]
