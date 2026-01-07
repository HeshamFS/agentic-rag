"""
ColBERT (Contextualized Late Interaction over BERT) Reranker.

ColBERT uses "late interaction" - documents are pre-encoded offline,
only queries are encoded at runtime. This provides the optimal balance
between speed and accuracy.

Key features:
- Pre-compute document embeddings (token-level)
- Only encode queries at runtime
- MaxSim scoring for relevance
- 15-40% accuracy improvement over semantic search alone

Reference:
- "ColBERT: Efficient and Effective Passage Search" (Khattab & Zaharia, 2020)
- "Jina-ColBERT-v2" achieves 61.94 nDCG@10 on BEIR
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from agentic_rag.core.models import Chunk
from agentic_rag.reranking.base import BaseReranker, RerankResult


class ColBERTScore(BaseModel):
    """Score from ColBERT late interaction."""

    chunk_id: str
    score: float = Field(description="MaxSim score")
    token_scores: list[float] = Field(
        default_factory=list, description="Per-query-token max similarity scores"
    )


class ColBERTReranker(BaseReranker):
    """
    ColBERT-style late interaction reranker.

    Late interaction means:
    1. Document tokens are embedded offline (pre-computed)
    2. Query tokens are embedded at runtime
    3. Scoring uses MaxSim: for each query token, find max similarity
       with any document token, then sum

    This is much faster than cross-encoders while maintaining high accuracy.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-colbert-v2",
        device: str = "cuda",
        max_query_length: int = 32,
        max_doc_length: int = 512,
    ):
        """
        Initialize ColBERT reranker.

        Args:
            model_name: ColBERT model to use.
            device: Device for inference.
            max_query_length: Maximum query tokens.
            max_doc_length: Maximum document tokens.
        """
        super().__init__()
        self._model_name_str = model_name
        self._device = device
        self._max_query_length = max_query_length
        self._max_doc_length = max_doc_length

        self._model = None
        self._tokenizer = None
        self._initialized = False

    @property
    def model_name(self) -> str:
        """Get the reranker model name."""
        return self._model_name_str

    def _ensure_initialized(self) -> None:
        """Lazy initialization of model."""
        if self._initialized:
            return

        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_str)
        self._model = AutoModel.from_pretrained(
            self._model_name_str,
            trust_remote_code=True,  # For Jina models
        )
        self._model.to(self._device)
        self._model.eval()
        self._initialized = True

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RerankResult:
        """
        Rerank chunks using ColBERT late interaction.

        Args:
            query: User query.
            chunks: Chunks to rerank.
            top_k: Number of results to return.

        Returns:
            RerankResult with reranked chunks, scores, and original indices.
        """
        if not chunks:
            return RerankResult(chunks=[], scores=[], original_indices=[])

        self._ensure_initialized()

        # Get query token embeddings
        query_embeddings = self._encode_query(query)

        # Get document token embeddings and scores
        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            doc_embeddings = self._encode_document(chunk.content)
            score, _ = self._compute_maxsim(query_embeddings, doc_embeddings)
            scored_chunks.append((chunk, score, idx))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k
        if top_k:
            scored_chunks = scored_chunks[:top_k]

        return RerankResult(
            chunks=[sc[0] for sc in scored_chunks],
            scores=[sc[1] for sc in scored_chunks],
            original_indices=[sc[2] for sc in scored_chunks],
        )

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode query into token embeddings.

        Returns:
            Array of shape (num_tokens, embedding_dim)
        """
        import torch

        inputs = self._tokenizer(
            query,
            max_length=self._max_query_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = outputs.last_hidden_state[0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            # Convert to float32 for numpy compatibility
            embeddings = embeddings.float()

        return embeddings.cpu().numpy()

    def _encode_document(self, text: str) -> np.ndarray:
        """
        Encode document into token embeddings.

        Returns:
            Array of shape (num_tokens, embedding_dim)
        """
        import torch

        inputs = self._tokenizer(
            text,
            max_length=self._max_doc_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = outputs.last_hidden_state[0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            # Convert to float32 for numpy compatibility
            embeddings = embeddings.float()

        return embeddings.cpu().numpy()

    def _compute_maxsim(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> tuple[float, list[float]]:
        """
        Compute ColBERT MaxSim score.

        For each query token, find the maximum similarity with any
        document token, then sum all these max values.

        Args:
            query_embeddings: (num_query_tokens, dim)
            doc_embeddings: (num_doc_tokens, dim)

        Returns:
            Tuple of (total_score, per_token_scores)
        """
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)
        max_similarities = np.max(similarity_matrix, axis=1)
        total_score = float(np.sum(max_similarities))
        return total_score, max_similarities.tolist()

    def precompute_embeddings(self, chunks: list[Chunk]) -> dict[str, np.ndarray]:
        """
        Pre-compute and cache document embeddings.

        Args:
            chunks: Chunks to embed.

        Returns:
            Dict mapping chunk_id to token embeddings.
        """
        self._ensure_initialized()

        if self._model is None:
            return {}

        embeddings = {}
        for chunk in chunks:
            embeddings[chunk.id] = self._encode_document(chunk.content)

        return embeddings


class CachedColBERTReranker(ColBERTReranker):
    """
    ColBERT reranker with embedding cache.

    Pre-computes and caches document embeddings for faster reranking.
    """

    def __init__(self, cache_dir: str | None = None, **kwargs: Any):
        """
        Initialize with cache.

        Args:
            cache_dir: Directory for embedding cache.
            **kwargs: Arguments for ColBERTReranker.
        """
        super().__init__(**kwargs)
        self._cache_dir = cache_dir
        self._embedding_cache: dict[str, np.ndarray] = {}

    def cache_chunks(self, chunks: list[Chunk]) -> None:
        """
        Pre-compute and cache embeddings for chunks.

        Args:
            chunks: Chunks to cache.
        """
        self._ensure_initialized()

        for chunk in chunks:
            if chunk.id not in self._embedding_cache:
                self._embedding_cache[chunk.id] = self._encode_document(chunk.content)

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RerankResult:
        """Rerank using cached embeddings when available."""
        if not chunks:
            return RerankResult(chunks=[], scores=[], original_indices=[])

        self._ensure_initialized()

        query_embeddings = self._encode_query(query)

        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            if chunk.id in self._embedding_cache:
                doc_embeddings = self._embedding_cache[chunk.id]
            else:
                doc_embeddings = self._encode_document(chunk.content)
                self._embedding_cache[chunk.id] = doc_embeddings

            score, _ = self._compute_maxsim(query_embeddings, doc_embeddings)
            scored_chunks.append((chunk, score, idx))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            scored_chunks = scored_chunks[:top_k]

        return RerankResult(
            chunks=[sc[0] for sc in scored_chunks],
            scores=[sc[1] for sc in scored_chunks],
            original_indices=[sc[2] for sc in scored_chunks],
        )

    def save_cache(self, path: str) -> None:
        """Save embedding cache to disk."""
        import pickle
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._embedding_cache, f)

    def load_cache(self, path: str) -> None:
        """Load embedding cache from disk."""
        import pickle
        from pathlib import Path

        if Path(path).exists():
            with open(path, "rb") as f:
                self._embedding_cache = pickle.load(f)


class LightweightColBERT(BaseReranker):
    """
    Lightweight ColBERT-style reranker without heavy dependencies.

    Uses the existing embedder to simulate late interaction scoring.
    Good for when you don't want to load a separate ColBERT model.
    """

    def __init__(
        self,
        embedder: Any,  # Embedder protocol
        num_query_expansions: int = 3,
    ):
        """
        Initialize lightweight ColBERT.

        Args:
            embedder: Embedding model (e.g., Qwen3Embedder).
            num_query_expansions: Number of query variations to try.
        """
        super().__init__()
        self._embedder = embedder
        self._num_expansions = num_query_expansions

    @property
    def model_name(self) -> str:
        """Get the reranker model name."""
        return "lightweight-colbert"

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RerankResult:
        """
        Rerank using lightweight late interaction approximation.

        Simulates token-level matching by:
        1. Splitting query into key phrases
        2. Computing similarity for each phrase
        3. Using max-aggregation
        """
        if not chunks:
            return RerankResult(chunks=[], scores=[], original_indices=[])

        # Split query into phrases (simulate query tokens)
        query_phrases = self._extract_phrases(query)

        # Get embeddings for all phrases
        phrase_embeddings = []
        for phrase in query_phrases:
            emb = await self._embedder.embed_text(phrase)
            phrase_embeddings.append(emb)

        scored_chunks = []
        for idx, chunk in enumerate(chunks):
            # Get chunk embedding (or use existing)
            if chunk.embedding:
                chunk_embedding = chunk.embedding
            else:
                chunk_embedding = await self._embedder.embed_text(chunk.content)

            # Compute max similarity across phrases
            phrase_scores = []
            for phrase_emb in phrase_embeddings:
                sim = self._cosine_similarity(phrase_emb, chunk_embedding)
                phrase_scores.append(sim)

            # Average of top phrase scores (soft MaxSim)
            phrase_scores.sort(reverse=True)
            score = sum(phrase_scores[:3]) / min(3, len(phrase_scores))

            scored_chunks.append((chunk, score, idx))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            scored_chunks = scored_chunks[:top_k]

        return RerankResult(
            chunks=[sc[0] for sc in scored_chunks],
            scores=[sc[1] for sc in scored_chunks],
            original_indices=[sc[2] for sc in scored_chunks],
        )

    def _extract_phrases(self, query: str) -> list[str]:
        """Extract key phrases from query."""
        import re

        # Split on common separators
        phrases = re.split(r"[,;:]|\band\b|\bor\b", query.lower())
        phrases = [p.strip() for p in phrases if p.strip()]

        # Also include the full query
        if query not in phrases:
            phrases.insert(0, query)

        # Include individual important words
        words = query.split()
        important_words = [w for w in words if len(w) > 4]
        phrases.extend(important_words[:5])

        return phrases[:10]  # Limit to 10 phrases

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)

        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))
