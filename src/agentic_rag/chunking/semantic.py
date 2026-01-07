"""
Semantic chunking using embedding-based topic detection.

Breaks text at natural topic boundaries using embedding similarity,
resulting in more coherent chunks than fixed-size splitting.
"""

import numpy as np

from agentic_rag.chunking.base import BaseChunker
from agentic_rag.core.models import Chunk, Document
from agentic_rag.core.protocols import Embedder


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using embedding similarity.

    Process:
    1. Split document into sentences
    2. Embed each sentence
    3. Calculate similarity between consecutive sentences
    4. Break at points where similarity drops below threshold

    Benefits:
    - Preserves semantic coherence
    - Chunks contain related content
    - Better retrieval quality
    """

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 512,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        window_size: int = 3,
    ):
        """
        Initialize semantic chunker.

        Args:
            embedder: Embedding model for similarity calculation.
            chunk_size: Target chunk size in characters.
            similarity_threshold: Break when similarity drops below this.
            min_chunk_size: Minimum chunk size.
            max_chunk_size: Maximum chunk size.
            window_size: Sentences to average for smoothing.
        """
        super().__init__(chunk_size, 0)  # No overlap - semantic breaks handle continuity
        self._embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.window_size = window_size

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        import re

        # Comprehensive sentence splitting
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+"
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            v1: First vector.
            v2: Second vector.

        Returns:
            Cosine similarity score.
        """
        a = np.array(v1)
        b = np.array(v2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    async def chunk_async(self, document: Document) -> list[Chunk]:
        """
        Divide a document into semantically coherent chunks using embedding similarity.

        The process follows these steps:
        1. Sentence Splitting: Uses regex to find natural sentence boundaries.
        2. Embedding: Generates vectors for each individual sentence.
        3. Similarity Analysis: Calculates cosine similarity between adjacent sentences.
        4. Smoothing: Applies a moving average window to reduce noise in similarity scores.
        5. Breakpoint Detection: Identifies topic shifts where similarity drops below a threshold.
        6. Chunk Assembly: Groups sentences into chunks within the target size limits.

        Args:
            document: The source Document object to be chunked.

        Returns:
            A list of Chunk objects with semantic metadata.
        """
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        if len(sentences) <= 2:
            # Too short for semantic chunking
            return [
                self._create_chunk(
                    content=document.content,
                    document=document,
                    index=0,
                    total_chunks=1,
                )
            ]

        # Embed all sentences
        embeddings = await self._embedder.embed_batch(sentences)

        # Calculate similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Apply smoothing with window
        smoothed_similarities = self._smooth_similarities(similarities)

        # Find breakpoints
        breakpoints = self._find_breakpoints(
            sentences,
            smoothed_similarities,
        )

        # Create chunks from breakpoints
        chunks = self._create_chunks_from_breakpoints(
            sentences,
            breakpoints,
            document,
        )

        return chunks

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Synchronous wrapper for semantic chunking.

        Note: For full async support, use chunk_async() directly.

        Args:
            document: Document to chunk.

        Returns:
            List of chunks.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
            # If we're in an async context, create a new task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.chunk_async(document))
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self.chunk_async(document))

    def _smooth_similarities(self, similarities: list[float]) -> list[float]:
        """
        Apply moving average smoothing to similarities.

        Args:
            similarities: Raw similarity scores.

        Returns:
            Smoothed similarity scores.
        """
        if len(similarities) <= self.window_size:
            return similarities

        smoothed = []
        half_window = self.window_size // 2

        for i in range(len(similarities)):
            start = max(0, i - half_window)
            end = min(len(similarities), i + half_window + 1)
            window = similarities[start:end]
            smoothed.append(sum(window) / len(window))

        return smoothed

    def _find_breakpoints(
        self,
        sentences: list[str],
        similarities: list[float],
    ) -> list[int]:
        """
        Find optimal breakpoints for chunking.

        Args:
            sentences: List of sentences.
            similarities: Similarity scores between consecutive sentences.

        Returns:
            List of indices where chunks should break.
        """
        breakpoints = [0]  # Always start at beginning
        current_length = 0

        for i, sentence in enumerate(sentences[:-1]):
            current_length += len(sentence) + 1  # +1 for space

            # Check if we should break here
            should_break = False

            # Break if similarity is below threshold
            if similarities[i] < self.similarity_threshold:
                should_break = True

            # Force break if max size exceeded
            if current_length > self.max_chunk_size:
                should_break = True

            # Don't break if chunk would be too small
            if should_break and current_length < self.min_chunk_size:
                should_break = False

            if should_break:
                breakpoints.append(i + 1)
                current_length = 0

        return breakpoints

    def _create_chunks_from_breakpoints(
        self,
        sentences: list[str],
        breakpoints: list[int],
        document: Document,
    ) -> list[Chunk]:
        """
        Create chunks from breakpoints.

        Args:
            sentences: All sentences.
            breakpoints: Indices where chunks start.
            document: Source document.

        Returns:
            List of chunks.
        """
        chunks = []
        breakpoints = breakpoints + [len(sentences)]  # Add end

        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_content = " ".join(chunk_sentences)

            if chunk_content.strip():
                chunk = self._create_chunk(
                    content=chunk_content,
                    document=document,
                    index=i,
                    total_chunks=len(breakpoints) - 1,
                    metadata={
                        "start_sentence": start_idx,
                        "end_sentence": end_idx,
                        "num_sentences": len(chunk_sentences),
                        "chunking_method": "semantic",
                    },
                )
                chunks.append(chunk)

        return chunks


class PercentileSemanticChunker(SemanticChunker):
    """
    Semantic chunker using percentile-based breakpoint detection.

    Instead of a fixed threshold, breaks at points where similarity
    drops below the Nth percentile of all similarities.
    """

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 512,
        breakpoint_percentile: float = 25,  # Break at 25th percentile and below
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        """
        Initialize percentile semantic chunker.

        Args:
            embedder: Embedding model.
            chunk_size: Target chunk size.
            breakpoint_percentile: Percentile below which to break.
            min_chunk_size: Minimum chunk size.
            max_chunk_size: Maximum chunk size.
        """
        super().__init__(
            embedder=embedder,
            chunk_size=chunk_size,
            similarity_threshold=0.5,  # Will be overridden
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        self.breakpoint_percentile = breakpoint_percentile

    def _find_breakpoints(
        self,
        sentences: list[str],
        similarities: list[float],
    ) -> list[int]:
        """
        Find breakpoints using percentile threshold.

        Args:
            sentences: List of sentences.
            similarities: Similarity scores.

        Returns:
            List of breakpoint indices.
        """
        if not similarities:
            return [0]

        # Calculate dynamic threshold based on percentile
        threshold = float(np.percentile(similarities, self.breakpoint_percentile))

        breakpoints = [0]
        current_length = 0

        for i, sentence in enumerate(sentences[:-1]):
            current_length += len(sentence) + 1

            should_break = False

            if similarities[i] < threshold:
                should_break = True

            if current_length > self.max_chunk_size:
                should_break = True

            if should_break and current_length < self.min_chunk_size:
                should_break = False

            if should_break:
                breakpoints.append(i + 1)
                current_length = 0

        return breakpoints
