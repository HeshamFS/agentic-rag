"""
Base chunking interface and utilities.

Defines the common interface for all chunking strategies.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Any

from agentic_rag.core.models import Chunk, Document


class BaseChunker(ABC):
    """
    Base class for all chunking strategies.

    Defines the interface for splitting documents into chunks.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens/characters.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split document into chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of chunks.
        """
        ...

    def chunk_batch(self, documents: list[Document]) -> list[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: Documents to chunk.

        Returns:
            All chunks from all documents.
        """
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks

    def _create_chunk(
        self,
        content: str,
        document: Document,
        index: int,
        total_chunks: int,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """
        Create a chunk from content.

        Args:
            content: Chunk content.
            document: Source document.
            index: Chunk index within document.
            total_chunks: Total chunks from this document.
            metadata: Additional metadata.

        Returns:
            Chunk object.
        """
        chunk_metadata = {
            "source": document.source,
            "chunk_index": index,
            "total_chunks": total_chunks,
            **(document.metadata or {}),
            **(metadata or {}),
        }

        return Chunk(
            id=str(uuid.uuid4()),
            content=content,
            document_id=document.id,
            metadata=chunk_metadata,
            embedding=None,
        )


class FixedSizeChunker(BaseChunker):
    """
    Simple fixed-size chunking with overlap.

    Splits text into chunks of fixed size with configurable overlap.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = " ",
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Chunk size in characters.
            chunk_overlap: Overlap in characters.
            separator: Word separator.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separator = separator

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split document into fixed-size chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of chunks.
        """
        text = document.content
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        # First pass: count total chunks
        temp_start = 0
        total_chunks = 0
        while temp_start < len(text):
            temp_start += self.chunk_size - self.chunk_overlap
            total_chunks += 1

        # Second pass: create chunks
        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at word boundary
            if end < len(text):
                # Look for separator in last 20% of chunk
                search_start = start + int(self.chunk_size * 0.8)
                last_sep = text.rfind(self.separator, search_start, end)
                if last_sep > search_start:
                    end = last_sep + 1

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunk = self._create_chunk(
                    content=chunk_content,
                    document=document,
                    index=chunk_index,
                    total_chunks=total_chunks,
                    metadata={"start_char": start, "end_char": end},
                )
                chunks.append(chunk)
                chunk_index += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks


class SentenceChunker(BaseChunker):
    """
    Sentence-based chunking.

    Groups sentences until reaching target chunk size.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 1,  # Overlap in sentences
        min_chunk_size: int = 100,
    ):
        """
        Initialize sentence chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Number of sentences to overlap.
            min_chunk_size: Minimum chunk size in characters.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.min_chunk_size = min_chunk_size

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        import re

        # Simple sentence splitting
        # In production, use spacy or nltk
        sentence_endings = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split document into sentence-based chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of chunks.
        """
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        chunks = []
        current_sentences: list[str] = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # Check if adding this sentence exceeds limit
            if current_length + sentence_len > self.chunk_size and current_sentences:
                # Create chunk
                chunk_content = " ".join(current_sentences)
                chunks.append(
                    self._create_chunk(
                        content=chunk_content,
                        document=document,
                        index=chunk_index,
                        total_chunks=0,  # Will update later
                        metadata={"num_sentences": len(current_sentences)},
                    )
                )
                chunk_index += 1

                # Keep overlap sentences
                overlap_sentences = (
                    current_sentences[-self.chunk_overlap :] if self.chunk_overlap > 0 else []
                )
                current_sentences = overlap_sentences
                current_length = sum(len(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_length += sentence_len

        # Create final chunk
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(
                    self._create_chunk(
                        content=chunk_content,
                        document=document,
                        index=chunk_index,
                        total_chunks=0,
                        metadata={"num_sentences": len(current_sentences)},
                    )
                )

        # Update total chunks count
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses simple word-based estimation.
    In production, use actual tokenizer.

    Args:
        text: Text to estimate.

    Returns:
        Estimated token count.
    """
    # Rough estimation: ~0.75 tokens per word for English
    words = len(text.split())
    return int(words * 1.3)
