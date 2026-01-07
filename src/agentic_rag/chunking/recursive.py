"""
Recursive character text splitter.

Classic chunking approach that recursively splits text
using a hierarchy of separators.
"""

from typing import Any

from agentic_rag.chunking.base import BaseChunker
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, Document


class RecursiveChunker(BaseChunker):
    """
    Recursive character text splitter.

    Splits text using a hierarchy of separators, trying
    larger separators first and falling back to smaller ones.

    Default separator hierarchy:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (. ! ?)
    4. Spaces (words)
    5. Characters (last resort)
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        length_function: Any = None,
        settings: Settings | None = None,
    ):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Target chunk size.
            chunk_overlap: Overlap between chunks.
            separators: Custom separator hierarchy.
            length_function: Function to measure text length.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._chunk_size = chunk_size or self._settings.default_chunk_size
        self._chunk_overlap = chunk_overlap or self._settings.default_chunk_overlap
        self._separators = separators or self.DEFAULT_SEPARATORS
        self._length_function = length_function or len

    async def chunk(self, document: Document, **kwargs: Any) -> list[Chunk]:
        """
        Split document into chunks.

        Args:
            document: Document to chunk.
            **kwargs: Additional parameters.

        Returns:
            List of chunks.
        """
        text = document.content
        chunks = self._split_text(text)

        return [
            Chunk(
                content=chunk_text,
                document_id=document.id,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "chunk_method": "recursive",
                },
            )
            for i, chunk_text in enumerate(chunks)
        ]

    def _split_text(self, text: str) -> list[str]:
        """
        Recursively split text.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        return self._split_text_recursive(text, self._separators)

    def _split_text_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """
        Recursively split text with separator fallback.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of text chunks.
        """
        final_chunks: list[str] = []
        separator = separators[-1]  # Default to smallest
        new_separators: list[str] = []

        # Find the best separator that exists in text
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # Split by character

        # Process each split
        good_splits: list[str] = []

        for split in splits:
            # Add separator back (except for empty separator)
            if separator and split != splits[-1]:
                split = split + separator

            if self._length_function(split) <= self._chunk_size:
                good_splits.append(split)
            elif new_separators:
                # Recurse with smaller separators
                sub_chunks = self._split_text_recursive(split, new_separators)
                final_chunks.extend(sub_chunks)
            else:
                # Can't split further, add as-is
                good_splits.append(split)

        # Merge good splits into chunks with overlap
        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """
        Merge splits into chunks with overlap.

        Args:
            splits: List of small text pieces.

        Returns:
            List of merged chunks.
        """
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for split in splits:
            split_length = self._length_function(split)

            if current_length + split_length > self._chunk_size and current_chunk:
                # Save current chunk
                chunks.append("".join(current_chunk))

                # Start new chunk with overlap
                overlap_length = 0
                overlap_splits: list[str] = []

                for prev_split in reversed(current_chunk):
                    if overlap_length + self._length_function(prev_split) > self._chunk_overlap:
                        break
                    overlap_splits.insert(0, prev_split)
                    overlap_length += self._length_function(prev_split)

                current_chunk = overlap_splits
                current_length = overlap_length

            current_chunk.append(split)
            current_length += split_length

        # Add final chunk
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks


class TokenRecursiveChunker(RecursiveChunker):
    """
    Recursive chunker that measures by tokens instead of characters.

    More accurate for LLM context limits.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        tokenizer: str = "cl100k_base",
        settings: Settings | None = None,
    ):
        """
        Initialize token-based chunker.

        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap in tokens.
            separators: Separator hierarchy.
            tokenizer: Tiktoken tokenizer name.
            settings: Settings instance.
        """
        try:
            import tiktoken

            self._tokenizer = tiktoken.get_encoding(tokenizer)
        except ImportError:
            raise ImportError("tiktoken required: pip install tiktoken")

        super().__init__(
            chunk_size=chunk_size or 512,
            chunk_overlap=chunk_overlap or 50,
            separators=separators,
            length_function=self._count_tokens,
            settings=settings,
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))
