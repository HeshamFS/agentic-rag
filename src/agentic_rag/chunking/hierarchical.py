"""
Hierarchical chunking with parent-child structure.

Creates a tree of chunks where child chunks contain specific details
and parent chunks contain broader context.
"""

import uuid

from agentic_rag.chunking.base import BaseChunker, SentenceChunker
from agentic_rag.core.models import Chunk, Document


class HierarchicalChunk(Chunk):
    """
    Chunk with hierarchical relationships.

    Extends base Chunk with parent/child references.
    """

    parent_id: str | None = None
    child_ids: list[str] = []
    level: int = 0  # 0 = root/largest, higher = smaller/more specific


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunker creating multi-level chunk structure.

    Creates:
    - Level 0: Large chunks (e.g., 2000 chars) for broad context
    - Level 1: Medium chunks (e.g., 512 chars) for standard retrieval
    - Level 2: Small chunks (e.g., 128 chars) for precise matching

    Benefits:
    - Retrieve at appropriate granularity
    - Parent chunks provide context
    - Child chunks provide precision
    """

    def __init__(
        self,
        levels: list[int] | None = None,
        base_chunker_class: type[BaseChunker] | None = None,
    ):
        """
        Initialize hierarchical chunker.

        Args:
            levels: Chunk sizes for each level (largest to smallest).
            base_chunker_class: Chunker class to use at each level.
        """
        # Default: 3 levels
        self.levels = levels or [2000, 512, 128]
        self._base_chunker_class = base_chunker_class or SentenceChunker

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Create hierarchical chunks.

        Args:
            document: Document to chunk.

        Returns:
            All chunks from all levels with parent-child links.
        """
        all_chunks: list[HierarchicalChunk] = []

        # Create top-level (largest) chunks first
        level_0_chunker = self._base_chunker_class(
            chunk_size=self.levels[0],
            chunk_overlap=0,
        )
        level_0_chunks = self._create_level_chunks(
            document=document,
            chunker=level_0_chunker,
            level=0,
            parent_chunks=None,
        )
        all_chunks.extend(level_0_chunks)

        # Create subsequent levels
        parent_chunks = level_0_chunks
        for level_idx, chunk_size in enumerate(self.levels[1:], start=1):
            chunker = self._base_chunker_class(
                chunk_size=chunk_size,
                chunk_overlap=0,
            )

            level_chunks = []
            for parent in parent_chunks:
                # Create a pseudo-document from parent content
                parent_doc = Document(
                    content=parent.content,
                    source=document.source,
                    metadata=document.metadata,
                )

                children = self._create_level_chunks(
                    document=parent_doc,
                    chunker=chunker,
                    level=level_idx,
                    parent_chunks=[parent],
                )

                # Link children to parent
                for child in children:
                    child.parent_id = parent.id
                    parent.child_ids.append(child.id)

                level_chunks.extend(children)

            all_chunks.extend(level_chunks)
            parent_chunks = level_chunks

        return all_chunks

    def _create_level_chunks(
        self,
        document: Document,
        chunker: BaseChunker,
        level: int,
        parent_chunks: list[HierarchicalChunk] | None,
    ) -> list[HierarchicalChunk]:
        """
        Create chunks at a specific level.

        Args:
            document: Document to chunk.
            chunker: Chunker to use.
            level: Hierarchy level.
            parent_chunks: Parent chunks (if any).

        Returns:
            Chunks at this level.
        """
        base_chunks = chunker.chunk(document)

        hierarchical_chunks = []
        for chunk in base_chunks:
            h_chunk = HierarchicalChunk(
                id=str(uuid.uuid4()),
                content=chunk.content,
                document_id=chunk.document_id,
                metadata={
                    **chunk.metadata,
                    "level": level,
                    "hierarchy_level": level,
                    "level_size": self.levels[level] if level < len(self.levels) else 0,
                },
                embedding=None,
                parent_id=None,
                child_ids=[],
                level=level,
            )
            hierarchical_chunks.append(h_chunk)

        return hierarchical_chunks

    def get_chunks_at_level(
        self,
        chunks: list[HierarchicalChunk],
        level: int,
    ) -> list[HierarchicalChunk]:
        """
        Get all chunks at a specific level.

        Args:
            chunks: All chunks.
            level: Level to filter by.

        Returns:
            Chunks at specified level.
        """
        return [c for c in chunks if c.level == level]

    def get_parent(
        self,
        chunk: HierarchicalChunk,
        all_chunks: list[HierarchicalChunk],
    ) -> HierarchicalChunk | None:
        """
        Get parent chunk.

        Args:
            chunk: Child chunk.
            all_chunks: All chunks.

        Returns:
            Parent chunk or None.
        """
        if not chunk.parent_id:
            return None

        for c in all_chunks:
            if c.id == chunk.parent_id:
                return c
        return None

    def get_children(
        self,
        chunk: HierarchicalChunk,
        all_chunks: list[HierarchicalChunk],
    ) -> list[HierarchicalChunk]:
        """
        Get child chunks.

        Args:
            chunk: Parent chunk.
            all_chunks: All chunks.

        Returns:
            List of child chunks.
        """
        return [c for c in all_chunks if c.id in chunk.child_ids]

    def get_with_context(
        self,
        chunk: HierarchicalChunk,
        all_chunks: list[HierarchicalChunk],
        include_parent: bool = True,
        include_children: bool = False,
    ) -> str:
        """
        Get chunk content with hierarchical context.

        Args:
            chunk: Target chunk.
            all_chunks: All chunks.
            include_parent: Include parent content.
            include_children: Include children content.

        Returns:
            Combined content string.
        """
        parts = []

        if include_parent:
            parent = self.get_parent(chunk, all_chunks)
            if parent:
                parts.append(f"[Parent Context]\n{parent.content}\n")

        parts.append(f"[Content]\n{chunk.content}")

        if include_children:
            children = self.get_children(chunk, all_chunks)
            if children:
                children_content = "\n\n".join(c.content for c in children)
                parts.append(f"\n[Details]\n{children_content}")

        return "\n".join(parts)


class SmallToBigRetriever:
    """
    Retrieval strategy using hierarchical chunks.

    Retrieves small chunks for precision, then expands to parent
    chunks for context before passing to LLM.
    """

    def __init__(self, hierarchical_chunker: HierarchicalChunker):
        """
        Initialize small-to-big retriever.

        Args:
            hierarchical_chunker: Chunker that created the chunks.
        """
        self._chunker = hierarchical_chunker

    def expand_to_parent(
        self,
        retrieved_chunks: list[HierarchicalChunk],
        all_chunks: list[HierarchicalChunk],
    ) -> list[HierarchicalChunk]:
        """
        Expand retrieved chunks to their parents.

        Args:
            retrieved_chunks: Initially retrieved small chunks.
            all_chunks: All hierarchical chunks.

        Returns:
            Parent chunks (deduplicated).
        """
        seen_ids: set[str] = set()
        parent_chunks: list[HierarchicalChunk] = []

        for chunk in retrieved_chunks:
            parent = self._chunker.get_parent(chunk, all_chunks)
            if parent and parent.id not in seen_ids:
                seen_ids.add(parent.id)
                parent_chunks.append(parent)

        return parent_chunks

    def get_context_window(
        self,
        retrieved_chunks: list[HierarchicalChunk],
        all_chunks: list[HierarchicalChunk],
        max_tokens: int = 4000,
    ) -> str:
        """
        Build context window from retrieved chunks.

        Args:
            retrieved_chunks: Retrieved chunks.
            all_chunks: All chunks.
            max_tokens: Maximum context size.

        Returns:
            Combined context string.
        """
        # Expand to parents
        parents = self.expand_to_parent(retrieved_chunks, all_chunks)

        # Build context
        context_parts = []
        total_length = 0
        char_limit = max_tokens * 4  # Rough char estimate

        for parent in parents:
            content = self._chunker.get_with_context(
                chunk=parent,
                all_chunks=all_chunks,
                include_parent=True,
                include_children=False,
            )

            if total_length + len(content) > char_limit:
                break

            context_parts.append(content)
            total_length += len(content)

        return "\n\n---\n\n".join(context_parts)


class MarkdownHierarchicalChunker(BaseChunker):
    """
    Hierarchical chunker for Markdown documents.

    Uses heading structure (h1, h2, h3) to create hierarchy.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        min_chunk_size: int = 50,
    ):
        """
        Initialize markdown hierarchical chunker.

        Args:
            chunk_size: Max chunk size for leaf chunks.
            min_chunk_size: Minimum chunk size.
        """
        super().__init__(chunk_size, 0)
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunk markdown document by headings.

        Args:
            document: Markdown document.

        Returns:
            Hierarchical chunks based on heading structure.
        """
        import re

        lines = document.content.split("\n")
        chunks: list[HierarchicalChunk] = []

        current_headers: dict[int, str] = {}  # level -> header text
        current_content: list[str] = []
        current_level = 0

        def create_chunk_from_content() -> None:
            if not current_content:
                return

            content = "\n".join(current_content).strip()
            if len(content) < self.min_chunk_size:
                return

            # Build header path
            header_path = []
            for level in sorted(current_headers.keys()):
                if level <= current_level:
                    header_path.append(current_headers[level])

            chunk = HierarchicalChunk(
                id=str(uuid.uuid4()),
                content=content,
                metadata={
                    "source": document.source,
                    "level": current_level,
                    "headers": header_path,
                    "header_path": " > ".join(header_path),
                    **(document.metadata or {}),
                },
                embedding=None,
                level=current_level,
                parent_id=None,
                child_ids=[],
            )
            chunks.append(chunk)

        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = heading_pattern.match(line)

            if match:
                # Save current content
                create_chunk_from_content()
                current_content = []

                # Parse heading
                level = len(match.group(1))
                header_text = match.group(2).strip()

                current_level = level
                current_headers[level] = header_text

                # Clear lower-level headers
                for l in list(current_headers.keys()):
                    if l > level:
                        del current_headers[l]

            current_content.append(line)

        # Don't forget last section
        create_chunk_from_content()

        return chunks
