"""
RAPTOR-aware retrieval for hierarchical document search.

Retrieves from RAPTOR trees at multiple abstraction levels,
combining leaf-level detail with summary-level context.
"""

import logging

from agentic_rag.chunking.raptor import RAPTORTree
from agentic_rag.core.models import Chunk, RetrievalResult
from agentic_rag.embeddings import BaseEmbedder
from agentic_rag.vectordb import BaseVectorDB

logger = logging.getLogger(__name__)


class RAPTORRetriever:
    """
    Tree-aware retriever for RAPTOR-indexed documents.

    Retrieval modes:
    - collapsed: Search all levels, deduplicate
    - tree_traversal: Start from summaries, drill down to leaves
    - level_specific: Search specific abstraction level

    Example:
        retriever = RAPTORRetriever(
            embedder=embedder,
            vectordb=vectordb,
        )
        result = await retriever.retrieve(
            query="What are the main themes?",
            collection="my-docs",
            mode="collapsed",
        )
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vectordb: BaseVectorDB,
        tree: RAPTORTree | None = None,
    ):
        """
        Initialize RAPTOR retriever.

        Args:
            embedder: Embedding model.
            vectordb: Vector database.
            tree: Optional in-memory RAPTOR tree.
        """
        self._embedder = embedder
        self._vectordb = vectordb
        self._tree = tree

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        mode: str = "collapsed",
        target_level: int | None = None,
    ) -> RetrievalResult:
        """
        Retrieve from RAPTOR-indexed collection.

        Args:
            query: Search query.
            collection: Collection name.
            top_k: Number of results.
            mode: Retrieval mode (collapsed, tree_traversal, level_specific).
            target_level: For level_specific mode.

        Returns:
            RetrievalResult with chunks.
        """
        if mode == "collapsed":
            return await self._collapsed_retrieval(query, collection, top_k)
        elif mode == "tree_traversal":
            return await self._tree_traversal(query, collection, top_k)
        elif mode == "level_specific" and target_level is not None:
            return await self._level_specific(query, collection, top_k, target_level)
        else:
            return await self._collapsed_retrieval(query, collection, top_k)

    async def _collapsed_retrieval(
        self,
        query: str,
        collection: str,
        top_k: int,
    ) -> RetrievalResult:
        """
        Collapsed retrieval: search all levels, deduplicate by content.

        Best for general queries where abstraction level is unknown.
        """
        # Get query embedding
        query_embedding = await self._embedder.embed(query)

        # Search with extra candidates for deduplication
        search_k = top_k * 2

        results = await self._vectordb.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=search_k,
        )

        # Deduplicate: prefer leaves when content overlaps with summary
        seen_content_hashes: set[str] = set()
        unique_chunks: list[Chunk] = []

        for chunk in results:
            # Simple content hash for deduplication
            content_hash = hash(chunk.content[:100])

            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                unique_chunks.append(chunk)

                if len(unique_chunks) >= top_k:
                    break

        # Sort by level (leaves first) then score
        unique_chunks.sort(
            key=lambda c: (
                c.metadata.get("raptor_level", 0),
                -c.metadata.get("score", 0),
            )
        )

        final_chunks = unique_chunks[:top_k]
        return RetrievalResult(
            chunks=final_chunks,
            scores=[c.metadata.get("score", 0.0) for c in final_chunks],
            retrieval_type="raptor_collapsed",
            metadata={
                "mode": "collapsed",
                "levels_searched": list(
                    {c.metadata.get("raptor_level", 0) for c in unique_chunks}
                ),
            },
        )

    async def _tree_traversal(
        self,
        query: str,
        collection: str,
        top_k: int,
    ) -> RetrievalResult:
        """
        Tree traversal: start from summaries, drill down to relevant leaves.

        Best for queries that need both context and detail.
        """
        query_embedding = await self._embedder.embed(query)

        # First: find relevant summaries
        all_results = await self._vectordb.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k * 3,
        )

        # Separate by level
        summaries = [c for c in all_results if c.metadata.get("is_summary", False)]
        leaves = [c for c in all_results if not c.metadata.get("is_summary", False)]

        # Combine: summaries for context, leaves for detail
        # Ratio: ~30% summaries, ~70% leaves
        n_summaries = max(1, top_k // 3)
        n_leaves = top_k - n_summaries

        selected = summaries[:n_summaries] + leaves[:n_leaves]

        return RetrievalResult(
            chunks=selected,
            scores=[c.metadata.get("score", 0.0) for c in selected],
            retrieval_type="raptor_tree_traversal",
            metadata={
                "mode": "tree_traversal",
                "summaries_selected": len(summaries[:n_summaries]),
                "leaves_selected": len(leaves[:n_leaves]),
            },
        )

    async def _level_specific(
        self,
        query: str,
        collection: str,
        top_k: int,
        target_level: int,
    ) -> RetrievalResult:
        """
        Level-specific: search only at a specific abstraction level.

        Level 0 = most detail (leaves)
        Higher levels = more abstract (summaries)
        """
        query_embedding = await self._embedder.embed(query)

        # Search with filter
        # Note: This requires the vectordb to support metadata filtering
        all_results = await self._vectordb.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k * 3,  # Extra for filtering
        )

        # Filter to target level
        level_results = [
            c for c in all_results if c.metadata.get("raptor_level", 0) == target_level
        ]

        final_chunks = level_results[:top_k]
        return RetrievalResult(
            chunks=final_chunks,
            scores=[c.metadata.get("score", 0.0) for c in final_chunks],
            retrieval_type="raptor_level_specific",
            metadata={
                "mode": "level_specific",
                "target_level": target_level,
                "results_at_level": len(level_results),
            },
        )

    async def retrieve_with_context(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve leaves with their parent summaries for context.

        Returns leaves enriched with parent summary context.
        """
        query_embedding = await self._embedder.embed(query)

        # Get relevant leaves
        results = await self._vectordb.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k * 2,
        )

        leaves = [c for c in results if not c.metadata.get("is_summary", False)][:top_k]

        # For each leaf, try to find parent summary
        enriched_chunks = []
        for leaf in leaves:
            # Add parent context if available
            parent_id = leaf.metadata.get("parent_id")
            if parent_id and self._tree:
                parent = self._tree.nodes.get(parent_id)
                if parent:
                    # Prepend summary context
                    enriched_content = f"[Context: {parent.content[:200]}...]\n\n{leaf.content}"
                    enriched_chunk = Chunk(
                        id=leaf.id,
                        content=enriched_content,
                        document_id=leaf.document_id,
                        embedding=leaf.embedding,
                        metadata={
                            **leaf.metadata,
                            "has_parent_context": True,
                        },
                    )
                    enriched_chunks.append(enriched_chunk)
                    continue

            enriched_chunks.append(leaf)

        return RetrievalResult(
            chunks=enriched_chunks,
            scores=[c.metadata.get("score", 0.0) for c in enriched_chunks],
            retrieval_type="raptor_context_enriched",
            metadata={
                "mode": "context_enriched",
                "enriched_count": sum(
                    1 for c in enriched_chunks if c.metadata.get("has_parent_context")
                ),
            },
        )
