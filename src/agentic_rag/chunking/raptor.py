"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

RAPTOR builds a hierarchical tree of chunks where:
- Level 0: Original document chunks (leaves)
- Level 1+: Summaries of clustered chunks from previous level

This enables retrieval at different abstraction levels:
- Leaf nodes for specific details
- Higher nodes for broader concepts
"""

import logging
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from agentic_rag.chunking.clustering import BaseClusterer, create_clusterer
from agentic_rag.core.models import Chunk, Document
from agentic_rag.embeddings import BaseEmbedder

logger = logging.getLogger(__name__)


class RAPTORNode(BaseModel):
    """A node in the RAPTOR tree."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="Node content (original or summary)")
    embedding: list[float] | None = Field(default=None, description="Node embedding")
    level: int = Field(default=0, description="Tree level (0 = leaf)")
    is_summary: bool = Field(default=False, description="Whether this is a summary node")
    parent_id: str | None = Field(default=None, description="Parent node ID")
    child_ids: list[str] = Field(default_factory=list, description="Child node IDs")
    cluster_id: int | None = Field(default=None, description="Cluster assignment")
    document_id: str | None = Field(default=None, description="Source document ID")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_chunk(self) -> Chunk:
        """Convert node to Chunk for retrieval."""
        return Chunk(
            id=self.id,
            content=self.content,
            embedding=self.embedding,
            document_id=self.document_id or "",
            metadata={
                **self.metadata,
                "raptor_level": self.level,
                "is_summary": self.is_summary,
                "cluster_id": self.cluster_id,
            },
        )


class RAPTORTree(BaseModel):
    """RAPTOR tree structure containing all nodes."""

    nodes: dict[str, RAPTORNode] = Field(default_factory=dict)
    root_ids: list[str] = Field(default_factory=list, description="Top-level node IDs")
    max_level: int = Field(default=0, description="Maximum tree depth")
    document_id: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_level(self, level: int) -> list[RAPTORNode]:
        """Get all nodes at a specific level."""
        return [n for n in self.nodes.values() if n.level == level]

    def get_leaves(self) -> list[RAPTORNode]:
        """Get leaf nodes (level 0)."""
        return self.get_level(0)

    def get_summaries(self) -> list[RAPTORNode]:
        """Get all summary nodes."""
        return [n for n in self.nodes.values() if n.is_summary]

    def all_chunks(self) -> list[Chunk]:
        """Convert all nodes to chunks for indexing."""
        return [node.to_chunk() for node in self.nodes.values()]

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def leaf_count(self) -> int:
        return len(self.get_leaves())

    @property
    def summary_count(self) -> int:
        return len(self.get_summaries())


class RAPTORChunker:
    """
    RAPTOR chunker that builds hierarchical document trees.

    Process:
    1. Create leaf chunks from document
    2. Embed chunks
    3. Cluster similar chunks
    4. Generate summary for each cluster
    5. Repeat clustering on summaries until threshold
    6. Build tree structure

    Example:
        chunker = RAPTORChunker(
            embedder=embedder,
            generator=generator,
            max_levels=3,
        )
        tree = await chunker.chunk_with_tree(document)

        # Get all nodes for indexing
        chunks = tree.all_chunks()
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        generator: Any,
        base_chunker: Any = None,
        clusterer: BaseClusterer | None = None,
        max_levels: int = 3,
        min_cluster_size: int = 2,
        summary_max_tokens: int = 200,
        clustering_algorithm: str = "gmm",
    ):
        """
        Initialize RAPTOR chunker.

        Args:
            embedder: Embedding model for similarity.
            generator: LLM for summarization.
            base_chunker: Chunker for initial document splitting.
            clusterer: Clustering algorithm.
            max_levels: Maximum tree depth.
            min_cluster_size: Minimum nodes to cluster.
            summary_max_tokens: Max tokens per summary.
            clustering_algorithm: Algorithm for clustering (gmm, kmeans).
        """
        self._embedder = embedder
        self._generator = generator
        self._base_chunker = base_chunker
        self._clusterer = clusterer or create_clusterer(clustering_algorithm)
        self._max_levels = max_levels
        self._min_cluster_size = min_cluster_size
        self._summary_max_tokens = summary_max_tokens

    async def _create_leaf_chunks(self, document: Document) -> list[RAPTORNode]:
        """Create leaf nodes from document."""
        if self._base_chunker is not None:
            chunks = await self._base_chunker.chunk_async(document)
        else:
            # Simple chunking fallback
            content = document.content
            chunk_size = 1000
            overlap = 100
            chunks = []

            start = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk_content = content[start:end].strip()
                if chunk_content:
                    chunks.append(
                        Chunk(
                            content=chunk_content,
                            document_id=document.id,
                        )
                    )
                start = end - overlap

        # Convert to RAPTOR nodes
        nodes = []
        for chunk in chunks:
            node = RAPTORNode(
                id=chunk.id,
                content=chunk.content,
                level=0,
                is_summary=False,
                document_id=document.id,
                metadata=chunk.metadata,
            )
            nodes.append(node)

        return nodes

    async def _embed_nodes(self, nodes: list[RAPTORNode]) -> None:
        """Add embeddings to nodes."""
        texts = [n.content for n in nodes]
        embeddings = await self._embedder.embed_batch(texts)

        for node, embedding in zip(nodes, embeddings, strict=False):
            node.embedding = embedding

    async def _summarize_cluster(
        self,
        nodes: list[RAPTORNode],
        cluster_id: int,
    ) -> str:
        """Generate summary for a cluster of nodes."""
        # Combine node contents
        combined = "\n\n---\n\n".join(n.content for n in nodes)

        prompt = f"""Summarize the following related text passages into a coherent summary.
Focus on the key concepts and main points.
Keep the summary concise ({self._summary_max_tokens} tokens max).

Text passages:
{combined[:8000]}

Summary:"""

        try:
            result = await self._generator.generate(
                query=prompt,
                context=[],
                max_tokens=self._summary_max_tokens,
                temperature=0.3,
            )
            return result.response.strip()
        except Exception as e:
            logger.warning(f"Summarization failed for cluster {cluster_id}: {e}")
            # Fallback: concatenate first sentence of each node
            summaries = []
            for n in nodes[:3]:
                first_sent = n.content.split(".")[0] + "."
                summaries.append(first_sent[:200])
            return " ".join(summaries)

    async def _build_level(
        self,
        nodes: list[RAPTORNode],
        level: int,
        tree: RAPTORTree,
    ) -> list[RAPTORNode]:
        """Build one level of the RAPTOR tree."""
        if len(nodes) < self._min_cluster_size:
            # Not enough nodes to cluster
            return []

        # Get embeddings as numpy array
        embeddings = np.array([n.embedding for n in nodes])

        # Cluster nodes
        cluster_result = self._clusterer.cluster(embeddings)

        logger.info(
            f"RAPTOR level {level}: Clustered {len(nodes)} nodes into "
            f"{cluster_result.n_clusters} clusters"
        )

        # Group nodes by cluster
        clusters: dict[int, list[RAPTORNode]] = {}
        for node, label in zip(nodes, cluster_result.labels, strict=False):
            node.cluster_id = label
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)

        # Create summary nodes for each cluster
        summary_nodes = []
        for cluster_id, cluster_nodes in clusters.items():
            if len(cluster_nodes) < 2:
                # Single node cluster - promote directly
                for node in cluster_nodes:
                    tree.nodes[node.id] = node
                continue

            # Generate summary
            summary_content = await self._summarize_cluster(cluster_nodes, cluster_id)

            # Create summary node
            summary_node = RAPTORNode(
                content=summary_content,
                level=level,
                is_summary=True,
                cluster_id=cluster_id,
                child_ids=[n.id for n in cluster_nodes],
                document_id=cluster_nodes[0].document_id,
                metadata={"cluster_size": len(cluster_nodes)},
            )

            # Update children with parent reference
            for child in cluster_nodes:
                child.parent_id = summary_node.id
                tree.nodes[child.id] = child

            summary_nodes.append(summary_node)

        # Embed summary nodes
        if summary_nodes:
            await self._embed_nodes(summary_nodes)

        return summary_nodes

    async def chunk_with_tree(self, document: Document) -> RAPTORTree:
        """
        Build a RAPTOR hierarchical tree from a document.

        The process follows these steps:
        1. Leaf Creation: Splits the document into initial small chunks (Level 0).
        2. Embedding: Generates vectors for all current level nodes.
        3. Clustering: Groups semantically similar nodes using GMM or KMeans.
        4. Summarization: Uses an LLM to generate a concise summary for each cluster.
        5. Recursion: The summaries become nodes for the next level (Level 1, 2, ...).
        6. Tree Assembly: Continues until the maximum level is reached or
           clusters can no longer be formed.

        Args:
            document: The source Document to process.

        Returns:
            A RAPTORTree containing all original chunks and hierarchical summaries.
        """
        logger.info(f"Building RAPTOR tree for document: {document.id}")

        tree = RAPTORTree(document_id=document.id)

        # Create leaf chunks
        leaf_nodes = await self._create_leaf_chunks(document)
        logger.info(f"RAPTOR: Created {len(leaf_nodes)} leaf nodes")

        if not leaf_nodes:
            return tree

        # Embed leaf nodes
        await self._embed_nodes(leaf_nodes)

        # Build tree levels
        current_level_nodes = leaf_nodes
        level = 1

        while level <= self._max_levels and len(current_level_nodes) >= self._min_cluster_size:
            summary_nodes = await self._build_level(current_level_nodes, level, tree)

            if not summary_nodes:
                break

            current_level_nodes = summary_nodes
            level += 1

        # Add final level nodes
        for node in current_level_nodes:
            tree.nodes[node.id] = node
            tree.root_ids.append(node.id)

        tree.max_level = level - 1

        logger.info(
            f"RAPTOR tree complete: {tree.total_nodes} nodes, "
            f"{tree.leaf_count} leaves, {tree.summary_count} summaries, "
            f"max level {tree.max_level}"
        )

        return tree

    async def chunk_async(self, document: Document) -> list[Chunk]:
        """
        Chunk a document with Anthropic-style context headers.

        This method:
        1. Segments the document into base chunks.
        2. Generates a brief context header for each chunk using an LLM to
           explain its location and context within the full document.
        3. Prepends these headers to the chunks to improve retrieval accuracy.

        Args:
            document: The Document object to be chunked.

        Returns:
            A list of Chunk objects, each containing its original content
            augmented with a context header.
        """
        tree = await self.chunk_with_tree(document)
        return tree.all_chunks()
