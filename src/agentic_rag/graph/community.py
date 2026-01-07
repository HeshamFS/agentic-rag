"""
Community detection for GraphRAG.

Uses the Leiden algorithm to detect communities in knowledge graphs,
enabling hierarchical summarization for global query answering.

Reference: "GraphRAG: Indexing and Retrieval Using Knowledge Graphs" (Microsoft)
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.core.protocols import Generator
from agentic_rag.graph.extractor import Entity, Relationship


class Community(BaseModel):
    """A community of related entities in the knowledge graph."""

    id: str = Field(description="Community ID")
    level: int = Field(default=0, description="Hierarchy level (0 = leaf)")
    title: str = Field(default="", description="Community title/name")
    summary: str = Field(default="", description="LLM-generated summary")
    entity_ids: list[str] = Field(default_factory=list, description="Member entity IDs")
    parent_id: str | None = Field(default=None, description="Parent community ID")
    child_ids: list[str] = Field(default_factory=list, description="Child community IDs")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of entities in this community."""
        return len(self.entity_ids)


class CommunityHierarchy(BaseModel):
    """Hierarchical structure of communities."""

    communities: dict[str, Community] = Field(default_factory=dict)
    levels: int = Field(default=0, description="Number of hierarchy levels")
    root_ids: list[str] = Field(default_factory=list, description="Top-level community IDs")

    def get_level(self, level: int) -> list[Community]:
        """Get all communities at a specific level."""
        return [c for c in self.communities.values() if c.level == level]

    def get_ancestors(self, community_id: str) -> list[Community]:
        """Get all ancestor communities."""
        ancestors = []
        current = self.communities.get(community_id)

        while current and current.parent_id:
            parent = self.communities.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break

        return ancestors


class CommunityDetector(ABC):
    """Base class for community detection algorithms."""

    @abstractmethod
    def detect(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> CommunityHierarchy:
        """
        Detect communities in the knowledge graph.

        Args:
            entities: List of entities (nodes).
            relationships: List of relationships (edges).

        Returns:
            CommunityHierarchy with detected communities.
        """
        ...


class LeidenCommunityDetector(CommunityDetector):
    """
    Leiden algorithm for community detection.

    The Leiden algorithm improves on Louvain by guaranteeing
    well-connected communities and providing faster convergence.

    Uses NetworkX with community detection (cdlib or leidenalg if available).
    """

    def __init__(
        self,
        resolution: float = 1.0,
        max_levels: int = 3,
        min_community_size: int = 2,
    ):
        """
        Initialize Leiden detector.

        Args:
            resolution: Resolution parameter (higher = smaller communities).
            max_levels: Maximum hierarchy levels to build.
            min_community_size: Minimum entities per community.
        """
        self._resolution = resolution
        self._max_levels = max_levels
        self._min_community_size = min_community_size

    def detect(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> CommunityHierarchy:
        """
        Detect communities using Leiden algorithm.

        Args:
            entities: List of entities.
            relationships: List of relationships.

        Returns:
            CommunityHierarchy with multi-level communities.
        """
        import networkx as nx

        if not entities:
            return CommunityHierarchy()

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes
        entity_map = {e.name.lower(): e for e in entities}
        for entity in entities:
            G.add_node(entity.id, name=entity.name, type=entity.type)

        # Add edges
        for rel in relationships:
            source_entity = entity_map.get(rel.source_entity.lower())
            target_entity = entity_map.get(rel.target_entity.lower())

            if source_entity and target_entity:
                G.add_edge(
                    source_entity.id,
                    target_entity.id,
                    weight=rel.weight,
                    type=rel.relationship_type,
                )

        # Detect communities
        communities = self._run_community_detection(G)

        # Build hierarchy
        hierarchy = self._build_hierarchy(entities, communities)

        return hierarchy

    def _run_community_detection(self, G) -> list[set[str]]:
        """
        Run community detection algorithm.

        Falls back to simpler methods if Leiden not available.
        """
        import networkx as nx

        if len(G.nodes()) == 0:
            return []

        # Try different community detection methods
        try:
            # Try leidenalg (best quality)
            import igraph as ig
            import leidenalg

            # Convert NetworkX to igraph
            ig_graph = ig.Graph.from_networkx(G)

            # Run Leiden
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self._resolution,
            )

            # Convert back to node IDs
            communities = []
            node_ids = list(G.nodes())
            for community in partition:
                comm_nodes = {node_ids[i] for i in community}
                if len(comm_nodes) >= self._min_community_size:
                    communities.append(comm_nodes)

            return communities

        except ImportError:
            pass

        try:
            # Fall back to NetworkX Louvain
            from networkx.algorithms.community import louvain_communities

            communities = louvain_communities(
                G,
                resolution=self._resolution,
                seed=42,
            )

            return [c for c in communities if len(c) >= self._min_community_size]

        except Exception:
            pass

        # Final fallback: connected components
        components = list(nx.connected_components(G))
        return [c for c in components if len(c) >= self._min_community_size]

    def _build_hierarchy(
        self,
        entities: list[Entity],
        base_communities: list[set[str]],
    ) -> CommunityHierarchy:
        """
        Build hierarchical community structure.

        Args:
            entities: All entities.
            base_communities: Base-level communities (sets of entity IDs).

        Returns:
            CommunityHierarchy with multi-level structure.
        """

        hierarchy = CommunityHierarchy()
        entity_map = {e.id: e for e in entities}

        # Create level 0 communities
        level_0_ids = []
        for i, comm_entities in enumerate(base_communities):
            comm_id = f"comm_0_{i}"

            # Generate title from top entities
            member_entities = [entity_map[eid] for eid in comm_entities if eid in entity_map]
            title = self._generate_title(member_entities)

            community = Community(
                id=comm_id,
                level=0,
                title=title,
                entity_ids=list(comm_entities),
            )
            hierarchy.communities[comm_id] = community
            level_0_ids.append(comm_id)

        # Build higher levels by merging communities
        current_level_ids = level_0_ids
        for level in range(1, self._max_levels):
            if len(current_level_ids) <= 1:
                break

            # Simple merging: pair communities
            next_level_ids = []
            for i in range(0, len(current_level_ids), 2):
                child_ids = current_level_ids[i : i + 2]

                # Merge entities from children
                merged_entities = []
                for child_id in child_ids:
                    child = hierarchy.communities[child_id]
                    merged_entities.extend(child.entity_ids)
                    child.parent_id = f"comm_{level}_{len(next_level_ids)}"

                comm_id = f"comm_{level}_{len(next_level_ids)}"
                member_entities = [entity_map[eid] for eid in merged_entities if eid in entity_map]

                community = Community(
                    id=comm_id,
                    level=level,
                    title=self._generate_title(member_entities),
                    entity_ids=merged_entities,
                    child_ids=child_ids,
                )
                hierarchy.communities[comm_id] = community
                next_level_ids.append(comm_id)

            current_level_ids = next_level_ids

        hierarchy.levels = (
            max(c.level for c in hierarchy.communities.values()) + 1 if hierarchy.communities else 0
        )
        hierarchy.root_ids = [c.id for c in hierarchy.communities.values() if c.parent_id is None]

        return hierarchy

    def _generate_title(self, entities: list[Entity]) -> str:
        """Generate a title from the top entities in a community."""
        if not entities:
            return "Unknown Community"

        # Sort by type importance and take top 3
        type_priority = {"Concept": 0, "Organization": 1, "Person": 2, "Technology": 3}
        sorted_entities = sorted(
            entities,
            key=lambda e: (type_priority.get(e.type, 5), -len(e.description)),
        )

        names = [e.name for e in sorted_entities[:3]]
        return ", ".join(names)


class CommunitySummarizer:
    """
    Generate summaries for communities using LLM.

    Creates hierarchical summaries that enable global query answering.
    """

    def __init__(self, generator: Generator, max_entities_in_prompt: int = 20):
        """
        Initialize summarizer.

        Args:
            generator: LLM for summary generation.
            max_entities_in_prompt: Max entities to include in prompt.
        """
        self._generator = generator
        self._max_entities = max_entities_in_prompt

    async def summarize_community(
        self,
        community: Community,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> str:
        """
        Generate a summary for a community.

        Args:
            community: The community to summarize.
            entities: Entities in the community.
            relationships: Relationships between community entities.

        Returns:
            Summary text.
        """
        # Build entity context
        entity_texts = []
        for entity in entities[: self._max_entities]:
            entity_texts.append(f"- {entity.name} ({entity.type}): {entity.description}")

        # Build relationship context
        rel_texts = []
        {e.id for e in entities}
        relevant_rels = [
            r
            for r in relationships
            if any(e.name.lower() == r.source_entity.lower() for e in entities)
            and any(e.name.lower() == r.target_entity.lower() for e in entities)
        ]

        for rel in relevant_rels[:20]:
            rel_texts.append(f"- {rel.source_entity} {rel.relationship_type} {rel.target_entity}")

        prompt = f"""Summarize this group of related concepts:

Entities:
{chr(10).join(entity_texts)}

Relationships:
{chr(10).join(rel_texts) if rel_texts else "No explicit relationships"}

Write a 2-3 sentence summary describing what this group represents,
their main themes, and key relationships. Be concise and informative."""

        summary = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=200,
        )

        return summary.strip()

    async def summarize_hierarchy(
        self,
        hierarchy: CommunityHierarchy,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> CommunityHierarchy:
        """
        Generate summaries for all communities in the hierarchy.

        Args:
            hierarchy: Community hierarchy.
            entities: All entities.
            relationships: All relationships.

        Returns:
            Updated hierarchy with summaries.
        """
        entity_map = {e.id: e for e in entities}

        # Process from bottom to top (leaf summaries first)
        for level in range(hierarchy.levels):
            level_communities = hierarchy.get_level(level)

            for community in level_communities:
                # Get entities in this community
                comm_entities = [
                    entity_map[eid] for eid in community.entity_ids if eid in entity_map
                ]

                # Generate summary
                summary = await self.summarize_community(
                    community,
                    comm_entities,
                    relationships,
                )
                community.summary = summary

        return hierarchy
