"""
Knowledge graph storage for GraphRAG.

Provides storage backends for knowledge graphs with support for:
- NetworkX (in-memory, good for small-medium graphs)
- Optional Neo4j integration (for production scale)
"""

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from agentic_rag.graph.community import Community, CommunityHierarchy
from agentic_rag.graph.extractor import Entity, Relationship


class GraphStats(BaseModel):
    """Statistics about the knowledge graph."""

    num_entities: int = 0
    num_relationships: int = 0
    num_communities: int = 0
    entity_types: dict[str, int] = Field(default_factory=dict)
    relationship_types: dict[str, int] = Field(default_factory=dict)
    avg_degree: float = 0.0


class GraphStorage(ABC):
    """Base class for knowledge graph storage."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        ...

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph."""
        ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        ...

    @abstractmethod
    def get_entity_by_name(self, name: str) -> Entity | None:
        """Get an entity by name."""
        ...

    @abstractmethod
    def get_relationships(self, entity_id: str) -> list[Relationship]:
        """Get all relationships for an entity."""
        ...

    @abstractmethod
    def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by name or description."""
        ...

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> list[Entity]:
        """Get neighboring entities up to specified depth."""
        ...

    @abstractmethod
    def get_stats(self) -> GraphStats:
        """Get graph statistics."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save graph to file."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load graph from file."""
        ...


class NetworkXStorage(GraphStorage):
    """
    NetworkX-based knowledge graph storage.

    Good for small-medium graphs (up to ~100K entities).
    Provides fast in-memory operations.
    """

    def __init__(self):
        """Initialize NetworkX storage."""
        import networkx as nx

        self._graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}
        self._name_to_id: dict[str, str] = {}  # name.lower() -> entity_id
        self._communities: dict[str, Community] = {}

    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity node to the knowledge graph.

        If an entity with the same name (case-insensitive) already exists,
        it merges the source chunk IDs and keeps the longest description.

        Args:
            entity: The Entity object to add.
        """
        # Check for existing entity with same name
        existing_id = self._name_to_id.get(entity.name.lower())
        if existing_id:
            # Merge with existing
            existing = self._entities[existing_id]
            existing.source_chunk_ids = list(
                set(existing.source_chunk_ids + entity.source_chunk_ids)
            )
            if len(entity.description) > len(existing.description):
                existing.description = entity.description
            return

        # Add new entity
        self._entities[entity.id] = entity
        self._name_to_id[entity.name.lower()] = entity.id
        self._graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
        )

    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship edge between two entities in the graph.

        The entities must already exist in the graph (added via add_entity).
        If an identical relationship already exists, its weight is incremented.

        Args:
            relationship: The Relationship object to add.
        """
        # Get entity IDs
        source_id = self._name_to_id.get(relationship.source_entity.lower())
        target_id = self._name_to_id.get(relationship.target_entity.lower())

        if not source_id or not target_id:
            return  # Skip if entities don't exist

        # Check for existing relationship
        existing_key = f"{source_id}:{target_id}:{relationship.relationship_type.lower()}"
        if existing_key in self._relationships:
            existing = self._relationships[existing_key]
            existing.weight += 1.0
            existing.source_chunk_ids = list(
                set(existing.source_chunk_ids + relationship.source_chunk_ids)
            )
            return

        # Add new relationship
        self._relationships[existing_key] = relationship
        self._graph.add_edge(
            source_id,
            target_id,
            type=relationship.relationship_type,
            weight=relationship.weight,
            description=relationship.description,
        )

    def add_entities_batch(self, entities: list[Entity]) -> None:
        """Add multiple entities efficiently."""
        for entity in entities:
            self.add_entity(entity)

    def add_relationships_batch(self, relationships: list[Relationship]) -> None:
        """Add multiple relationships efficiently."""
        for relationship in relationships:
            self.add_relationship(relationship)

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Entity | None:
        """Get an entity by name."""
        entity_id = self._name_to_id.get(name.lower())
        if entity_id:
            return self._entities.get(entity_id)
        return None

    def get_relationships(self, entity_id: str) -> list[Relationship]:
        """Get all relationships for an entity."""
        relationships = []

        # Outgoing relationships
        for _, target_id, data in self._graph.out_edges(entity_id, data=True):
            source = self._entities.get(entity_id)
            target = self._entities.get(target_id)
            if source and target:
                relationships.append(
                    Relationship(
                        source_entity=source.name,
                        target_entity=target.name,
                        relationship_type=data.get("type", "related_to"),
                        weight=data.get("weight", 1.0),
                        description=data.get("description", ""),
                    )
                )

        # Incoming relationships
        for source_id, _, data in self._graph.in_edges(entity_id, data=True):
            source = self._entities.get(source_id)
            target = self._entities.get(entity_id)
            if source and target:
                relationships.append(
                    Relationship(
                        source_entity=source.name,
                        target_entity=target.name,
                        relationship_type=data.get("type", "related_to"),
                        weight=data.get("weight", 1.0),
                        description=data.get("description", ""),
                    )
                )

        return relationships

    def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """
        Search for entities by name or description using keyword matching.

        Args:
            query: The search string.
            entity_type: Optional filter by entity type (e.g., "Person", "Concept").
            limit: Maximum number of results to return.

        Returns:
            List of matching Entity objects, ranked by relevance.
        """
        query_lower = query.lower()
        matches = []

        for entity in self._entities.values():
            # Filter by type if specified
            if entity_type and entity.type.lower() != entity_type.lower():
                continue

            # Check name and description
            score = 0
            if query_lower in entity.name.lower():
                score += 2  # Name match is more important
            if query_lower in entity.description.lower():
                score += 1

            if score > 0:
                matches.append((entity, score))

        # Sort by score and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:limit]]

    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> list[Entity]:
        """
        Get neighboring entities up to specified depth.

        Uses BFS to traverse the graph.
        """

        if entity_id not in self._graph:
            return []

        # Convert to undirected for neighbor search
        undirected = self._graph.to_undirected()

        # BFS to find neighbors
        neighbors = set()
        current_level = {entity_id}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                for neighbor in undirected.neighbors(node):
                    if neighbor != entity_id and neighbor not in neighbors:
                        next_level.add(neighbor)
                        neighbors.add(neighbor)
            current_level = next_level

        return [self._entities[n] for n in neighbors if n in self._entities]

    def get_subgraph(
        self,
        entity_ids: list[str],
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Get a subgraph containing specified entities.

        Args:
            entity_ids: List of entity IDs.

        Returns:
            Tuple of (entities, relationships) in the subgraph.
        """
        entities = [self._entities[eid] for eid in entity_ids if eid in self._entities]
        entity_id_set = set(entity_ids)

        relationships = []
        for source_id, target_id, data in self._graph.edges(data=True):
            if source_id in entity_id_set and target_id in entity_id_set:
                source = self._entities.get(source_id)
                target = self._entities.get(target_id)
                if source and target:
                    relationships.append(
                        Relationship(
                            source_entity=source.name,
                            target_entity=target.name,
                            relationship_type=data.get("type", "related_to"),
                            weight=data.get("weight", 1.0),
                            description=data.get("description", ""),
                        )
                    )

        return entities, relationships

    def set_communities(self, hierarchy: CommunityHierarchy) -> None:
        """Store community hierarchy."""
        self._communities = hierarchy.communities.copy()

    def get_community(self, community_id: str) -> Community | None:
        """Get a community by ID."""
        return self._communities.get(community_id)

    def get_communities_for_entity(self, entity_id: str) -> list[Community]:
        """Get all communities containing an entity."""
        return [c for c in self._communities.values() if entity_id in c.entity_ids]

    def get_stats(self) -> GraphStats:
        """Get graph statistics."""

        entity_types: dict[str, int] = {}
        for entity in self._entities.values():
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1

        rel_types: dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            rel_type = data.get("type", "unknown")
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

        avg_degree = 0.0
        if len(self._graph.nodes()) > 0:
            avg_degree = sum(dict(self._graph.degree()).values()) / len(self._graph.nodes())

        return GraphStats(
            num_entities=len(self._entities),
            num_relationships=len(self._relationships),
            num_communities=len(self._communities),
            entity_types=entity_types,
            relationship_types=rel_types,
            avg_degree=avg_degree,
        )

    def save(self, path: str | Path) -> None:
        """
        Save graph to file.

        Saves as JSON for portability.
        """
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entities": [e.model_dump() for e in self._entities.values()],
            "relationships": [r.model_dump() for r in self._relationships.values()],
            "communities": [c.model_dump() for c in self._communities.values()],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load graph from file."""
        import json
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Clear existing data
        self._graph.clear()
        self._entities.clear()
        self._relationships.clear()
        self._name_to_id.clear()
        self._communities.clear()

        # Load entities
        for e_data in data.get("entities", []):
            entity = Entity(**e_data)
            self.add_entity(entity)

        # Load relationships
        for r_data in data.get("relationships", []):
            relationship = Relationship(**r_data)
            self.add_relationship(relationship)

        # Load communities
        for c_data in data.get("communities", []):
            community = Community(**c_data)
            self._communities[community.id] = community

    def clear(self) -> None:
        """Clear all data from the graph."""
        self._graph.clear()
        self._entities.clear()
        self._relationships.clear()
        self._name_to_id.clear()
        self._communities.clear()

    @property
    def networkx_graph(self):
        """Get the underlying NetworkX graph."""
        return self._graph
