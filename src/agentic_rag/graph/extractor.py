"""
Entity and relationship extraction for GraphRAG.

Extracts knowledge graph elements from text using LLM-based extraction.
This follows Microsoft's GraphRAG approach of building knowledge graphs
from unstructured text.
"""

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic_rag.core.models import Chunk
from agentic_rag.core.protocols import Generator


class Entity(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type (Person, Organization, Concept, etc.)")
    description: str = Field(default="", description="Entity description")
    source_chunk_ids: list[str] = Field(default_factory=list, description="Source chunks")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.name.lower(), self.type.lower()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower() and self.type.lower() == other.type.lower()


class Relationship(BaseModel):
    """An edge in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    relationship_type: str = Field(description="Type of relationship")
    description: str = Field(default="", description="Relationship description")
    weight: float = Field(default=1.0, ge=0.0, description="Relationship strength")
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Result of entity/relationship extraction."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    source_chunk_id: str = ""


class EntityExtractor(ABC):
    """Base class for entity extraction."""

    @abstractmethod
    async def extract(self, chunk: Chunk) -> ExtractionResult:
        """
        Extract entities and relationships from a chunk.

        Args:
            chunk: Text chunk to process.

        Returns:
            ExtractionResult with entities and relationships.
        """
        ...

    async def extract_batch(self, chunks: list[Chunk]) -> list[ExtractionResult]:
        """
        Extract from multiple chunks.

        Args:
            chunks: List of chunks.

        Returns:
            List of extraction results.
        """
        results = []
        for chunk in chunks:
            result = await self.extract(chunk)
            results.append(result)
        return results


# Extraction prompt following GraphRAG methodology
ENTITY_EXTRACTION_PROMPT = """Extract all entities and relationships from the following text.

Text:
{text}

Instructions:
1. Identify all named entities (people, organizations, concepts, locations, events, etc.)
2. For each entity, provide:
   - name: The entity's canonical name
   - type: One of [Person, Organization, Concept, Location, Event, Technology, Document, Other]
   - description: Brief description of the entity based on the text

3. Identify relationships between entities. For each relationship:
   - source: Source entity name
   - target: Target entity name
   - type: Relationship type (e.g., "works_for", "located_in", "related_to", "developed_by")
   - description: Brief description of the relationship

Output format (JSON):
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}},
    ...
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "...", "description": "..."}},
    ...
  ]
}}

Extract entities and relationships:"""


class LLMEntityExtractor(EntityExtractor):
    """
    LLM-based entity and relationship extraction.

    Uses an LLM to extract structured knowledge graph elements
    from unstructured text, following the GraphRAG methodology.
    """

    def __init__(
        self,
        generator: Generator,
        entity_types: list[str] | None = None,
        max_entities_per_chunk: int = 50,
        min_entity_mentions: int = 1,
    ):
        """
        Initialize LLM entity extractor.

        Args:
            generator: LLM for extraction.
            entity_types: Allowed entity types (None = all).
            max_entities_per_chunk: Max entities to extract per chunk.
            min_entity_mentions: Minimum mentions to include entity.
        """
        self._generator = generator
        self._entity_types = entity_types or [
            "Person",
            "Organization",
            "Concept",
            "Location",
            "Event",
            "Technology",
            "Document",
            "Other",
        ]
        self._max_entities = max_entities_per_chunk
        self._min_mentions = min_entity_mentions

    async def extract(self, chunk: Chunk) -> ExtractionResult:
        """
        Extract entities and relationships from a chunk using LLM.

        Args:
            chunk: Text chunk to process.

        Returns:
            ExtractionResult with entities and relationships.
        """
        if not chunk.content.strip():
            return ExtractionResult(source_chunk_id=chunk.id)

        prompt = ENTITY_EXTRACTION_PROMPT.format(text=chunk.content[:4000])

        try:
            response = await self._generator.generate_text(
                prompt=prompt,
                temperature=0.1,  # Low temp for consistent extraction
                max_tokens=2000,
            )

            # Parse JSON response
            entities, relationships = self._parse_response(response, chunk.id)

            return ExtractionResult(
                entities=entities[: self._max_entities],
                relationships=relationships,
                source_chunk_id=chunk.id,
            )

        except Exception as e:
            # Return empty result on error
            return ExtractionResult(
                source_chunk_id=chunk.id,
                metadata={"error": str(e)},
            )

    def _parse_response(
        self,
        response: str,
        chunk_id: str,
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Parse LLM response into entities and relationships.

        Args:
            response: LLM response text.
            chunk_id: Source chunk ID.

        Returns:
            Tuple of (entities, relationships).
        """
        import json
        import re

        entities = []
        relationships = []

        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return entities, relationships

        try:
            data = json.loads(json_match.group())

            # Parse entities
            for e in data.get("entities", []):
                if not e.get("name"):
                    continue

                entity_type = e.get("type", "Other")
                if self._entity_types and entity_type not in self._entity_types:
                    entity_type = "Other"

                entity = Entity(
                    name=e["name"],
                    type=entity_type,
                    description=e.get("description", ""),
                    source_chunk_ids=[chunk_id],
                )
                entities.append(entity)

            # Parse relationships
            entity_names = {e.name.lower() for e in entities}
            for r in data.get("relationships", []):
                source = r.get("source", "")
                target = r.get("target", "")

                # Validate entities exist
                if source.lower() not in entity_names or target.lower() not in entity_names:
                    continue

                relationship = Relationship(
                    source_entity=source,
                    target_entity=target,
                    relationship_type=r.get("type", "related_to"),
                    description=r.get("description", ""),
                    source_chunk_ids=[chunk_id],
                )
                relationships.append(relationship)

        except json.JSONDecodeError:
            pass

        return entities, relationships


class CachingEntityExtractor(LLMEntityExtractor):
    """
    Entity extractor with caching to avoid re-extraction.
    """

    def __init__(self, generator: Generator, cache_size: int = 1000, **kwargs: Any):
        """Initialize with cache."""
        super().__init__(generator, **kwargs)
        self._cache: dict[str, ExtractionResult] = {}
        self._cache_size = cache_size

    def _get_cache_key(self, chunk: Chunk) -> str:
        """Generate cache key for chunk."""
        import hashlib

        return hashlib.md5(chunk.content[:500].encode()).hexdigest()

    async def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract with caching."""
        cache_key = self._get_cache_key(chunk)

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = await super().extract(chunk)

        # Add to cache
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        """Clear extraction cache."""
        self._cache.clear()


def merge_entities(entities: list[Entity]) -> list[Entity]:
    """
    Merge duplicate entities and aggregate their source chunks.

    Args:
        entities: List of entities (may have duplicates).

    Returns:
        Deduplicated list with merged source chunks.
    """
    entity_map: dict[tuple[str, str], Entity] = {}

    for entity in entities:
        key = (entity.name.lower(), entity.type.lower())

        if key in entity_map:
            existing = entity_map[key]
            # Merge source chunks
            existing.source_chunk_ids = list(
                set(existing.source_chunk_ids + entity.source_chunk_ids)
            )
            # Keep longer description
            if len(entity.description) > len(existing.description):
                existing.description = entity.description
        else:
            entity_map[key] = entity.model_copy()

    return list(entity_map.values())


def merge_relationships(relationships: list[Relationship]) -> list[Relationship]:
    """
    Merge duplicate relationships and increase weights.

    Args:
        relationships: List of relationships.

    Returns:
        Deduplicated list with aggregated weights.
    """
    rel_map: dict[tuple[str, str, str], Relationship] = {}

    for rel in relationships:
        key = (
            rel.source_entity.lower(),
            rel.target_entity.lower(),
            rel.relationship_type.lower(),
        )

        if key in rel_map:
            existing = rel_map[key]
            existing.weight += 1.0  # Increase weight for repeated relationships
            existing.source_chunk_ids = list(set(existing.source_chunk_ids + rel.source_chunk_ids))
        else:
            rel_map[key] = rel.model_copy()

    return list(rel_map.values())
