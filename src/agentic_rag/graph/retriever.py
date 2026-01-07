"""
Graph-based retrieval for GraphRAG.

Enables answering "global queries" that require understanding
the overall structure and themes of a document collection.

Reference: "GraphRAG: Indexing and Retrieval Using Knowledge Graphs" (Microsoft)
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.core.protocols import Embedder, Generator
from agentic_rag.graph.community import Community
from agentic_rag.graph.extractor import Entity, Relationship
from agentic_rag.graph.storage import GraphStorage


class GraphRetrievalResult(BaseModel):
    """Result from graph-based retrieval."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    communities: list[Community] = Field(default_factory=list)
    context: str = Field(default="", description="Generated context from graph")
    query_type: str = Field(default="local", description="local or global")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GraphRetriever(ABC):
    """Base class for graph-based retrieval."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> GraphRetrievalResult:
        """
        Retrieve relevant graph elements for a query.

        Args:
            query: User query.
            top_k: Number of results.

        Returns:
            GraphRetrievalResult with relevant entities, relationships, communities.
        """
        ...


class GraphRAGRetriever(GraphRetriever):
    """
    GraphRAG retriever implementing Microsoft's approach.

    Supports two query modes:
    - Local: Find specific entities and their neighborhoods
    - Global: Answer questions about overall themes using community summaries

    The retriever automatically routes queries to the appropriate mode.
    """

    def __init__(
        self,
        storage: GraphStorage,
        generator: Generator,
        embedder: Embedder | None = None,
        community_level: int = 0,
    ):
        """
        Initialize GraphRAG retriever.

        Args:
            storage: Graph storage backend.
            generator: LLM for context generation.
            embedder: Embedder for entity search (optional).
            community_level: Default community level for global queries.
        """
        self._storage = storage
        self._generator = generator
        self._embedder = embedder
        self._community_level = community_level

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> GraphRetrievalResult:
        """
        Execute graph-based retrieval.

        Automatically determines the query type (local vs. global) and
        routes to the appropriate search mode:
        - Local Search: Focuses on specific entities and their relationships.
        - Global Search: Uses hierarchical community summaries for holistic answers.

        Args:
            query: The user search query.
            top_k: Maximum number of graph elements to return.

        Returns:
            GraphRetrievalResult with entities, relationships, and generated context.
        """
        # Classify query type
        query_type = await self._classify_query(query)

        if query_type == "global":
            return await self._global_search(query, top_k)
        else:
            return await self._local_search(query, top_k)

    async def local_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> GraphRetrievalResult:
        """
        Local search: Find specific entities and their context.

        Best for queries like "What is X?" or "How does Y work?"
        """
        return await self._local_search(query, top_k)

    async def global_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> GraphRetrievalResult:
        """
        Global search: Answer using community summaries.

        Best for queries like "What are the main themes?" or "Summarize the key concepts"
        """
        return await self._global_search(query, top_k)

    async def _classify_query(self, query: str) -> str:
        """
        Classify query as local or global.

        Global queries ask about overall patterns, themes, summaries.
        Local queries ask about specific entities or facts.
        """
        prompt = f"""Classify this query as either "local" or "global":

- LOCAL: Asks about specific entities, facts, or details
  Examples: "What is the Transformer architecture?", "Who developed BERT?"

- GLOBAL: Asks about overall themes, patterns, summaries, or comparisons
  Examples: "What are the main themes?", "Compare the approaches", "What trends emerge?"

Query: "{query}"

Answer with just "local" or "global":"""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=10,
        )

        return "global" if "global" in response.lower() else "local"

    async def _local_search(
        self,
        query: str,
        top_k: int,
    ) -> GraphRetrievalResult:
        """
        Execute local search.

        1. Extract entities from query
        2. Find matching entities in graph
        3. Get neighborhoods
        4. Generate context
        """
        # Extract key terms from query
        entities = self._storage.search_entities(query, limit=top_k)

        if not entities:
            return GraphRetrievalResult(
                query_type="local",
                context="No relevant entities found in the knowledge graph.",
            )

        # Get neighborhoods
        all_entities = set()
        all_relationships = []

        for entity in entities[:5]:  # Top 5 seed entities
            all_entities.add(entity.id)

            # Get neighbors
            neighbors = self._storage.get_neighbors(entity.id, depth=1)
            for neighbor in neighbors[:10]:
                all_entities.add(neighbor.id)

            # Get relationships
            rels = self._storage.get_relationships(entity.id)
            all_relationships.extend(rels)

        # Get full entity objects
        result_entities = [
            self._storage.get_entity(eid) for eid in all_entities if self._storage.get_entity(eid)
        ]

        # Deduplicate relationships
        seen_rels = set()
        unique_rels = []
        for rel in all_relationships:
            key = (rel.source_entity, rel.target_entity, rel.relationship_type)
            if key not in seen_rels:
                seen_rels.add(key)
                unique_rels.append(rel)

        # Generate context
        context = await self._generate_local_context(
            query,
            result_entities,
            unique_rels[:20],
        )

        # Get communities for entities
        communities = []
        for entity in result_entities[:5]:
            comms = self._storage.get_communities_for_entity(entity.id)
            communities.extend(comms)

        return GraphRetrievalResult(
            entities=result_entities[:top_k],
            relationships=unique_rels[:top_k],
            communities=list({c.id: c for c in communities}.values())[:5],
            context=context,
            query_type="local",
            confidence=0.8 if entities else 0.0,
        )

    async def _global_search(
        self,
        query: str,
        top_k: int,
    ) -> GraphRetrievalResult:
        """
        Execute global search using community summaries.

        This is the key innovation of GraphRAG - using pre-computed
        community summaries to answer holistic questions.
        """
        # Get communities at the specified level
        stats = self._storage.get_stats()

        if stats.num_communities == 0:
            return GraphRetrievalResult(
                query_type="global",
                context="No community structure available. Run community detection first.",
            )

        # Collect community summaries
        all_communities = []
        for i in range(stats.num_communities):
            community = self._storage.get_community(f"comm_{self._community_level}_{i}")
            if community:
                all_communities.append(community)

        # If no communities at that level, try level 0
        if not all_communities:
            i = 0
            while True:
                community = self._storage.get_community(f"comm_0_{i}")
                if community:
                    all_communities.append(community)
                    i += 1
                else:
                    break

        if not all_communities:
            return GraphRetrievalResult(
                query_type="global",
                context="No communities found in the graph.",
            )

        # Map-reduce over community summaries
        context = await self._generate_global_context(query, all_communities)

        return GraphRetrievalResult(
            communities=all_communities[:top_k],
            context=context,
            query_type="global",
            confidence=0.7,
        )

    async def _generate_local_context(
        self,
        query: str,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> str:
        """Generate context from local graph elements."""
        # Build entity descriptions
        entity_text = "\n".join([f"- {e.name} ({e.type}): {e.description}" for e in entities[:15]])

        # Build relationship descriptions
        rel_text = "\n".join(
            [
                f"- {r.source_entity} --[{r.relationship_type}]--> {r.target_entity}"
                for r in relationships[:15]
            ]
        )

        prompt = f"""Based on the following knowledge graph elements, provide context relevant to the query.

Query: {query}

Entities:
{entity_text}

Relationships:
{rel_text}

Synthesize the information into a coherent context (2-3 paragraphs) that addresses the query:"""

        context = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500,
        )

        return context.strip()

    async def _generate_global_context(
        self,
        query: str,
        communities: list[Community],
    ) -> str:
        """
        Generate context from community summaries.

        Uses map-reduce approach from GraphRAG paper.
        """
        # Map: Get relevant information from each community
        community_texts = []
        for comm in communities:
            if comm.summary:
                community_texts.append(f"**{comm.title}**\n{comm.summary}")
            else:
                # Generate simple summary from title
                community_texts.append(
                    f"**{comm.title}**\nA group of {comm.size} related concepts."
                )

        # Reduce: Synthesize into final answer
        prompt = f"""Answer the query by synthesizing information from these concept groups:

Query: {query}

Concept Groups:
{chr(10).join(community_texts)}

Provide a comprehensive answer that draws on the themes and patterns across these groups:"""

        context = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=800,
        )

        return context.strip()


class HybridGraphRetriever(GraphRetriever):
    """
    Hybrid retriever combining graph and vector search.

    Uses vector search to find relevant chunks, then enriches
    with graph context for better understanding.
    """

    def __init__(
        self,
        graph_retriever: GraphRAGRetriever,
        vector_retriever: Any,  # RAG retriever
        graph_weight: float = 0.3,
    ):
        """
        Initialize hybrid retriever.

        Args:
            graph_retriever: GraphRAG retriever.
            vector_retriever: Standard vector retriever.
            graph_weight: Weight for graph results (0-1).
        """
        self._graph = graph_retriever
        self._vector = vector_retriever
        self._graph_weight = graph_weight

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> GraphRetrievalResult:
        """
        Hybrid retrieval combining graph and vector search.
        """
        # Get graph results
        graph_result = await self._graph.retrieve(query, top_k)

        # Get vector results (if retriever supports async)
        # vector_result = await self._vector.retrieve(query, top_k)

        # Combine contexts
        # For now, just return graph result
        # In production, would merge and rerank

        return graph_result
