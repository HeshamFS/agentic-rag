"""
GraphRAG module for knowledge graph-based retrieval.

GraphRAG addresses a fundamental limitation of vector-only retrieval:
the inability to answer "global queries" like "what are the main themes?"

Components:
- EntityExtractor: Extract entities and relationships from text
- CommunityDetector: Build community hierarchies using Leiden algorithm
- GraphRetriever: Graph-based retrieval for global and local queries
- GraphStorage: Store and query knowledge graphs
"""

from agentic_rag.graph.community import (
    Community,
    CommunityDetector,
    LeidenCommunityDetector,
)
from agentic_rag.graph.extractor import (
    Entity,
    EntityExtractor,
    LLMEntityExtractor,
    Relationship,
)
from agentic_rag.graph.retriever import (
    GraphRAGRetriever,
    GraphRetriever,
)
from agentic_rag.graph.storage import (
    GraphStorage,
    NetworkXStorage,
)

__all__ = [
    # Extractor
    "Entity",
    "Relationship",
    "EntityExtractor",
    "LLMEntityExtractor",
    # Community
    "Community",
    "CommunityDetector",
    "LeidenCommunityDetector",
    # Retriever
    "GraphRetriever",
    "GraphRAGRetriever",
    # Storage
    "GraphStorage",
    "NetworkXStorage",
]
