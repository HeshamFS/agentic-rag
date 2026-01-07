# GraphRAG: Knowledge Graph-Enhanced Retrieval

> **Answering Global Queries Through Graph Structure**
>
> This document covers Microsoft's GraphRAG approach including entity extraction, the Leiden community detection algorithm, and hierarchical summarization for global queries.

---

## Table of Contents

1. [Overview](#overview)
2. [Entity Extraction](#entity-extraction)
3. [Knowledge Graph Construction](#knowledge-graph-construction)
4. [Community Detection: Leiden Algorithm](#community-detection-leiden-algorithm)
5. [Community Summarization](#community-summarization)
6. [Query Modes](#query-modes)
7. [Configuration](#configuration)

---

## Overview

**GraphRAG** (Graph-based Retrieval Augmented Generation) enhances traditional RAG by building a knowledge graph from documents, enabling answers to "global" questions that require understanding the overall structure and themes of a document collection.

> **Reference**: Microsoft Research. (2024). "GraphRAG: New tool for complex data discovery now on GitHub." [microsoft.com/en-us/research/blog/graphrag](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/)

### Traditional RAG vs GraphRAG

| Aspect | Traditional RAG | GraphRAG |
|--------|----------------|----------|
| Query type | Local (specific facts) | Global + Local |
| "What are the main themes?" | Poor | Excellent |
| "What is X?" | Good | Good |
| Pre-processing | Embed chunks | Extract graph + communities |
| Context | Individual chunks | Structured knowledge |

### GraphRAG Architecture

```
Documents
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│                     Indexing Pipeline                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Text Chunking                                          │
│     └── Split into processable units                       │
│                                                             │
│  2. Entity Extraction (LLM)                                │
│     └── Extract entities + relationships                   │
│                                                             │
│  3. Knowledge Graph Construction                           │
│     └── Build graph from entities & relationships          │
│                                                             │
│  4. Community Detection (Leiden)                           │
│     └── Hierarchical clustering of entities                │
│                                                             │
│  5. Community Summarization (LLM)                          │
│     └── Generate summaries for each community              │
│                                                             │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌───────────────┐
                     │   Query Time  │
                     └───────┬───────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
    ┌───────────────┐                 ┌───────────────┐
    │  Local Query  │                 │ Global Query  │
    │ (Entity-based)│                 │ (Map-Reduce)  │
    └───────────────┘                 └───────────────┘
```

---

## Entity Extraction

Entity extraction identifies named entities (people, organizations, concepts) and their relationships from text using LLM prompts.

### Entity Model

```python
class Entity(BaseModel):
    """A node in the knowledge graph."""

    id: str
    name: str                    # Entity name
    type: str                    # Person, Organization, Concept, etc.
    description: str             # LLM-generated description
    source_chunk_ids: list[str]  # Source documents
```

### Relationship Model

```python
class Relationship(BaseModel):
    """An edge in the knowledge graph."""

    source_entity: str       # Source entity name
    target_entity: str       # Target entity name
    relationship_type: str   # e.g., "works_for", "related_to"
    description: str         # Relationship description
    weight: float            # Strength (higher = mentioned more)
```

### Extraction Process

The extraction uses domain-specific LLM prompts:

```
Input Text: "The Transformer architecture, introduced by Vaswani et al.
at Google in 2017, revolutionized NLP through the attention mechanism."

Extracted:
├── Entities:
│   ├── Transformer (Technology): "A neural network architecture"
│   ├── Vaswani (Person): "Researcher who introduced Transformers"
│   ├── Google (Organization): "Company where Transformers were developed"
│   └── Attention (Concept): "Core mechanism in Transformers"
│
└── Relationships:
    ├── Vaswani --[developed]--> Transformer
    ├── Google --[employer_of]--> Vaswani
    └── Transformer --[uses]--> Attention
```

### Entity Merging

Duplicate entities from different chunks are merged:

$$\text{merged\_entity} = \begin{cases}
\text{combine source\_chunks} \\
\text{keep longer description} \\
\text{normalize name to lowercase}
\end{cases}$$

Relationship weights increase with each duplicate:

$$w_{\text{merged}} = \sum_{i=1}^{n} w_i$$

Where $n$ is the number of times the relationship appears across chunks.

---

## Knowledge Graph Construction

### Graph Structure

The knowledge graph is a directed multigraph:

$$G = (V, E)$$

Where:
- $V$ = set of entities (nodes)
- $E$ = set of relationships (edges)

### Adjacency Representation

For entity similarity and community detection, we use an adjacency matrix:

$$A_{ij} = \begin{cases}
w_{ij} & \text{if edge exists between } v_i \text{ and } v_j \\
0 & \text{otherwise}
\end{cases}$$

Where $w_{ij}$ is the edge weight (relationship strength).

### Graph Statistics

```python
@dataclass
class GraphStats:
    num_entities: int
    num_relationships: int
    num_communities: int
    avg_degree: float  # Average connections per entity
    density: float     # E / (V * (V-1))
```

---

## Community Detection: Leiden Algorithm

The **Leiden algorithm** detects communities (clusters of related entities) in the knowledge graph. It improves upon the Louvain method by guaranteeing well-connected communities.

> **Reference**: Traag, V.A., Waltman, L., & van Eck, N.J. (2019). "From Louvain to Leiden: guaranteeing well-connected communities." [Scientific Reports](https://www.nature.com/articles/s41598-019-41695-z)

### Modularity Optimization

The algorithm optimizes **modularity** $Q$, which measures the quality of community structure:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \gamma \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where:
- $A_{ij}$ = edge weight between nodes $i$ and $j$
- $k_i$ = degree of node $i$ (sum of edge weights)
- $m$ = total edge weight in the graph: $m = \frac{1}{2} \sum_{ij} A_{ij}$
- $\gamma$ = resolution parameter (controls community size)
- $c_i$ = community assignment of node $i$
- $\delta(c_i, c_j)$ = 1 if $c_i = c_j$, 0 otherwise

### Resolution Parameter

The resolution parameter $\gamma$ controls community granularity:

| $\gamma$ Value | Effect |
|---------------|--------|
| $\gamma < 1$ | Fewer, larger communities |
| $\gamma = 1$ | Standard modularity |
| $\gamma > 1$ | More, smaller communities |

### Modularity Gain Formula

When moving node $i$ to community $C$, the modularity gain is:

$$\Delta Q = \frac{k_{i,in}}{2m} - \gamma \frac{\Sigma_{tot} \cdot k_i}{2m^2}$$

Where:
- $k_{i,in}$ = sum of edge weights from node $i$ to nodes in community $C$
- $\Sigma_{tot}$ = sum of all edge weights for nodes in community $C$
- $k_i$ = degree of node $i$

### Leiden Algorithm Phases

```
┌─────────────────────────────────────────────────────────────┐
│                    Leiden Algorithm                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Local Moving                                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ For each node:                                          │ │
│  │   • Calculate ΔQ for moving to each neighbor's community│ │
│  │   • Move to community with highest positive ΔQ          │ │
│  │   • Repeat until no improvement                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  Phase 2: Refinement (Leiden only)                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ • Check each community for well-connectedness           │ │
│  │ • Split poorly connected communities                    │ │
│  │ • Guarantees no disconnected subcommunities             │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  Phase 3: Aggregation                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ • Merge nodes in same community into super-node         │ │
│  │ • Aggregate edges between communities                   │ │
│  │ • Repeat from Phase 1 on aggregated graph               │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Hierarchical Communities

Leiden produces hierarchical communities at multiple levels:

```
Level 2 (Coarse):   [Machine Learning]────[NLP]
                          │                 │
                    ┌─────┴─────┐     ┌─────┴─────┐
Level 1:            │           │     │           │
              [Deep Learning] [Classical ML] [Text] [Speech]
                    │           │           │        │
              ┌─────┴─────┐     │     ┌─────┴───┐   │
Level 0:      │     │     │     │     │    │    │   │
            [CNN] [RNN] [Transformer] [SVM] [BERT] [T5] [Whisper]
```

### Implementation

```python
class LeidenCommunityDetector:
    def __init__(
        self,
        resolution: float = 1.0,      # γ parameter
        max_levels: int = 3,          # Hierarchy depth
        min_community_size: int = 2,  # Minimum entities
    ):
        ...

    def detect(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> CommunityHierarchy:
        # Build NetworkX graph
        G = nx.Graph()
        for entity in entities:
            G.add_node(entity.id, ...)
        for rel in relationships:
            G.add_edge(source_id, target_id, weight=rel.weight)

        # Run Leiden (falls back to Louvain if leidenalg unavailable)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self._resolution,
        )

        # Build hierarchy
        return self._build_hierarchy(entities, partition)
```

### Complexity

| Operation | Time Complexity |
|-----------|-----------------|
| Leiden algorithm | $O(L \cdot |E|)$ |
| Space | $O(|V| + |E|)$ |

Where $L$ is the number of iterations.

---

## Community Summarization

Each community receives an LLM-generated summary, enabling global query answering.

### Summary Generation

```python
class CommunitySummarizer:
    async def summarize_community(
        self,
        community: Community,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> str:
        # Build context from entities and relationships
        entity_texts = [
            f"- {e.name} ({e.type}): {e.description}"
            for e in entities
        ]

        rel_texts = [
            f"- {r.source_entity} {r.relationship_type} {r.target_entity}"
            for r in relationships
        ]

        prompt = f"""Summarize this group of related concepts:

Entities:
{entity_texts}

Relationships:
{rel_texts}

Write a 2-3 sentence summary describing what this group
represents, their main themes, and key relationships."""

        return await llm.generate(prompt)
```

### Hierarchical Summarization

Summaries are generated bottom-up:

1. **Level 0**: Summarize leaf communities (5-10 entities each)
2. **Level 1**: Summarize using child community summaries
3. **Level N**: Root summaries capture global themes

```
Level 2 Summary: "This collection covers machine learning and
natural language processing, with a focus on neural network
architectures and their applications to text understanding."
     │
     ├── Level 1 Summary: "Deep learning approaches including
     │   CNNs, RNNs, and Transformers for feature learning."
     │        │
     │        ├── Level 0: "Convolutional networks for image..."
     │        └── Level 0: "Attention mechanisms enable..."
     │
     └── Level 1 Summary: "NLP techniques spanning traditional
         and neural approaches for text processing."
              │
              └── Level 0: "BERT and T5 represent..."
```

---

## Query Modes

GraphRAG supports two query modes optimized for different question types.

### Local Search

For entity-specific queries like "What is X?" or "How does Y work?"

```
Query: "What is the Transformer architecture?"
    │
    ▼
┌───────────────────────────────────────┐
│           Local Search                 │
├───────────────────────────────────────┤
│ 1. Search entities matching "Transformer" │
│ 2. Get entity neighborhoods (1-2 hops)   │
│ 3. Collect relevant relationships        │
│ 4. Generate context from graph elements  │
└───────────────────────────────────────┘
    │
    ▼
Result: Transformer entity + related entities
        (Attention, BERT, GPT) + relationships
```

### Global Search

For thematic queries like "What are the main themes?" or "Compare the approaches."

Uses a **map-reduce** approach over community summaries:

```
Query: "What are the main themes in this document collection?"
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                     Global Search                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  MAP Phase:                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│  │ Community 1 │ │ Community 2 │ │ Community 3 │             │
│  │  Summary    │ │  Summary    │ │  Summary    │             │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘             │
│         │               │               │                     │
│         ▼               ▼               ▼                     │
│  ┌────────────────────────────────────────────┐              │
│  │        Collect relevant summaries           │              │
│  └───────────────────┬────────────────────────┘              │
│                      │                                        │
│  REDUCE Phase:       ▼                                        │
│  ┌────────────────────────────────────────────┐              │
│  │  LLM synthesizes final answer from         │              │
│  │  community summaries                        │              │
│  └───────────────────┬────────────────────────┘              │
│                      │                                        │
└──────────────────────┼────────────────────────────────────────┘
                       ▼
Result: "The main themes are: (1) Deep learning architectures
        including Transformers... (2) NLP applications..."
```

### Query Classification

The retriever automatically classifies queries:

```python
async def _classify_query(self, query: str) -> str:
    prompt = """Classify this query as either "local" or "global":

- LOCAL: Asks about specific entities, facts, or details
  Examples: "What is the Transformer architecture?"

- GLOBAL: Asks about overall themes, patterns, summaries
  Examples: "What are the main themes?", "Compare approaches"

Query: "{query}"
Answer with just "local" or "global":"""

    response = await llm.generate(prompt)
    return "global" if "global" in response.lower() else "local"
```

---

## Configuration

### Pipeline Builder

```python
from agentic_rag.pipeline import PipelineBuilder
from agentic_rag.graph import GraphRAGRetriever, LeidenCommunityDetector

pipeline = (
    PipelineBuilder()
    .with_graphrag(
        enabled=True,
        graph_path="knowledge_graph.json"
    )
    .with_generator(provider="claude", model="claude-sonnet-4-5-20250929")
    .build()
)

# Query with graph
result = await pipeline.query(
    "What are the main themes in this research?",
    collection="papers",
    use_graph=True,
)
```

### Community Detection Settings

```python
from agentic_rag.graph import LeidenCommunityDetector

detector = LeidenCommunityDetector(
    resolution=1.5,           # Higher = smaller communities
    max_levels=4,             # Hierarchy depth
    min_community_size=3,     # Filter tiny communities
)

hierarchy = detector.detect(entities, relationships)
```

### Entity Extractor Settings

```python
from agentic_rag.graph import LLMEntityExtractor

extractor = LLMEntityExtractor(
    generator=generator,
    entity_types=["Person", "Organization", "Concept", "Location"],
    max_entities_per_chunk=50,
    min_entity_mentions=1,
)

result = await extractor.extract(chunk)
```

---

## Performance Considerations

### Indexing Costs

| Operation | Cost Factor |
|-----------|-------------|
| Entity extraction | 1 LLM call per chunk |
| Community summarization | 1 LLM call per community |
| Graph construction | O(entities + relationships) |

### Query Costs

| Query Type | LLM Calls |
|------------|-----------|
| Local search | 1 (context generation) |
| Global search | 1-2 (map-reduce) |

### Optimization Tips

1. **Batch entity extraction** to reduce API calls
2. **Cache community summaries** - they rarely change
3. **Use appropriate resolution** - too fine creates many small communities
4. **Pre-filter entities** by type for domain-specific graphs

---

## References

1. Microsoft Research. (2024). "GraphRAG: New tool for complex data discovery now on GitHub." [microsoft.com/en-us/research/blog/graphrag](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/)

2. Traag, V.A., Waltman, L., & van Eck, N.J. (2019). "From Louvain to Leiden: guaranteeing well-connected communities." [Scientific Reports](https://www.nature.com/articles/s41598-019-41695-z)

3. Wikipedia. "Leiden algorithm." [en.wikipedia.org/wiki/Leiden_algorithm](https://en.wikipedia.org/wiki/Leiden_algorithm)

4. Weaviate. (2024). "Exploring RAG and GraphRAG: Understanding when and how to use both." [weaviate.io/blog/graph-rag](https://weaviate.io/blog/graph-rag)

5. Neo4j. (2024). "Implementing 'From Local to Global' GraphRAG With Neo4j and LangChain." [neo4j.com/blog/developer/global-graphrag-neo4j-langchain](https://neo4j.com/blog/developer/global-graphrag-neo4j-langchain/)

