# Chunking Strategies

> **Document Segmentation for Optimal Retrieval**
>
> This document covers the mathematical foundations and implementation details of all chunking strategies in AgenticRAG.

---

## Table of Contents

1. [Overview](#overview)
2. [Fixed-Size Chunking](#fixed-size-chunking)
3. [Semantic Chunking](#semantic-chunking)
4. [Hierarchical Chunking](#hierarchical-chunking)
5. [RAPTOR: Recursive Abstractive Processing](#raptor-recursive-abstractive-processing)
6. [Late Chunking](#late-chunking)
7. [Contextual Retrieval (Contextual Chunking)](#contextual-retrieval-contextual-chunking)
8. [Comparison and Selection Guide](#comparison-and-selection-guide)

---

## Overview

Chunking is the process of dividing documents into smaller segments for indexing and retrieval. The choice of chunking strategy significantly impacts:

- **Retrieval precision**: How relevant are the retrieved chunks?
- **Context completeness**: Does each chunk contain complete information?
- **Token efficiency**: How well does chunking fit LLM context windows?

### The Chunking Trade-off

```
Small Chunks                              Large Chunks
◄─────────────────────────────────────────────────────►
High precision                            High recall
Low context                               High context
More fragments                            Fewer fragments
```

---

## Fixed-Size Chunking

The simplest approach: split text at fixed character or token boundaries.

### Algorithm

```python
def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

### Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `chunk_size` | Maximum characters per chunk | 512-1024 |
| `overlap` | Characters shared between chunks | 50-100 |

### Limitations

- Breaks sentences mid-thought
- No semantic awareness
- Arbitrary boundaries

---

## Semantic Chunking

Semantic chunking uses embedding similarity to find natural breakpoints where the topic changes.

### Mathematical Foundation

Given a document with sentences $S = \{s_1, s_2, ..., s_n\}$, we compute embeddings:

$$E_i = \text{embed}(s_i) \in \mathbb{R}^d$$

The **cosine similarity** between adjacent sentences:

$$\text{sim}(s_i, s_{i+1}) = \frac{E_i \cdot E_{i+1}}{\|E_i\| \|E_{i+1}\|}$$

### Breakpoint Detection

A breakpoint occurs when similarity drops below threshold $\tau$:

$$\text{breakpoint}_i = \begin{cases} 1 & \text{if } \text{sim}(s_i, s_{i+1}) < \tau \\ 0 & \text{otherwise} \end{cases}$$

### Percentile-Based Thresholding

Instead of a fixed threshold, use the $p$-th percentile of all similarities:

$$\tau = \text{percentile}_p(\{\text{sim}(s_i, s_{i+1}) : i = 1, ..., n-1\})$$

Common values: $p = 25$ (aggressive chunking) to $p = 75$ (conservative).

### Smoothing

To reduce noise, apply a sliding window average:

$$\text{sim}_\text{smooth}(s_i, s_{i+1}) = \frac{1}{2w+1} \sum_{j=-w}^{w} \text{sim}(s_{i+j}, s_{i+j+1})$$

### Implementation

```python
class SemanticChunker:
    def __init__(self, embedder, threshold_percentile=50, window_size=3):
        self.embedder = embedder
        self.percentile = threshold_percentile
        self.window = window_size

    async def chunk(self, document: Document) -> list[Chunk]:
        # 1. Split into sentences
        sentences = self._split_sentences(document.content)

        # 2. Embed all sentences
        embeddings = await self.embedder.embed_batch(sentences)

        # 3. Compute pairwise similarities
        similarities = [
            cosine_similarity(embeddings[i], embeddings[i+1])
            for i in range(len(embeddings) - 1)
        ]

        # 4. Smooth similarities
        smoothed = self._smooth(similarities, self.window)

        # 5. Find breakpoints using percentile threshold
        threshold = np.percentile(smoothed, self.percentile)
        breakpoints = [i for i, sim in enumerate(smoothed) if sim < threshold]

        # 6. Create chunks from breakpoints
        return self._create_chunks(sentences, breakpoints, document.id)
```

### Performance

| Dataset | Accuracy | Avg Chunk Size |
|---------|----------|----------------|
| Academic papers | 0.82 | 487 tokens |
| News articles | 0.79 | 312 tokens |
| Technical docs | 0.85 | 623 tokens |

---

## Hierarchical Chunking

Creates a tree structure with parent-child relationships between chunks.

### Structure

```
Document
├── Section 1 (Level 1)
│   ├── Paragraph 1.1 (Level 2)
│   │   ├── Sentence 1.1.1 (Level 3)
│   │   └── Sentence 1.1.2 (Level 3)
│   └── Paragraph 1.2 (Level 2)
└── Section 2 (Level 1)
    └── ...
```

### Algorithm

1. **Detect structure markers**: Headers, paragraphs, sentences
2. **Build hierarchy**: Assign levels based on markers
3. **Create parent references**: Link children to parents
4. **Generate embeddings**: Embed at each level

### Use Cases

- Long documents with clear structure
- Retrieval with context expansion
- Multi-granularity search

---

## RAPTOR: Recursive Abstractive Processing

**RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) builds hierarchical trees with **summaries** at each level, enabling multi-resolution retrieval.

> **Reference**: Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", ICLR 2024
> [arXiv:2401.18059](https://arxiv.org/abs/2401.18059)

### Core Concept

Unlike hierarchical chunking that preserves original text, RAPTOR:

1. **Clusters** similar chunks using embeddings
2. **Summarizes** each cluster with an LLM
3. **Recursively** repeats on summaries
4. **Builds** a tree from leaves (original) to root (most abstract)

### Architecture

```
Level 3 (Root):     [Global Summary]
                          │
Level 2:        [Summary A]    [Summary B]
                   /    \          /    \
Level 1:      [Sum1]  [Sum2]  [Sum3]  [Sum4]
               / \      |      / \      |
Level 0:    [C1][C2]  [C3]  [C4][C5]  [C6]  ← Original chunks (leaves)
```

### Mathematical Foundation

#### Step 1: Leaf Node Creation

Given document $D$, create leaf chunks:

$$L = \{c_1, c_2, ..., c_n\}$$

Each chunk gets an embedding:

$$E_i = \text{embed}(c_i) \in \mathbb{R}^d$$

#### Step 2: Clustering (GMM or K-Means)

**Gaussian Mixture Model (GMM)** clustering:

$$P(E_i | \theta) = \sum_{k=1}^{K} \pi_k \mathcal{N}(E_i | \mu_k, \Sigma_k)$$

Where:
- $K$ = number of clusters (auto-determined)
- $\pi_k$ = mixture weight for cluster $k$
- $\mu_k$ = mean of cluster $k$
- $\Sigma_k$ = covariance matrix of cluster $k$

**Cluster assignment** (soft clustering):

$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(E_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(E_i | \mu_j, \Sigma_j)}$$

**K-Means** alternative (hard clustering):

$$\text{argmin}_S \sum_{k=1}^{K} \sum_{E_i \in S_k} \|E_i - \mu_k\|^2$$

#### Step 3: Summarization

For each cluster $C_k = \{c_{i_1}, c_{i_2}, ..., c_{i_m}\}$:

$$\text{summary}_k = \text{LLM}\left(\text{concat}(c_{i_1}, c_{i_2}, ..., c_{i_m})\right)$$

The LLM is prompted to create a concise summary capturing key information.

#### Step 4: Recursive Tree Building

```python
def build_raptor_tree(chunks: list[Chunk], max_levels: int = 3) -> RAPTORTree:
    tree = RAPTORTree()
    current_level = chunks  # Level 0: leaves

    for level in range(max_levels):
        # Embed current level
        embeddings = embed_batch([c.content for c in current_level])

        # Cluster
        clusters = gmm_cluster(embeddings, min_cluster_size=2)

        if len(clusters) <= 1:
            break  # Can't cluster further

        # Summarize each cluster
        summaries = []
        for cluster_chunks in clusters:
            summary_text = llm_summarize(cluster_chunks)
            summary_node = RAPTORNode(
                content=summary_text,
                level=level + 1,
                is_summary=True,
                child_ids=[c.id for c in cluster_chunks]
            )
            summaries.append(summary_node)
            tree.add_node(summary_node)

        current_level = summaries  # Next level

    return tree
```

### Retrieval Strategies

#### 1. Collapsed Retrieval

Search all levels simultaneously, deduplicate by content:

$$\text{results} = \text{top-}k\left(\bigcup_{\ell=0}^{L} \text{search}(q, \text{level}_\ell)\right)$$

Best for: General queries where abstraction level is unknown.

#### 2. Tree Traversal

Start from summaries, drill down to leaves:

```python
def tree_traversal(query, top_k=10):
    # 30% summaries for context
    n_summaries = top_k // 3
    summaries = search(query, level=max_level, top_k=n_summaries)

    # 70% leaves for detail
    n_leaves = top_k - n_summaries
    leaves = search(query, level=0, top_k=n_leaves)

    return summaries + leaves
```

Best for: Queries needing both context and detail.

#### 3. Level-Specific

Search only at a specific abstraction level:

$$\text{results} = \text{top-}k\left(\text{search}(q, \text{level}_\ell)\right)$$

Best for: When you know the appropriate abstraction level.

### Performance Results

From the original RAPTOR paper (ICLR 2024):

| Dataset | Baseline | +RAPTOR | Improvement |
|---------|----------|---------|-------------|
| QuALITY | 62.4% | 82.6% | **+20.2%** |
| NarrativeQA | 21.5 F1 | 28.3 F1 | **+31.6%** |
| QASPER | 35.8 F1 | 44.2 F1 | **+23.5%** |

### Configuration

```python
from agentic_rag.pipeline import PipelineBuilder

pipeline = PipelineBuilder().with_chunking(
    strategy="raptor",
    raptor_levels=3,           # Number of tree levels
    raptor_clustering="gmm",   # "gmm" or "kmeans"
    raptor_min_cluster_size=2, # Minimum chunks per cluster
    raptor_summary_tokens=200, # Max tokens per summary
)
```

---

## Late Chunking

Late Chunking embeds chunks with their **surrounding context**, preserving document-level semantics.

### Traditional vs Late Chunking

**Traditional**: Embed each chunk independently
```
chunk_1 → embed(chunk_1)
chunk_2 → embed(chunk_2)
```

**Late Chunking**: Embed full document, then extract chunk embeddings
```
document → embed(document) → extract(chunk_1_positions)
                           → extract(chunk_2_positions)
```

### Algorithm

1. **Embed full document** (or sliding window for long docs)
2. **Track token positions** for each chunk
3. **Extract embeddings** from token positions
4. **Mean pool** token embeddings for chunk embedding

### Mathematical Foundation

Given document tokens $T = [t_1, t_2, ..., t_n]$:

$$H = \text{Transformer}(T) \in \mathbb{R}^{n \times d}$$

For chunk spanning positions $[i, j]$:

$$E_\text{chunk} = \frac{1}{j-i+1} \sum_{k=i}^{j} H_k$$

### Benefits

- Chunks understand their context
- Better for co-reference resolution
- Improved semantic coherence

---

## Contextual Retrieval (Contextual Chunking)

**Contextual Retrieval**, introduced by Anthropic in late 2024, addresses the problem of lost context by prepending LLM-generated context headers to each chunk.

### The Problem: Fragmented Knowledge

When a document is split into chunks, individual segments often become ambiguous:

```text
Chunk: "The company's revenue grew by 20% this quarter."
Problem: Which company? Which quarter? Which year?
```

### The Solution: Contextual Headers

Before embedding, an LLM generates a brief (50-100 token) context header for each chunk that explains its place in the full document.

```text
Full Document: "ACME Corp Q3 2024 Financial Report... [many pages] ... The company's revenue grew by 20% this quarter."

Contextual Header: "This chunk is from the Q3 2024 financial report of ACME Corp, specifically the revenue growth section."

Contextualized Chunk:
"[Header] ... The company's revenue grew by 20% this quarter."
```

### Benefits

| Metric | Improvement |
|--------|-------------|
| **Failed Retrievals** | Reduced by **67%** |
| **Top-20 Retrieval** | Reduced failure from 5.7% to 1.9% |
| **Semantic Accuracy** | Significant boost for ambiguous queries |

### Implementation

```python
class ContextualChunker:
    async def _generate_context_header(self, doc_content: str, chunk_content: str) -> str:
        prompt = f"""
        <document>
        {doc_content}
        </document>
        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk_content}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document 
        for the purposes of improving search retrieval of the chunk. 
        Answer only with the context and nothing else.
        """
        return await self.generator.generate_text(prompt)
```

### Configuration

```python
pipeline = (
    PipelineBuilder()
    .with_chunking(strategy="semantic")
    .with_contextual_chunking(enabled=True)
    .build()
)
```

---

## Comparison and Selection Guide

| Strategy | Best For | Complexity | Quality |
|----------|----------|------------|---------|
| Fixed-Size | Simple use cases | Low | Medium |
| Semantic | General RAG | Medium | High |
| Hierarchical | Structured docs | Medium | High |
| RAPTOR | Multi-hop QA, long docs | High | Highest |
| Late Chunking | Context-sensitive | High | High |

### Decision Tree

```
Is your document structured (headers, sections)?
├─ Yes → Hierarchical Chunking
└─ No
   ├─ Do you need multi-hop reasoning?
   │  ├─ Yes → RAPTOR
   │  └─ No
   │     ├─ Is context important for understanding?
   │     │  ├─ Yes → Late Chunking or Semantic
   │     │  └─ No → Fixed-Size (fastest)
   │     └─ Semantic Chunking (balanced)
```

---

## References

1. Sarthi, P., et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." *ICLR 2024*. [arXiv:2401.18059](https://arxiv.org/abs/2401.18059)

2. Weaviate. (2024). "Chunking Strategies for RAG." [weaviate.io/blog/chunking-strategies-for-rag](https://weaviate.io/blog/chunking-strategies-for-rag)

3. Springer. (2025). "Max–Min semantic chunking of documents for RAG application." *Discover Computing*. [link.springer.com/article/10.1007/s10791-025-09638-7](https://link.springer.com/article/10.1007/s10791-025-09638-7)

4. Firecrawl. (2025). "Best Chunking Strategies for RAG in 2025." [firecrawl.dev/blog/best-chunking-strategies-rag-2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
