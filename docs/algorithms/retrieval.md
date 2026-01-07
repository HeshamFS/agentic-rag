# Retrieval Methods

> **Finding Relevant Documents at Scale**
>
> This document covers dense retrieval, sparse retrieval (BM25), hybrid approaches, and query enhancement techniques like HyDE.

---

## Table of Contents

1. [Overview](#overview)
2. [Dense Retrieval](#dense-retrieval)
3. [Sparse Retrieval (BM25)](#sparse-retrieval-bm25)
4. [Hybrid Retrieval](#hybrid-retrieval)
5. [HyDE: Hypothetical Document Embeddings](#hyde-hypothetical-document-embeddings)
6. [Multi-Query Retrieval](#multi-query-retrieval)
7. [Comparison and Selection](#comparison-and-selection)

---

## Overview

Retrieval is the process of finding the most relevant documents (or chunks) for a given query. Modern RAG systems use multiple retrieval strategies:

| Method | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| Dense | Semantic embeddings | Meaning understanding | Rare terms |
| Sparse (BM25) | Term frequency | Exact matches | No semantics |
| Hybrid | Dense + Sparse | Best of both | Complexity |
| HyDE | Query expansion | Better matching | Latency |

---

## Dense Retrieval

Dense retrieval uses neural network embeddings to represent queries and documents in a shared vector space.

### Mathematical Foundation

#### Embedding Space

Both queries and documents are mapped to vectors in $\mathbb{R}^d$:

$$E_q = \text{encode}_\theta(q) \in \mathbb{R}^d$$
$$E_d = \text{encode}_\phi(d) \in \mathbb{R}^d$$

#### Similarity Metrics

**Cosine Similarity** (most common):

$$\text{sim}(q, d) = \frac{E_q \cdot E_d}{\|E_q\| \|E_d\|} = \frac{\sum_{i=1}^{d} E_{q,i} \cdot E_{d,i}}{\sqrt{\sum_{i=1}^{d} E_{q,i}^2} \cdot \sqrt{\sum_{i=1}^{d} E_{d,i}^2}}$$

**Dot Product** (when vectors are normalized):

$$\text{sim}(q, d) = E_q \cdot E_d = \sum_{i=1}^{d} E_{q,i} \cdot E_{d,i}$$

**Euclidean Distance** (inverse):

$$\text{sim}(q, d) = -\|E_q - E_d\|_2 = -\sqrt{\sum_{i=1}^{d} (E_{q,i} - E_{d,i})^2}$$

#### Retrieval

Find top-k documents:

$$\text{top-}k(q) = \text{argmax}_{d_1, ..., d_k \in D} \sum_{i=1}^{k} \text{sim}(q, d_i)$$

### Approximate Nearest Neighbor (ANN)

For large collections, exact search is infeasible. ANN algorithms provide fast approximate search:

| Algorithm | Complexity | Recall@10 |
|-----------|------------|-----------|
| Flat (exact) | O(n) | 100% |
| HNSW | O(log n) | 95-99% |
| IVF | O(√n) | 90-95% |
| PQ | O(n/k) | 85-90% |

**HNSW (Hierarchical Navigable Small World)** is used by default in Qdrant:

```python
# Qdrant HNSW configuration
collection_config = {
    "vectors": {
        "size": 1024,
        "distance": "Cosine"
    },
    "hnsw_config": {
        "m": 16,              # Connections per node
        "ef_construct": 100,  # Build-time search width
    }
}
```

### Implementation

```python
class DenseRetriever:
    def __init__(self, embedder, vectordb):
        self.embedder = embedder
        self.vectordb = vectordb

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 10
    ) -> RetrievalResult:
        # 1. Embed query
        query_embedding = await self.embedder.embed(query)

        # 2. Search vector database
        results = await self.vectordb.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 3. Return chunks with scores
        return RetrievalResult(
            chunks=[r.chunk for r in results],
            scores=[r.score for r in results],
            retrieval_type="dense"
        )
```

---

## Sparse Retrieval (BM25)

**BM25** (Best Matching 25) is a probabilistic ranking function based on term frequency.

> **Reference**: Robertson, S. E., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond."

### The BM25 Formula

$$\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:

| Symbol | Description |
|--------|-------------|
| $Q$ | Query with terms $q_1, ..., q_n$ |
| $D$ | Document |
| $f(q_i, D)$ | Term frequency of $q_i$ in $D$ |
| $\|D\|$ | Document length (in words) |
| $\text{avgdl}$ | Average document length in corpus |
| $k_1$ | Term frequency saturation parameter (default: 1.5) |
| $b$ | Document length normalization (default: 0.75) |

### IDF (Inverse Document Frequency)

$$\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$$

Where:
- $N$ = total number of documents
- $n(q_i)$ = number of documents containing $q_i$

### Parameter Effects

**k1 (Term Frequency Saturation)**:
```
k1 = 0   → Term frequency ignored (binary)
k1 = 1.5 → Moderate saturation (default)
k1 = ∞   → No saturation (raw TF)
```

**b (Length Normalization)**:
```
b = 0   → No length normalization
b = 0.75 → Standard normalization (default)
b = 1   → Full length normalization
```

### BM25 vs TF-IDF

| Aspect | TF-IDF | BM25 |
|--------|--------|------|
| TF scaling | Linear or log | Saturation curve |
| Length norm | Optional | Built-in |
| Probabilistic | No | Yes |
| Performance | Good | Better |

### Implementation

```python
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents: list[str]):
        # Tokenize documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        # Tokenize query
        query_tokens = query.lower().split()

        # Get scores
        scores = self.bm25.get_scores(query_tokens)

        # Return top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i], scores[i]) for i in top_indices]
```

---

## Hybrid Retrieval

Hybrid retrieval combines dense and sparse methods to leverage both semantic understanding and exact matching.

### Fusion Strategies

#### 1. Linear Combination

$$\text{score}_\text{hybrid}(q, d) = \alpha \cdot \text{score}_\text{dense}(q, d) + (1-\alpha) \cdot \text{score}_\text{sparse}(q, d)$$

Typical $\alpha = 0.5$ to $0.7$ (favor dense).

#### 2. Reciprocal Rank Fusion (RRF)

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Where:
- $R$ = set of ranking lists (dense, sparse)
- $k$ = constant (typically 60)
- $\text{rank}_r(d)$ = rank of document $d$ in list $r$

**Example**:
```
Dense ranking:  [A, B, C, D]  → A: rank 1, B: rank 2, ...
Sparse ranking: [C, A, D, B]  → C: rank 1, A: rank 2, ...

RRF(A) = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
RRF(B) = 1/(60+2) + 1/(60+4) = 0.0161 + 0.0156 = 0.0317
RRF(C) = 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
```

#### 3. Convex Combination with Normalization

First normalize scores to [0, 1]:

$$\text{norm}(s) = \frac{s - s_\text{min}}{s_\text{max} - s_\text{min}}$$

Then combine:

$$\text{score}_\text{hybrid} = \alpha \cdot \text{norm}(\text{score}_\text{dense}) + (1-\alpha) \cdot \text{norm}(\text{score}_\text{sparse})$$

### Implementation

```python
class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha=0.6):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha

    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        # Get both results (2x top_k for fusion)
        dense_results = await self.dense.retrieve(query, top_k=top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion
        doc_scores = defaultdict(float)
        k = 60

        for rank, (doc, _) in enumerate(dense_results, 1):
            doc_scores[doc.id] += 1 / (k + rank)

        for rank, (doc, _) in enumerate(sparse_results, 1):
            doc_scores[doc.id] += 1 / (k + rank)

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])

        return RetrievalResult(
            chunks=[self._get_doc(doc_id) for doc_id, _ in sorted_docs[:top_k]],
            scores=[score for _, score in sorted_docs[:top_k]],
            retrieval_type="hybrid"
        )
```

### When to Use Hybrid

| Scenario | Recommendation |
|----------|----------------|
| Technical queries with jargon | Hybrid (BM25 helps with exact terms) |
| Natural language questions | Dense often sufficient |
| Mixed query types | Hybrid recommended |
| Known vocabulary domain | BM25 may be sufficient |

---

## HyDE: Hypothetical Document Embeddings

**HyDE** improves retrieval by generating a "hypothetical" answer and using its embedding for search.

> **Reference**: Gao, L., et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

### The Problem HyDE Solves

Traditional retrieval matches **query embeddings** to **document embeddings**:

```
Query: "What causes climate change?"
Document: "Greenhouse gases trap heat in Earth's atmosphere..."

Problem: Query is short, document is long → embedding mismatch
```

### HyDE Approach

1. **Generate hypothetical answer** using LLM
2. **Embed the hypothetical answer**
3. **Search using answer embedding**

```
Query: "What causes climate change?"
     ↓ LLM generates
Hypothetical: "Climate change is primarily caused by greenhouse gas emissions
              from burning fossil fuels. These gases, including CO2 and methane,
              trap heat in the atmosphere, leading to global warming..."
     ↓ Embed
Search with hypothetical embedding → Find similar documents
```

### Mathematical Formulation

Given query $q$:

1. **Generate hypothesis**:
$$h = \text{LLM}(\text{prompt}(q))$$

2. **Embed hypothesis**:
$$E_h = \text{embed}(h)$$

3. **Retrieve using hypothesis embedding**:
$$\text{top-}k(q) = \text{argmax}_{d \in D} \text{sim}(E_h, E_d)$$

### Multiple Hypotheses

Generate $n$ hypotheses and average:

$$E_\text{HyDE} = \frac{1}{n} \sum_{i=1}^{n} \text{embed}(h_i)$$

This reduces variance from individual LLM generations.

### Implementation

```python
class HyDERetriever:
    def __init__(self, llm, embedder, vectordb, n_hypotheses=3):
        self.llm = llm
        self.embedder = embedder
        self.vectordb = vectordb
        self.n_hypotheses = n_hypotheses

    async def retrieve(self, query: str, collection: str, top_k: int = 10):
        # Generate hypothetical documents
        prompt = f"""Write a detailed passage that would answer this question:
        Question: {query}

        Passage:"""

        hypotheses = []
        for _ in range(self.n_hypotheses):
            response = await self.llm.generate(prompt)
            hypotheses.append(response.text)

        # Embed hypotheses
        embeddings = await self.embedder.embed_batch(hypotheses)

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)

        # Search with averaged embedding
        results = await self.vectordb.search(
            collection=collection,
            query_embedding=avg_embedding.tolist(),
            top_k=top_k
        )

        return RetrievalResult(
            chunks=results,
            retrieval_type="hyde"
        )
```

### Performance

From original HyDE paper:

| Dataset | Contriever | +HyDE | Improvement |
|---------|------------|-------|-------------|
| MS MARCO | 0.312 | 0.378 | +21.2% |
| Natural Questions | 0.319 | 0.401 | +25.7% |
| TriviaQA | 0.463 | 0.519 | +12.1% |

### Trade-offs

| Aspect | Without HyDE | With HyDE |
|--------|--------------|-----------|
| Latency | ~50ms | ~500ms (LLM call) |
| Cost | Low | Higher (LLM tokens) |
| Recall | Good | Better |
| Complexity | Low | Medium |

---

## Multi-Query Retrieval

**Multi-Query Retrieval** automates the process of prompt tuning by using an LLM to generate multiple queries from different perspectives for a single user input.

### The Problem: Query Sensitivity

Distance-based vector search is sensitive to small changes in wording. A user might phrase a question in a way that doesn't perfectly align with the document embeddings.

### The Solution: Query Variations

1. **LLM Generation**: Use an LLM to generate 3-5 variations of the original query.
2. **Parallel Retrieval**: Execute retrieval for all variations (including the original) in parallel.
3. **Union & Deduplication**: Combine all retrieved chunks and remove duplicates.

### Implementation

```python
class MultiQueryRetriever:
    async def _generate_queries(self, query: str, num_queries: int = 4) -> list[str]:
        prompt = f"""You are an AI language model assistant. Your task is to generate {num_queries} 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {query}"""
        
        response = await self.generator.generate_text(prompt)
        return [q.strip() for q in response.split("\n") if q.strip()]

    async def retrieve(self, query: str, collection: str, top_k: int = 10) -> RetrievalResult:
        # Generate variations
        queries = await self._generate_queries(query)
        queries.append(query) # Include original
        
        # Retrieve for all (simplified)
        all_chunks = []
        for q in queries:
            result = await self.base_retriever.retrieve(q, collection, top_k=top_k)
            all_chunks.extend(result.chunks)
            
        # Deduplicate and return top_k
        unique_chunks = self._deduplicate(all_chunks)
        return RetrievalResult(chunks=unique_chunks[:top_k], ...)
```

### Benefits

| Benefit | Impact |
|---------|--------|
| **Recall** | Significant improvement by covering multiple phrasings |
| **Robustness** | Less sensitive to user's specific wording |
| **Diversity** | Retrieves chunks that might be missed by a single query |

### Configuration

```python
pipeline = (
    PipelineBuilder()
    .with_retrieval(
        strategy="hybrid",
        use_multi_query=True,
        num_queries=4
    )
    .build()
)
```

---

## Comparison and Selection

### Performance Summary

| Method | Semantic Understanding | Exact Match | Latency | Cost |
|--------|----------------------|-------------|---------|------|
| Dense | High | Low | Fast | Low |
| BM25 | None | High | Fast | Low |
| Hybrid | High | High | Medium | Low |
| HyDE | Very High | Low | Slow | High |

### Selection Guide

```
Query Type Analysis:
├─ Contains technical terms / acronyms?
│  ├─ Yes → Hybrid or BM25
│  └─ No
│     ├─ Conceptual / semantic query?
│     │  ├─ Yes → Dense or HyDE
│     │  └─ No → Dense
└─ High accuracy required?
   ├─ Yes → HyDE (if latency acceptable)
   └─ No → Hybrid
```

### Configuration

```python
from agentic_rag.pipeline import PipelineBuilder

# Dense only
pipeline = PipelineBuilder().with_retrieval("dense", top_k=10)

# BM25 only
pipeline = PipelineBuilder().with_retrieval("bm25", top_k=10)

# Hybrid (recommended)
pipeline = PipelineBuilder().with_retrieval(
    "hybrid",
    top_k=10,
    dense_weight=0.6,
    fusion="rrf"
)

# With HyDE
pipeline = PipelineBuilder().with_retrieval(
    "hybrid",
    top_k=10,
    use_hyde=True,
    hyde_hypotheses=3
)
```

---

## References

1. Gao, L., et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

2. Robertson, S. E., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*.

3. Wikipedia. "Okapi BM25." [en.wikipedia.org/wiki/Okapi_BM25](https://en.wikipedia.org/wiki/Okapi_BM25)

4. Zilliz. (2024). "Better RAG with HyDE." [zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings)
