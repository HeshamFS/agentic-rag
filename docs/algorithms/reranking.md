# Reranking Methods

> **Improving Retrieval Quality with Neural Rerankers**
>
> This document covers ColBERT's late interaction mechanism, cross-encoders, and other reranking strategies.

---

## Table of Contents

1. [Overview](#overview)
2. [Two-Stage Retrieval](#two-stage-retrieval)
3. [ColBERT: Late Interaction](#colbert-late-interaction)
4. [Cross-Encoders](#cross-encoders)
5. [Comparison](#comparison)
6. [Configuration](#configuration)

---

## Overview

Reranking is a **second-stage** process that refines the initial retrieval results using more sophisticated (and expensive) models.

### Why Rerank?

| Stage | Model | Speed | Quality |
|-------|-------|-------|---------|
| First-stage (Retrieval) | Bi-encoder | Fast (ms) | Good |
| Second-stage (Reranking) | Late interaction / Cross-encoder | Slow (100ms+) | Excellent |

The key insight: **compute expensive models only on top candidates**.

---

## Two-Stage Retrieval

```
Query
  │
  ▼
┌─────────────────────────────────────┐
│  First Stage: Fast Retrieval        │
│  (Bi-encoder, 1M+ documents)        │
│  Returns: Top 100-500 candidates    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Second Stage: Reranking            │
│  (ColBERT/Cross-encoder, ~100 docs) │
│  Returns: Top 10-20 reranked        │
└──────────────────┬──────────────────┘
                   │
                   ▼
            Final Results
```

---

## ColBERT: Late Interaction

**ColBERT** (Contextualized Late Interaction over BERT) uses **token-level** representations for fine-grained matching.

> **Reference**: Khattab, O., & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR 2020. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)

### Core Concept

Unlike bi-encoders that compress the entire text into a single vector, ColBERT:

1. **Keeps all token embeddings** for query and document
2. **Computes fine-grained similarities** between all token pairs
3. **Aggregates using MaxSim** operation

### Architecture

```
Query: "What is attention?"
       │
       ▼
   ┌───────┐
   │ BERT  │
   └───┬───┘
       │
       ▼
   Token Embeddings: [E_what, E_is, E_attention, E_?]
                           │
                           │    MaxSim for each query token
                           ▼
   Document Embeddings: [E_the, E_transformer, E_uses, E_attention, E_mechanism]
                                    ↓
                           Score = sum of max similarities
```

### Mathematical Foundation

#### Token Embeddings

For query $q$ with tokens $[q_1, q_2, ..., q_m]$:

$$E_q = \text{BERT}(q) \in \mathbb{R}^{m \times d}$$

For document $d$ with tokens $[d_1, d_2, ..., d_n]$:

$$E_d = \text{BERT}(d) \in \mathbb{R}^{n \times d}$$

Where $d$ is the embedding dimension (typically 128 for ColBERT).

#### MaxSim Operation

For each query token $i$, find the maximum similarity with any document token:

$$\text{MaxSim}(q_i, d) = \max_{j \in [1,n]} \text{sim}(E_{q_i}, E_{d_j})$$

The similarity function is typically **cosine similarity** or **dot product**:

$$\text{sim}(E_{q_i}, E_{d_j}) = \frac{E_{q_i} \cdot E_{d_j}}{\|E_{q_i}\| \|E_{d_j}\|}$$

#### Final Score

Sum all MaxSim scores:

$$\text{score}(q, d) = \sum_{i=1}^{m} \text{MaxSim}(q_i, d) = \sum_{i=1}^{m} \max_{j \in [1,n]} \text{sim}(E_{q_i}, E_{d_j})$$

### Visual Explanation

```
Query tokens:      what    is    attention    ?
                    │      │        │         │
                    ▼      ▼        ▼         ▼
                 ┌──────────────────────────────┐
Document         │  Similarity Matrix           │
tokens:          │  (m × n)                     │
                 │                              │
  the       ──▶  │ 0.12   0.08    0.05    0.03  │
  transformer──▶ │ 0.15   0.12    0.23    0.04  │
  uses      ──▶  │ 0.18   0.45    0.11    0.06  │
  attention ──▶  │ 0.21   0.14    0.92    0.05  │  ← Max for "attention"
  mechanism ──▶  │ 0.09   0.11    0.34    0.02  │
                 └──────────────────────────────┘
                    │      │        │         │
                    ▼      ▼        ▼         ▼
  Max per query:  0.21   0.45     0.92      0.06

  Final Score = 0.21 + 0.45 + 0.92 + 0.06 = 1.64
```

### Why MaxSim Works

1. **Handles synonyms**: "car" in query can match "automobile" in document
2. **Robust to length**: Works regardless of document length
3. **Interpretable**: Can trace which tokens contributed to the score
4. **Soft matching**: Allows approximate matches with high similarity

### ColBERT Variants

| Model | Use Case | Best For |
|-------|----------|----------|
| `jinaai/jina-reranker-v2-base-multilingual` | Multilingual | General purpose |
| `jinaai/jina-colbert-v2` | Late Interaction | High quality, 8K context |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Cross-Encoder | Maximum accuracy |
| `BAAI/bge-reranker-v2-m3` | Lightweight | Multilingual efficiency |

### Implementation

```python
import torch
from transformers import AutoTokenizer, AutoModel

class ColBERTReranker:
    def __init__(self, model_name="colbert-ir/colbertv2.0", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to token embeddings."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Normalize embeddings
        embeddings = outputs.last_hidden_state
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        return embeddings

    def maxsim(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        """Compute MaxSim score between query and document."""
        # query_emb: (1, m, d)
        # doc_emb: (1, n, d)

        # Compute similarity matrix: (m, n)
        sim_matrix = torch.matmul(
            query_emb.squeeze(0),
            doc_emb.squeeze(0).transpose(0, 1)
        )

        # MaxSim: max over document tokens for each query token
        max_sims = sim_matrix.max(dim=1).values  # (m,)

        # Sum all MaxSim scores
        return max_sims.sum().item()

    async def rerank(
        self,
        query: str,
        documents: list[Chunk],
        top_k: int = 10
    ) -> RerankResult:
        # Encode query
        query_emb = self.encode([query])

        # Encode all documents
        doc_texts = [d.content for d in documents]
        doc_embs = self.encode(doc_texts)

        # Compute MaxSim scores
        scores = []
        for i in range(len(documents)):
            score = self.maxsim(query_emb, doc_embs[i:i+1])
            scores.append((documents[i], score))

        # Sort by score descending
        scores.sort(key=lambda x: -x[1])

        return RerankResult(
            chunks=[s[0] for s in scores[:top_k]],
            scores=[s[1] for s in scores[:top_k]]
        )
```

### Complexity Analysis

| Operation | Complexity |
|-----------|------------|
| Query encoding | O(m · d) |
| Document encoding | O(n · d) |
| Similarity matrix | O(m · n · d) |
| MaxSim | O(m · n) |
| **Total per document** | O(m · n · d) |

For reranking 100 documents with m=32 query tokens, n=256 doc tokens, d=128:
- Similarity computations: 32 × 256 × 100 = 819,200

### Optimizations

1. **Quantization**: Reduce embedding precision (fp32 → int8)
2. **Pruning**: Remove low-impact query tokens
3. **Batching**: Process multiple documents in parallel
4. **Pre-computation**: Store document embeddings offline

---

## Cross-Encoders

Cross-encoders process query and document **together** through the model.

### Architecture

```
Input: [CLS] query [SEP] document [SEP]
              │
              ▼
         ┌─────────┐
         │  BERT   │
         └────┬────┘
              │
              ▼
         [CLS] embedding
              │
              ▼
         ┌─────────┐
         │ Linear  │
         └────┬────┘
              │
              ▼
         Score (0-1)
```

### Mathematical Formulation

$$\text{score}(q, d) = \sigma\left(W \cdot \text{BERT}([q; d])_\text{[CLS]} + b\right)$$

Where:
- $[q; d]$ = concatenated query and document
- $\text{BERT}(\cdot)_\text{[CLS]}$ = CLS token embedding
- $\sigma$ = sigmoid activation
- $W, b$ = learned parameters

### Comparison: ColBERT vs Cross-Encoder

| Aspect | ColBERT | Cross-Encoder |
|--------|---------|---------------|
| Encoding | Separate (query, doc) | Joint |
| Interaction | Late (after encoding) | Early (during encoding) |
| Quality | Very High | Highest |
| Speed | Fast (embeddings cached) | Slow (no caching) |
| Scalability | Good | Limited |
| Interpretability | High (token-level) | Low (black box) |

### When to Use What

```
Need interpretability?
├─ Yes → ColBERT
└─ No
   └─ Maximum quality required?
      ├─ Yes → Cross-Encoder
      └─ No → ColBERT (better speed)
```

---

## Comparison

### Performance Benchmarks

| Model | MS MARCO MRR@10 | Latency (100 docs) |
|-------|-----------------|-------------------|
| BM25 | 0.187 | 5ms |
| Bi-encoder | 0.334 | 15ms |
| ColBERT | 0.360 | 45ms |
| Cross-encoder | 0.389 | 500ms |

### Quality vs Speed Trade-off

```
Quality
   ▲
   │          ● Cross-encoder
   │      ● ColBERT
   │  ● Bi-encoder
   │● BM25
   └──────────────────────────▶ Speed
```

---

## Configuration

### Using ColBERT

```python
from agentic_rag.pipeline import PipelineBuilder
from agentic_rag.reranking import ColBERTReranker

pipeline = (
    PipelineBuilder()
    .with_retrieval("hybrid", top_k=100)
    .with_reranker(reranker="colbert", top_k=10)
    .build()
)
```

### Using Cross-Encoder

```python
from agentic_rag.reranking import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

pipeline = (
    PipelineBuilder()
    .with_retrieval("dense", top_k=50)
    .with_reranker(reranker)
    .build()
)
```

### Custom Reranker

```python
from agentic_rag.reranking import BaseReranker

class MyReranker(BaseReranker):
    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 10
    ) -> RerankResult:
        # Your reranking logic
        ...
```

---

## References

1. Khattab, O., & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR 2020. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)

2. Weaviate. (2024). "An Overview of Late Interaction Retrieval Models: ColBERT, ColPali, and ColQwen." [weaviate.io/blog/late-interaction-overview](https://weaviate.io/blog/late-interaction-overview)

3. Stanford ColBERT. GitHub. [github.com/stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT)

4. Qdrant. "Working with ColBERT." [qdrant.tech/documentation/fastembed/fastembed-colbert](https://qdrant.tech/documentation/fastembed/fastembed-colbert/)
