# Embedding Models

> **Semantic Vector Representations for RAG**
>
> This document covers the Qwen3 embedding family, late chunking technique, and best practices for embedding in RAG systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Qwen3 Embeddings](#qwen3-embeddings)
3. [Late Chunking](#late-chunking)
4. [Embedding Best Practices](#embedding-best-practices)
5. [Configuration](#configuration)

---

## Overview

Embeddings transform text into dense vector representations that capture semantic meaning. In RAG systems, embeddings enable:

- **Semantic search**: Find relevant documents by meaning, not just keywords
- **Similarity matching**: Compare query to document embeddings
- **Clustering**: Group related content together

### Embedding Dimension Trade-offs

| Dimension | Storage | Speed | Quality |
|-----------|---------|-------|---------|
| 256 | Low | Fast | Good |
| 512 | Medium | Medium | Better |
| 1024+ | High | Slower | Best |

### Mathematical Foundation

Given text $t$, an embedding model $f_\theta$ maps it to a vector:

$$E = f_\theta(t) \in \mathbb{R}^d$$

Where $d$ is the embedding dimension.

**Similarity** between vectors is typically computed using cosine similarity:

$$\text{sim}(E_1, E_2) = \frac{E_1 \cdot E_2}{\|E_1\| \|E_2\|} = \frac{\sum_{i=1}^{d} E_{1,i} \cdot E_{2,i}}{\sqrt{\sum_{i=1}^{d} E_{1,i}^2} \cdot \sqrt{\sum_{i=1}^{d} E_{2,i}^2}}$$

---

## Qwen3 Embeddings

**Qwen3-Embedding** is the state-of-the-art open-source embedding model family from Alibaba.

> **Reference**: Alibaba Cloud. (2025). "Mastering Text Embedding and Reranker with Qwen3." [alibabacloud.com/blog](https://www.alibabacloud.com/blog/mastering-text-embedding-and-reranker-with-qwen3_602308)

### Model Variants

| Model | Size | MTEB Score | Use Case |
|-------|------|------------|----------|
| **gte-Qwen2-1.5B-instruct** | 1.5B | 70.5 | Highest quality, long context |
| Qwen3-Embedding-0.6B | 0.6B | 68.5 | Fast inference, edge |
| Qwen3-Embedding-4B | 4B | 69.8 | Balanced |
| Qwen3-Embedding-8B | 8B | 70.58 | Largest state-of-the-art |

### Key Features

1. **Multilingual**: 100+ languages in unified semantic space
2. **Long Context**: 32K token context window
3. **Matryoshka Learning**: Flexible dimensions (32-1024)
4. **Instruction Mode**: Task-specific embeddings via prompts
5. **Apache 2.0 License**: Open source for commercial use

### Architecture

```
Input Text: "What is machine learning?"
         │
         ▼
    ┌─────────────────────────────────────┐
    │       Qwen3 Transformer             │
    │  (Causal attention, 32K context)    │
    └───────────────────┬─────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  [EOS] Token Hidden   │
            │       State           │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   L2 Normalization    │
            └───────────┬───────────┘
                        │
                        ▼
              Embedding ∈ ℝ^d
```

### Instruction Mode

Qwen3 supports instruction-conditioned embeddings:

```python
# Query embedding with instruction
query_prompt = "Instruct: Retrieve documents about machine learning\nQuery: {text}"

# Document embedding (no instruction needed)
doc_prompt = "{text}"
```

### Implementation

```python
class Qwen3Embedder(BaseEmbedder):
    """Qwen3-Embedding model implementation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 8192,
        normalize_embeddings: bool = True,
        use_cache: bool = True,
    ):
        self._model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        self._model.max_seq_length = max_length

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
        )
        return embeddings.tolist()
```

### Performance Benchmarks

From MTEB Multilingual Leaderboard:

| Model | Score | Languages | Context |
|-------|-------|-----------|---------|
| **Qwen3-Embedding-8B** | **70.58** | 100+ | 32K |
| Gemini Embedding | 69.2 | 100+ | 2K |
| BGE-M3 | 68.5 | 100+ | 8K |
| E5-Large-v2 | 66.8 | 100+ | 512 |

---

## Late Chunking

**Late Chunking** is a technique that preserves document context when embedding chunks, introduced by Jina AI in 2024.

> **Reference**: Jina AI. (2024). "Late Chunking in Long-Context Embedding Models." [jina.ai/news](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
>
> **Paper**: [arXiv:2409.04701](https://arxiv.org/abs/2409.04701)

### The Problem

Traditional chunking destroys context:

```
Document: "Berlin is the capital of Germany. It has a population of 3.5 million."

Traditional Chunking:
├── Chunk 1: "Berlin is the capital of Germany."
│   └── Embedding: [0.12, 0.45, ...]  ← Knows Berlin
│
└── Chunk 2: "It has a population of 3.5 million."
    └── Embedding: [0.08, 0.23, ...]  ← Lost reference to "It"

Query: "What is the population of Berlin?"
Result: ❌ Chunk 2 doesn't connect to Berlin
```

### The Solution

Late chunking embeds the **full document first**, then extracts chunk embeddings:

```
Document: "Berlin is the capital of Germany. It has a population of 3.5 million."
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Encode Full Document                            │
    │  Token embeddings: [E_Berlin, E_is, ..., E_It, E_has, ...]  │
    └─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
    ┌──────────────────┐            ┌──────────────────┐
    │ Extract tokens   │            │ Extract tokens   │
    │ for Chunk 1      │            │ for Chunk 2      │
    │ [E_Berlin...E_.]  │            │ [E_It...E_.]     │
    └────────┬─────────┘            └────────┬─────────┘
             │                               │
             ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │   Mean Pool      │            │   Mean Pool      │
    │   Chunk 1        │            │   Chunk 2        │
    │   Embedding      │            │   Embedding      │
    │                  │            │  (Knows "It" =   │
    │                  │            │   Berlin!)       │
    └──────────────────┘            └──────────────────┘
```

### Mathematical Formulation

Given document $D$ with tokens $T = [t_1, t_2, ..., t_n]$:

**Step 1**: Encode full document through transformer:

$$H = \text{Transformer}(T) \in \mathbb{R}^{n \times d}$$

Where $H_i$ is the contextualized embedding for token $i$.

**Step 2**: For chunk spanning positions $[i, j]$, mean pool tokens:

$$E_{\text{chunk}} = \frac{1}{j-i+1} \sum_{k=i}^{j} H_k$$

### Key Insight

Each token embedding $H_k$ is **contextualized** by the full document through self-attention:

$$H_k = \text{Attention}(Q_k, K_{1:n}, V_{1:n})$$

So chunk embeddings retain information about the entire document.

### Implementation

```python
class LateChunkingEmbedder(BaseEmbedder):
    """Late chunking embedder."""

    async def embed_document_with_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> list[list[float]]:
        """Embed chunks using late chunking."""
        model, tokenizer = self._load_model()

        # Step 1: Tokenize full document with offset mapping
        encoding = tokenizer(
            document.content,
            return_offsets_mapping=True,
            max_length=self._max_length,
        )
        offset_mapping = encoding.pop("offset_mapping")

        # Step 2: Get token embeddings for full document
        with torch.no_grad():
            outputs = model(**encoding)
            token_embeddings = outputs.last_hidden_state[0]

        # Step 3: Extract embeddings for each chunk
        chunk_embeddings = []
        for chunk in chunks:
            # Find chunk position in document
            chunk_start = document.content.find(chunk.content)
            chunk_end = chunk_start + len(chunk.content)

            # Find corresponding token indices
            token_start, token_end = self._find_token_range(
                offset_mapping, chunk_start, chunk_end
            )

            # Mean pool chunk tokens
            chunk_tokens = token_embeddings[token_start:token_end]
            chunk_emb = chunk_tokens.mean(dim=0)
            chunk_embeddings.append(chunk_emb)

        return chunk_embeddings
```

### Benefits

| Aspect | Traditional | Late Chunking |
|--------|-------------|---------------|
| Context preservation | ❌ Lost | ✅ Preserved |
| Anaphora resolution | ❌ Poor | ✅ Good |
| Semantic coherence | ❌ Fragmented | ✅ Coherent |
| Chunk boundary sensitivity | ❌ High | ✅ Low |

### Requirements

- **Long-context model**: 8K+ token context (Jina v2, Qwen3)
- **Offset mapping**: Tokenizer must provide character positions
- **Memory**: Must fit entire document in memory

### When to Use Late Chunking

| Scenario | Recommendation |
|----------|---------------|
| Documents with many pronouns/references | ✅ Use late chunking |
| Short, self-contained chunks | ❌ Traditional is fine |
| Very long documents (>32K tokens) | ❌ May need sliding window |
| Speed-critical applications | ❌ Traditional is faster |

---

## Embedding Best Practices

### 1. Normalization

Always L2-normalize embeddings for cosine similarity:

$$E_{\text{norm}} = \frac{E}{\|E\|_2} = \frac{E}{\sqrt{\sum_{i=1}^{d} E_i^2}}$$

After normalization, cosine similarity equals dot product:

$$\cos(E_1, E_2) = E_1 \cdot E_2$$

### 2. Batching

Process texts in batches for GPU efficiency:

```python
# Good: Batch processing
embeddings = model.encode(texts, batch_size=32)

# Bad: One at a time
embeddings = [model.encode(t) for t in texts]
```

### 3. Caching

Cache embeddings to avoid recomputation:

```python
class EmbeddingCache:
    """LRU cache for embeddings."""

    def __init__(self, max_size: int = 10000):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size

    def get(self, text: str) -> list[float] | None:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        key = hashlib.md5(text.encode()).hexdigest()
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = embedding
```

### 4. Instruction Tuning

Use task-specific prompts for better results:

| Task | Query Prompt | Document Prompt |
|------|-------------|-----------------|
| Retrieval | "Retrieve documents about: {query}" | "{document}" |
| QA | "Answer this question: {query}" | "{document}" |
| Classification | "Classify this text: {query}" | "{document}" |

---

## Configuration

### Basic Usage

```python
from agentic_rag.embeddings import Qwen3Embedder, create_embedder

# Use factory with preset
embedder = create_embedder("default")  # 0.6B model

# Or configure directly
embedder = Qwen3Embedder(
    model_name="Qwen/Qwen3-Embedding-8B",
    device="cuda",
    batch_size=16,
    max_length=8192,
    normalize_embeddings=True,
    use_cache=True,
)

# Embed text
embedding = await embedder.embed_text("What is machine learning?")

# Batch embed
embeddings = await embedder.embed_batch([
    "Text 1",
    "Text 2",
    "Text 3",
])
```

### Late Chunking Usage

```python
from agentic_rag.embeddings import LateChunkingEmbedder
from agentic_rag.core.models import Document, Chunk

embedder = LateChunkingEmbedder(
    model="Qwen/Qwen3-Embedding-0.6B",
    device="cuda",
    max_length=8192,
)

# Create document and chunks
document = Document(content="Berlin is the capital...")
chunks = [
    Chunk(content="Berlin is the capital of Germany.", document_id=document.id),
    Chunk(content="It has a population of 3.5 million.", document_id=document.id),
]

# Get context-aware embeddings
embeddings = await embedder.embed_document_with_chunks(document, chunks)
```

### Pipeline Integration

```python
from agentic_rag.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .with_embedder(
        model_variant="large",  # 8B model
        batch_size=16,
        use_cache=True,
    )
    .with_retrieval("hybrid")
    .build()
)
```

### Environment Variables

```bash
# Embedding model
EMBEDDING_MODEL=Alibaba-NLP/gte-Qwen2-1.5B-instruct
EMBEDDING_DEVICE=cuda  # or cpu, mps
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_LENGTH=8192
```

---

## References

1. Alibaba Cloud. (2025). "Mastering Text Embedding and Reranker with Qwen3." [alibabacloud.com/blog](https://www.alibabacloud.com/blog/mastering-text-embedding-and-reranker-with-qwen3_602308)

2. Jina AI. (2024). "Late Chunking in Long-Context Embedding Models." [jina.ai/news](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)

3. Günther, M., et al. (2024). "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models." [arXiv:2409.04701](https://arxiv.org/abs/2409.04701)

4. Milvus. (2025). "Hands-on RAG with Qwen3 Embedding and Reranking Models." [milvus.io/blog](https://milvus.io/blog/hands-on-rag-with-qwen3-embedding-and-reranking-models-using-milvus.md)

5. Weaviate. (2024). "Late Chunking: Balancing Precision and Cost in Long Context Retrieval." [weaviate.io/blog](https://weaviate.io/blog/late-chunking)

