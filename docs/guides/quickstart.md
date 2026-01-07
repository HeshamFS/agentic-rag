# Quick Start Guide

> **Get Up and Running in 5 Minutes**
>
> This guide covers installation, basic configuration, and your first RAG pipeline.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Your First Pipeline](#your-first-pipeline)
4. [Basic Operations](#basic-operations)
5. [Examples](#examples)

---

## Installation

### Requirements

- Python 3.12+
- CUDA 12.1+ (for GPU acceleration, optional)
- 16GB+ RAM recommended
- Qdrant Cloud or local instance

### Install from Source

```bash
# Clone repository
git clone https://github.com/heshamfs/agentic-rag.git
cd agentic-rag

# Create virtual environment with uv (recommended)
uv venv --python 3.12
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with GPU support
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ".[dev]"

# Or with pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# LLM Provider
RAG_ANTHROPIC_API_KEY=sk-ant-...
# RAG_OPENAI_API_KEY=sk-...
# RAG_GOOGLE_API_KEY=...

# Default LLM settings
RAG_LLM_PROVIDER=claude
RAG_LLM_MODEL=claude-sonnet-4-5-20250929
RAG_LLM_TEMPERATURE=0.3

# Embedding settings
RAG_EMBEDDING_MODEL=Alibaba-NLP/gte-Qwen2-1.5B-instruct
RAG_EMBEDDING_DEVICE=cuda

# Vector database
RAG_QDRANT_URL=http://localhost:6333
RAG_QDRANT_API_KEY=  # Optional

# Redis caching (optional)
RAG_REDIS_URL=redis://localhost:6379/0
```

### Verify Configuration

```python
from agentic_rag.config import get_settings

settings = get_settings()
print(f"LLM Provider: {settings.llm_provider}")
print(f"Embedding Model: {settings.embedding_model}")
print(f"Qdrant URL: {settings.qdrant_url}")
```

---

## Your First Pipeline

### Minimal Example

```python
import asyncio
from agentic_rag.pipeline import PipelineBuilder
from agentic_rag.core.models import Document

async def main():
    # Build pipeline with defaults
    pipeline = PipelineBuilder().build()

    # Create documents
    documents = [
        Document(content="The Transformer architecture uses self-attention."),
        Document(content="BERT is a bidirectional transformer model."),
        Document(content="GPT uses causal (left-to-right) attention."),
    ]

    # Ingest into collection
    await pipeline.ingest(documents, collection="demo")

    # Query
    result = await pipeline.query(
        query="What is the Transformer architecture?",
        collection="demo",
    )

    print(f"Response: {result.response}")
    print(f"Sources: {len(result.sources)}")

asyncio.run(main())
```

### Full-Featured Pipeline

```python
from agentic_rag.pipeline import PipelineBuilder

# Build with all features
pipeline = (
    PipelineBuilder()
    # Embedding
    .with_embedder(embedder="default")

    # Chunking
    .with_chunking(
        strategy="semantic",
        chunk_size=512,
        chunk_overlap=50,
    )

    # Retrieval
    .with_retrieval(
        strategy="hybrid",
        top_k=20,
        use_hyde=True,
    )

    # Reranking
    .with_reranker(reranker="colbert", top_k=10)

    # Compression
    .with_compression(
        method="extractive",
        compression_ratio=0.5,
    )

    # Caching
    .with_cache(
        backend="memory",
        similarity_threshold=0.95,
        ttl_seconds=3600,
    )

    # Generation
    .with_generator(
        provider="claude",
        model="claude-sonnet-4-5-20250929",
        temperature=0.3,
    )

    # Evaluation
    .with_evaluation(enable_ragas=True)

    .build()
)
```

---

## Basic Operations

### Ingesting Documents

```python
from agentic_rag.core.models import Document

# From text
documents = [
    Document(
        content="Your document text here...",
        metadata={"source": "manual", "category": "tech"},
    ),
]

# From files
import pathlib

docs = []
for file in pathlib.Path("./data").glob("*.txt"):
    docs.append(Document(
        content=file.read_text(),
        source=str(file),
        metadata={"filename": file.name},
    ))

# Ingest
result = await pipeline.ingest(
    documents=docs,
    collection="my_docs",
)

print(f"Ingested: {result.num_documents} documents")
print(f"Chunks: {result.num_chunks}")
```

### Querying

```python
# Basic query
result = await pipeline.query(
    query="What is machine learning?",
    collection="my_docs",
)

print(result.response)

# With options
result = await pipeline.query(
    query="Explain transformers in detail",
    collection="my_docs",
    top_k=15,           # Override retrieval count
    temperature=0.5,    # Override generation temp
)

# Access sources
for chunk in result.sources:
    print(f"- {chunk.content[:100]}...")
```

### Retrieval Only

```python
# Just retrieve, no generation
retrieval = await pipeline.retrieve(
    query="attention mechanism",
    collection="my_docs",
    top_k=10,
)

for chunk, score in zip(retrieval.chunks, retrieval.scores):
    print(f"[{score:.3f}] {chunk.content[:80]}...")
```

---

## Examples

### Research Paper QA

```python
import asyncio
from pathlib import Path
from agentic_rag.pipeline import PipelineBuilder
from agentic_rag.core.models import Document

async def research_qa():
    # Pipeline optimized for research papers
    pipeline = (
        PipelineBuilder()
        .with_chunking(strategy="semantic", chunk_size=800)
        .with_retrieval("hybrid", top_k=20, use_hyde=True)
        .with_reranker(reranker="colbert", top_k=10)
        .with_generator(provider="claude", temperature=0.2)
        .build()
    )

    # Load papers
    papers = []
    for pdf in Path("./papers").glob("*.txt"):  # Pre-extracted text
        papers.append(Document(
            content=pdf.read_text(),
            source=pdf.name,
            metadata={"type": "research_paper"},
        ))

    await pipeline.ingest(papers, collection="research")

    # Query
    result = await pipeline.query(
        "What are the main contributions of the attention paper?",
        collection="research",
    )

    print(result.response)

asyncio.run(research_qa())
```

### Customer Support Bot

```python
async def support_bot():
    # Optimized for FAQ-style responses
    pipeline = (
        PipelineBuilder()
        .with_chunking(strategy="hierarchical")
        .with_retrieval("dense", top_k=5)
        .with_cache(backend="redis", ttl_seconds=86400)
        .with_generator(
            provider="gemini",
            model="gemini-2.5-flash",
            temperature=0.1,
        )
        .build()
    )

    # Ingest FAQ
    faqs = [
        Document(content="Q: How do I reset my password?\nA: Go to Settings > Security > Reset Password."),
        Document(content="Q: What are your business hours?\nA: We're open Monday-Friday, 9 AM - 5 PM EST."),
        # ... more FAQs
    ]

    await pipeline.ingest(faqs, collection="support")

    # Handle query
    result = await pipeline.query(
        "How can I change my password?",
        collection="support",
    )

    print(result.response)
```

### Multi-Hop Reasoning with RAPTOR

```python
async def multi_hop_qa():
    # RAPTOR for complex reasoning
    pipeline = (
        PipelineBuilder()
        .with_chunking(
            strategy="raptor",
            raptor_levels=3,
            raptor_clustering="gmm",
        )
        .with_retrieval("hybrid", top_k=15)
        .with_reranker("colbert", top_k=8)
        .with_generator(
            provider="claude",
            model="claude-sonnet-4-20250514",
            temperature=0.3,
        )
        .build()
    )

    # Load knowledge base
    docs = [Document(content=text) for text in load_knowledge_base()]
    await pipeline.ingest(docs, collection="knowledge")

    # Complex query
    result = await pipeline.query(
        "What are the main themes across all documents and how do they relate?",
        collection="knowledge",
    )

    print(result.response)
```

### GraphRAG for Global Queries

```python
async def global_analysis():
    # GraphRAG for thematic queries
    pipeline = (
        PipelineBuilder()
        .with_graphrag(
            enabled=True,
            entity_types=["Person", "Organization", "Concept"],
            community_detection="leiden",
        )
        .with_generator(provider="gemini")
        .build()
    )

    # Ingest with entity extraction
    await pipeline.ingest(documents, collection="corpus")

    # Global query
    result = await pipeline.query(
        "What are the main themes and how do they connect?",
        collection="corpus",
        use_graph=True,
    )

    print(result.response)
```

---

## Next Steps

- **[Architecture](../architecture/index.md)**: Understand the system design
- **[Algorithms](../algorithms/)**: Deep dive into each algorithm
- **[API Reference](../api/index.md)**: Complete API documentation
- **[Configuration](./configuration.md)**: Advanced configuration options

