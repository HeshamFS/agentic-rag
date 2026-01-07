# AgenticRAG Documentation

> **Production-Ready Retrieval-Augmented Generation Pipeline**
>
> A comprehensive, state-of-the-art RAG system implementing the latest 2024-2025 research advances for academic and enterprise applications.

---

## Overview

AgenticRAG is a modular, extensible framework for building high-performance Retrieval-Augmented Generation systems. It combines cutting-edge algorithms from recent research with production-ready implementation patterns.

### Key Features

| **RAPTOR Chunking** | Hierarchical tree-organized retrieval | +20% accuracy on multi-hop QA |
| **ColBERT Reranking** | Late interaction with MaxSim scoring | +15-40% MRR improvement |
| **Contextual Retrieval** | Anthropic-style chunk context headers | -67% failed retrievals |
| **Late Chunking** | Context-aware embeddings (embed-then-chunk) | 95% context preservation |
| **GraphRAG** | Knowledge graph with community detection | Enhanced global query answering |
| **Hybrid Retrieval** | Dense + BM25 sparse fusion | Robust across query types |

---

## Documentation Structure

### Architecture

- **[Core Architecture](architecture/index.md)** - Pipeline design, protocols, and data models

### Algorithms

Deep-dive documentation for each algorithm with mathematical foundations:

- **[Embeddings](algorithms/embeddings.md)** - Qwen3, Late Chunking
- **[Chunking Strategies](algorithms/chunking.md)** - Semantic, Hierarchical, and RAPTOR
- **[Retrieval Methods](algorithms/retrieval.md)** - Dense, Sparse (BM25), Hybrid, HyDE
- **[Reranking](algorithms/reranking.md)** - ColBERT MaxSim, Cross-Encoders
- **[Context Compression](algorithms/compression.md)** - Extractive, LongLLMLingua
- **[Semantic Caching](algorithms/caching.md)** - Similarity-based query caching
- **[GraphRAG](algorithms/graphrag.md)** - Knowledge graphs and community detection
- **[Generation](algorithms/generation.md)** - LLM providers and prompting
- **[Evaluation Metrics](algorithms/evaluation.md)** - RAGAS framework and Self-RAG

### Guides

Practical guides for common use cases:

- **[Quick Start](guides/quickstart.md)** - Get running in 5 minutes
- **[Configuration & Deployment](guides/configuration.md)** - Environment, settings, and production deployment

### API Reference

- **[Complete API Reference](api/index.md)** - All classes, methods, and configuration options

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │  Ingest  │──▶│  Chunk   │──▶│  Embed   │──▶│  Index   │             │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘             │
│       │              │              │              │                     │
│       ▼              ▼              ▼              ▼                     │
│   PDF/Text      Semantic/       Qwen3/         Qdrant/                  │
│   Loader        RAPTOR          Late Chunk     Vector DB                │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │  Query   │──▶│ Retrieve │──▶│ Rerank   │──▶│ Compress │             │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘             │
│       │              │              │              │                     │
│       ▼              ▼              ▼              ▼                     │
│   HyDE           Hybrid         ColBERT       Extractive/               │
│   Expansion      Dense+BM25     MaxSim        LongLLMLingua             │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                             │
│  │  Cache   │──▶│ Generate │──▶│ Evaluate │                             │
│  └──────────┘   └──────────┘   └──────────┘                             │
│       │              │              │                                    │
│       ▼              ▼              ▼                                    │
│   Semantic       Gemini/         RAGAS                                  │
│   Cache          Claude/GPT      Metrics                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```python
from agentic_rag.pipeline import PipelineBuilder
from agentic_rag.embeddings import Qwen3Embedder

# Build a production-ready pipeline
pipeline = (
    PipelineBuilder()
    .with_embedder("default")
    .with_chunking("semantic", chunk_size=512)
    .with_retrieval("hybrid", top_k=10)
    .with_reranker("colbert")
    .with_compression("extractive", compression_ratio=0.5)
    .with_cache(backend="memory")
    .with_generator(provider="claude", model="claude-sonnet-4-5-20250929")
    .build()
)

# Ingest documents
await pipeline.ingest("papers/", collection="research")

# Query with full pipeline
response = await pipeline.query(
    question="What are the key innovations in transformer architecture?",
    collection="research"
)

print(response.response)
print(f"Sources: {len(response.sources)}")
```

---

## Research References

This implementation is based on the following research:

| Algorithm | Paper | Year |
|-----------|-------|------|
| RAPTOR | [Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) | ICLR 2024 |
| ColBERT | [Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832) | SIGIR 2020 |
| LongLLMLingua | [Accelerating and Enhancing LLMs in Long Context Scenarios](https://arxiv.org/abs/2310.06839) | ACL 2024 |
| GraphRAG | [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) | Microsoft 2024 |
| RAGAS | [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) | 2023 |
| HyDE | [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) | 2022 |

---

## Installation

```bash
# Clone repository
git clone https://github.com/heshamfs/agentic-rag.git
cd agentic-rag

# Install with uv (recommended)
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Requirements

- Python 3.12+
- CUDA 12.1+ (for GPU acceleration, optional)
- Qdrant Cloud or local instance (for vector storage)

---

## License

MIT License - See [LICENSE](../LICENSE) for details.
