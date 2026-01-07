# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of AgenticRAG
- Multi-agent orchestration system (Router, Retriever, Evaluator, Generator)
- Hybrid retrieval with Dense + BM25 and RRF fusion
- ColBERT late-interaction reranking with Jina-ColBERT-v2
- Late chunking for context-preserving embeddings
- Semantic chunking with embedding-based topic segmentation
- HyDE (Hypothetical Document Embeddings) for query enhancement
- Contextual retrieval with chunk context headers
- GraphRAG knowledge graph extraction and retrieval
- Self-RAG with reflection tokens (ISREL, ISSUP, ISUSE)
- CRAG (Corrective RAG) with confidence-based fallbacks
- Multi-provider LLM support:
  - Claude (claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022)
  - OpenAI (gpt-4o, gpt-4o-mini)
  - Gemini (gemini-2.0-flash, gemini-1.5-pro, gemini-3-flash-preview)
  - Local via Ollama (qwen2.5, llama3.1)
- Qwen3-Embedding-0.6B for high-quality embeddings (1024 dimensions)
- Qdrant Cloud integration for vector storage
- RAGAS evaluation metrics (Context Precision, Recall, Faithfulness, Answer Relevancy)
- OpenTelemetry observability and tracing
- FastAPI REST API
- Typer CLI application
- React frontend with TypeScript and Tailwind CSS
- Comprehensive documentation (algorithms, architecture, API reference)
- Benchmark system with industry comparison
- Example scripts for basic usage, agentic pipelines, and multi-provider setup

### Benchmarks
- Achieved 0.958 MRR (outperforming ColBERT v2, Cohere Rerank, OpenAI RAG)
- 100% Hit Rate @5
- P95 Latency: 206ms with ColBERT reranking
- Embedding throughput: 1.7 texts/sec with Qwen3-Embedding-0.6B

## [0.1.0] - 2025-01-07

### Added
- Initial public release
- Core RAG pipeline with all major features
- Complete test suite
- Documentation and examples

[Unreleased]: https://github.com/heshamfs/agentic-rag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/heshamfs/agentic-rag/releases/tag/v0.1.0
