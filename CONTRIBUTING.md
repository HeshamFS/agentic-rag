# Contributing to AgenticRAG

Thank you for your interest in contributing to the AgenticRAG! This project aims to provide a state-of-the-art, production-grade agentic RAG framework implementing the latest research from 2024-2025.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Coding Standards](#coding-standards)
4. [Documentation Standards](#documentation-standards)
5. [Testing & Benchmarking](#testing--benchmarking)
6. [Pull Request Process](#pull-request-process)

## Development Setup

We use `uv` for dependency management and virtual environments.

```bash
# Clone the repository
git clone https://github.com/heshamfs/agentic-rag.git
cd agentic-rag

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### Environment Configuration

Copy the `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

## Project Structure

```text
src/agentic_rag/
├── agents/          # Multi-agent orchestration (Router, Retriever, Evaluator, Generator)
├── chunking/        # Chunking strategies (Semantic, Late, RAPTOR, Contextual)
├── embeddings/      # Embedding model integrations (Qwen3-Embedding)
├── generation/      # LLM provider clients (Claude, OpenAI, Gemini, Local)
├── graph/           # GraphRAG implementation (Entities, Relationships, Communities)
├── ingestion/       # Document loading and processing
├── pipeline/        # Pipeline orchestration and builder
├── retrieval/       # Retrieval strategies (Hybrid, HyDE, Multi-Query)
├── reranking/       # Reranking models (ColBERT, Cross-Encoders)
├── vectordb/        # Vector database clients (Qdrant)
├── evaluation/      # RAGAS metrics and Self-RAG reflection
└── observability/   # OpenTelemetry and tracing
```

## Coding Standards

- **Type Hints**: All new code must use Python type hints.
- **Async/Await**: The pipeline is async-first. Use `async/await` for all I/O operations.
- **Pydantic**: Use Pydantic models for data structures and settings.
- **Linting**: We use `ruff` for linting and formatting.
  ```bash
  ruff check src/
  ruff format src/
  ```
- **Type Checking**: We use `mypy` for static type checking.
  ```bash
  mypy src/
  ```

## Documentation Standards

- **Docstrings**: All public classes and methods must have Google-style docstrings.
- **Type Information**: Include type information in docstrings if it helps clarity, although type hints are preferred.
- **README Updates**: If you add a new feature, update the `README.md` and relevant files in `docs/`.

## Testing & Benchmarking

### Running Tests
```bash
pytest
```

### Running Benchmarks
The benchmark system is critical for verifying performance gains.
```bash
# Run the core benchmark
python scripts/run_benchmark.py

# Run industry comparison (requires benchmark collection to be created first)
python scripts/run_comparison.py
```

## Pull Request Process

1. Create a new branch for your feature or bugfix.
2. Ensure all tests pass and linting is clean.
3. Update documentation if necessary.
4. Submit a PR with a clear description of the changes and their impact on performance (if applicable).
5. Link any related issues.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
