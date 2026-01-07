# Configuration & Deployment

> **Production Configuration and Deployment Guide**
>
> This document covers environment variables, advanced configuration, and deployment patterns.

---

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Settings Reference](#settings-reference)
3. [Provider Configuration](#provider-configuration)
4. [Performance Tuning](#performance-tuning)
5. [Deployment Patterns](#deployment-patterns)

---

## Environment Variables

### Core Settings

```bash
# =============================================================================
# LLM Provider Settings
# =============================================================================

# Claude (Anthropic)
ANTHROPIC_API_KEY=sk-ant-api03-...

# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini
GOOGLE_API_KEY=AIza...

# Local (Ollama)
OLLAMA_BASE_URL=http://localhost:11434

# Default provider
LLM_PROVIDER=claude           # claude, openai, gemini, local
LLM_MODEL=claude-sonnet-4-5-20250929    # Model identifier
LLM_TEMPERATURE=0.3           # 0.0-1.0
LLM_MAX_TOKENS=4096           # Maximum output tokens

# =============================================================================
# Embedding Settings
# =============================================================================

EMBEDDING_MODEL=Alibaba-NLP/gte-Qwen2-1.5B-instruct
EMBEDDING_DEVICE=cuda         # cuda, cpu, mps
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_LENGTH=8192

# =============================================================================
# Vector Database Settings
# =============================================================================

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=               # Optional for local
QDRANT_PREFER_GRPC=true       # Use gRPC for better performance

# HNSW index tuning
HNSW_M=16                     # Connections per node
HNSW_EF_CONSTRUCT=100         # Build-time search width

# =============================================================================
# Caching Settings
# =============================================================================

CACHE_BACKEND=memory          # memory, disk, redis
CACHE_SIMILARITY_THRESHOLD=0.95
CACHE_TTL_SECONDS=3600

# Redis settings
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=               # Optional
REDIS_MAX_CONNECTIONS=20

# =============================================================================
# Chunking Settings
# =============================================================================

CHUNK_STRATEGY=semantic       # semantic, hierarchical, raptor
CHUNK_SIZE=512
CHUNK_OVERLAP=50
ADD_CONTEXT_HEADERS=true

# RAPTOR settings
RAPTOR_MAX_LEVELS=3
RAPTOR_CLUSTERING=gmm         # gmm, kmeans
RAPTOR_SUMMARY_TOKENS=200

# =============================================================================
# Retrieval Settings
# =============================================================================

RETRIEVAL_STRATEGY=hybrid     # dense, sparse, hybrid
RETRIEVAL_TOP_K=10
USE_HYDE=true
USE_RRF=true
SPARSE_WEIGHT=0.3

# Reranking
RERANKER_MODEL=colbert
RERANKER_TOP_K=5

# =============================================================================
# Compression Settings
# =============================================================================

ENABLE_COMPRESSION=false
COMPRESSION_TYPE=extractive   # extractive, longllmlingua
COMPRESSION_RATIO=0.5

# =============================================================================
# Evaluation Settings
# =============================================================================

ENABLE_RAGAS=true
ENABLE_SELF_RAG=false
REGENERATE_THRESHOLD=0.5

# =============================================================================
# Agentic Settings
# =============================================================================

ENABLE_REFLECTION=true
ENABLE_PLANNING=true
MAX_ITERATIONS=3
CONFIDENCE_THRESHOLD=0.7
WEB_FALLBACK=false
```

---

## Settings Reference

### Settings Class

```python
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr

class Settings(BaseSettings):
    """Application configuration."""

    # LLM Provider
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    llm_provider: str = "claude"
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # Embeddings
    embedding_model: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    embedding_max_length: int = 8192

    # Vector Database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_prefer_grpc: bool = True

    # Caching
    cache_backend: str = "memory"
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 3600
    redis_url: str = "redis://localhost:6379/0"
    redis_password: SecretStr | None = None

    # Chunking
    chunk_strategy: str = "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 50
    add_context_headers: bool = True

    # Retrieval
    retrieval_strategy: str = "hybrid"
    retrieval_top_k: int = 10
    use_hyde: bool = True
    use_rrf: bool = True
    sparse_weight: float = 0.3

    # Reranking
    reranker_model: str = "jina"
    reranker_top_k: int = 5

    # Compression
    enable_compression: bool = False
    compression_type: str = "extractive"
    compression_ratio: float = 0.5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### Loading Settings

```python
from agentic_rag.config import get_settings

# Load from environment
settings = get_settings()

# Override for testing
from agentic_rag.config import Settings
test_settings = Settings(
    llm_provider="local",
    cache_backend="memory",
)
```

---

## Provider Configuration

### Claude (Anthropic)

```python
# Required
ANTHROPIC_API_KEY=sk-ant-api03-...

# Optional
LLM_PROVIDER=claude
LLM_MODEL=claude-sonnet-4-20250514  # or claude-opus-4-20250514
```

**Available Models**:
- `claude-sonnet-4-20250514` - Balanced, 200K context
- `claude-opus-4-20250514` - Highest quality, 200K context

### OpenAI

```python
# Required
OPENAI_API_KEY=sk-...

# Optional
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
```

**Available Models**:
- `gpt-4o` - Latest GPT-4, 128K context
- `gpt-4o-mini` - Faster, cheaper
- `o1-preview` - Reasoning model

### Google Gemini

```python
# Required
GOOGLE_API_KEY=AIza...

# Optional
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
```

**Available Models**:
- `gemini-2.5-pro` - Highest quality, 1M context
- `gemini-2.5-flash` - Fast, 1M context
- `gemini-2.0-flash` - Balanced

### Local (Ollama)

```python
# Required
OLLAMA_BASE_URL=http://localhost:11434

# Optional
LLM_PROVIDER=local
LLM_MODEL=qwen2.5:7b
```

**Setup Ollama**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull qwen2.5:7b

# Verify
ollama list
```

---

## Performance Tuning

### Embedding Optimization

```bash
# Use GPU
EMBEDDING_DEVICE=cuda

# Increase batch size (if GPU memory allows)
EMBEDDING_BATCH_SIZE=64

# Enable caching
CACHE_BACKEND=redis
```

### Retrieval Optimization

```bash
# Tune HNSW for speed vs recall
HNSW_M=16           # Higher = better recall, slower
HNSW_EF_CONSTRUCT=100

# Use hybrid for best quality
RETRIEVAL_STRATEGY=hybrid
USE_RRF=true

# Enable HyDE for complex queries
USE_HYDE=true
```

### Generation Optimization

```bash
# Use fastest model
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash

# Lower temperature for factual responses
LLM_TEMPERATURE=0.1

# Reduce max tokens if possible
LLM_MAX_TOKENS=2048
```

### Memory Optimization

```bash
# Use smaller embedding model
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# Reduce batch size
EMBEDDING_BATCH_SIZE=16

# Use disk cache for large deployments
CACHE_BACKEND=disk
```

### Latency Optimization

| Component | Optimization |
|-----------|-------------|
| Embedding | Use GPU, increase batch size |
| Retrieval | Tune HNSW, use hybrid |
| Reranking | Use lighter model, reduce top_k |
| Compression | Disable if latency critical |
| Caching | Use Redis for distributed |
| Generation | Use flash models |

---

## Deployment Patterns

### Single-Server Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  agentic-rag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - qdrant
      - redis

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  qdrant_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentic-rag
  template:
    metadata:
      labels:
        app: agentic-rag
    spec:
      containers:
      - name: agentic-rag
        image: agentic-rag:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: google-api-key
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
```

### Production Checklist

- [ ] **API Keys**: Store in secrets manager
- [ ] **Logging**: Configure structured logging
- [ ] **Monitoring**: Set up metrics collection
- [ ] **Rate Limiting**: Implement per-user limits
- [ ] **Error Handling**: Graceful degradation
- [ ] **Caching**: Enable Redis for multi-node
- [ ] **Backups**: Regular Qdrant snapshots
- [ ] **Health Checks**: Liveness and readiness probes

### Scaling Considerations

| Component | Scaling Strategy |
|-----------|-----------------|
| API Server | Horizontal (add replicas) |
| Qdrant | Sharding, replicas |
| Redis | Cluster mode |
| Embeddings | GPU pool, batching |
| LLM | API concurrency limits |

---

## Security Best Practices

### API Key Management

```python
# Never hardcode keys
# Use environment variables or secrets manager

from pydantic import SecretStr

class Settings(BaseSettings):
    anthropic_api_key: SecretStr | None = None

    def validate_provider_config(self, provider: str) -> bool:
        if provider == "claude":
            return self.anthropic_api_key is not None
        # ...
```

### Input Validation

```python
# Validate query length
MAX_QUERY_LENGTH = 10000

if len(query) > MAX_QUERY_LENGTH:
    raise ValueError("Query too long")

# Sanitize collection names
import re
if not re.match(r'^[a-zA-Z0-9_-]+$', collection):
    raise ValueError("Invalid collection name")
```

### Rate Limiting

```python
from fastapi import FastAPI, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, query: str):
    # ...
```

---

## Monitoring

### Metrics to Track

| Metric | Type | Description |
|--------|------|-------------|
| `rag_query_latency_seconds` | Histogram | End-to-end query time |
| `rag_retrieval_latency_seconds` | Histogram | Retrieval time |
| `rag_generation_latency_seconds` | Histogram | LLM generation time |
| `rag_cache_hit_ratio` | Gauge | Cache hit percentage |
| `rag_tokens_used_total` | Counter | Total LLM tokens |
| `rag_errors_total` | Counter | Error count by type |

### Prometheus Integration

```python
from prometheus_client import Histogram, Counter, Gauge

QUERY_LATENCY = Histogram(
    'rag_query_latency_seconds',
    'Query latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

CACHE_HIT_RATIO = Gauge(
    'rag_cache_hit_ratio',
    'Cache hit ratio'
)

@QUERY_LATENCY.time()
async def query(query: str, collection: str):
    # ...
```

---

## Next Steps

- **[Quick Start](./quickstart.md)**: Get started quickly
- **[API Reference](../api/index.md)**: Complete API documentation
- **[Algorithms](../algorithms/)**: Deep dive into algorithms

