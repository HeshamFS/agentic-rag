# Semantic Caching

> **Intelligent Query Deduplication for RAG Systems**
>
> This document covers semantic caching mechanisms that reduce latency and costs by caching similar queries.

---

## Table of Contents

1. [Overview](#overview)
2. [How Semantic Caching Works](#how-semantic-caching-works)
3. [Cache Backends](#cache-backends)
4. [Similarity Search](#similarity-search)
5. [Cache Management](#cache-management)
6. [Configuration](#configuration)

---

## Overview

Semantic caching stores query results and returns cached responses for **semantically similar** queries, not just exact matches.

### Traditional vs Semantic Caching

| Aspect | Traditional Cache | Semantic Cache |
|--------|------------------|----------------|
| Key matching | Exact string match | Embedding similarity |
| "What is ML?" vs "What's machine learning?" | Cache miss | Cache hit |
| Storage | Key-value | Embeddings + values |
| Lookup | O(1) hash | O(log n) or O(n) similarity |

### Benefits

| Benefit | Impact |
|---------|--------|
| **Latency reduction** | 10-100x faster on cache hits |
| **Cost savings** | Avoid LLM calls for similar queries |
| **Consistency** | Same answer for similar questions |
| **Scalability** | Handle query spikes efficiently |

### Cache in the Pipeline

```
Query
  │
  ▼
┌─────────────────┐
│  Semantic Cache │──── Hit? ────▶ Return cached response
└────────┬────────┘
         │ Miss
         ▼
┌─────────────────┐
│   RAG Pipeline  │
│  (Retrieve →    │
│   Rerank →      │
│   Generate)     │
└────────┬────────┘
         │
         ▼
   Store in cache
         │
         ▼
  Return response
```

---

## How Semantic Caching Works

### Step 1: Query Embedding

Convert the incoming query to a vector:

$$E_q = \text{embed}(q) \in \mathbb{R}^d$$

### Step 2: Similarity Search

Find cached queries with similar embeddings:

$$\text{matches} = \{(q_i, r_i) : \text{sim}(E_q, E_{q_i}) > \tau\}$$

Where:
- $\tau$ = similarity threshold (e.g., 0.95)
- $r_i$ = cached response for query $q_i$

### Step 3: Cache Decision

```
If matches exist:
    Return highest-similarity cached response
Else:
    Execute full RAG pipeline
    Store (query, embedding, response) in cache
```

### Similarity Threshold Selection

| Threshold | Behavior |
|-----------|----------|
| 0.99 | Very strict, almost exact match |
| 0.95 | Standard, good balance |
| 0.90 | Lenient, more cache hits |
| 0.85 | Very lenient, risk of wrong answers |

**Recommended**: Start with 0.95, adjust based on error analysis.

---

## Cache Backends

### In-Memory Cache

Fastest option, suitable for single-server deployments.

```python
from agentic_rag.caching import SemanticCache

cache = SemanticCache(
    embedder=embedder,
    similarity_threshold=0.95,
    ttl_seconds=3600,  # 1 hour expiry
    max_entries=10000
)
```

**Architecture**:
```
┌─────────────────────────────────────┐
│            In-Memory Cache          │
├─────────────────────────────────────┤
│  embeddings: numpy array (N × d)    │
│  queries: list[str]                 │
│  responses: list[str]               │
│  metadata: list[dict]               │
│  timestamps: list[float]            │
└─────────────────────────────────────┘
```

**Pros**: Fastest, no external dependencies
**Cons**: Lost on restart, not shared across instances

### Disk-Based Cache

Persistent cache using disk storage.

```python
from agentic_rag.caching import DiskSemanticCache

cache = DiskSemanticCache(
    embedder=embedder,
    cache_dir="/path/to/cache",
    similarity_threshold=0.95
)
```

**Pros**: Survives restarts, larger capacity
**Cons**: Slower than memory, single-node only

### Redis Cache

Distributed cache for production deployments.

```python
from agentic_rag.caching import RedisSemanticCache

cache = RedisSemanticCache(
    embedder=embedder,
    redis_url="redis://localhost:6379/0",
    similarity_threshold=0.95,
    ttl_seconds=86400  # 24 hours
)
```

**Architecture**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  App Node 1 │────▶│             │◀────│  App Node 2 │
└─────────────┘     │    Redis    │     └─────────────┘
                    │   Cluster   │
┌─────────────┐     │             │     ┌─────────────┐
│  App Node 3 │────▶│  (shared)   │◀────│  App Node 4 │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Redis Data Structure**:
```
Key: "rag:cache:{hash}"
Value: {
    "query": "original query string",
    "embedding": [0.1, 0.2, ...],  # Stored as bytes
    "response": "cached response",
    "sources": [...],
    "created_at": 1704067200,
    "hits": 5
}
```

**Pros**: Shared across nodes, persistent, scalable
**Cons**: Network latency, external dependency

---

## Similarity Search

### Exact Nearest Neighbor

For small caches (< 10,000 entries), brute-force search is fast enough:

```python
def find_similar(query_embedding, cache_embeddings, threshold):
    # Compute similarities with all cached embeddings
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        cache_embeddings
    )[0]

    # Find matches above threshold
    matches = np.where(similarities > threshold)[0]

    if len(matches) > 0:
        best_idx = matches[np.argmax(similarities[matches])]
        return best_idx, similarities[best_idx]

    return None, 0.0
```

### Approximate Nearest Neighbor (ANN)

For large caches, use ANN algorithms:

| Algorithm | Library | Complexity |
|-----------|---------|------------|
| HNSW | hnswlib, faiss | O(log n) |
| IVF | faiss | O(√n) |
| LSH | datasketch | O(1) expected |

**FAISS Example**:
```python
import faiss

class FAISSCache:
    def __init__(self, dimension):
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors

    def add(self, embedding):
        self.index.add(np.array([embedding], dtype='float32'))

    def search(self, embedding, threshold):
        distances, indices = self.index.search(
            np.array([embedding], dtype='float32'),
            k=1
        )
        similarity = 1 - distances[0][0]  # Convert distance to similarity
        if similarity > threshold:
            return indices[0][0], similarity
        return None, 0.0
```

---

## Cache Management

### TTL (Time-To-Live)

Entries expire after a specified time:

```python
cache = SemanticCache(
    ttl_seconds=3600,  # Expire after 1 hour
)
```

**Use cases for different TTLs**:

| TTL | Use Case |
|-----|----------|
| 5 min | Real-time data (stock prices) |
| 1 hour | Semi-static data (news) |
| 24 hours | Static reference data |
| 7 days | Stable knowledge bases |

### LRU Eviction

When cache is full, evict least recently used entries:

```python
class LRUSemanticCache:
    def __init__(self, max_entries=10000):
        self.max_entries = max_entries
        self.cache = OrderedDict()

    def get(self, query):
        if query_hash in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(query_hash)
            return self.cache[query_hash]
        return None

    def set(self, query, response):
        if len(self.cache) >= self.max_entries:
            # Remove oldest entry
            self.cache.popitem(last=False)
        self.cache[query_hash] = response
```

### Cache Invalidation

Invalidate cache when underlying data changes:

```python
# Invalidate all entries for a collection
await cache.invalidate(collection="research_papers")

# Invalidate entries matching a pattern
await cache.invalidate(pattern="*transformer*")

# Clear entire cache
await cache.clear()
```

### Cache Warming

Pre-populate cache with common queries:

```python
common_queries = [
    "What is machine learning?",
    "Explain transformer architecture",
    "How does attention work?",
]

for query in common_queries:
    response = await pipeline.query(query, collection="docs")
    await cache.set(query, response)
```

---

## Cache Entry Model

```python
from pydantic import BaseModel
from datetime import datetime

class CacheEntry(BaseModel):
    """A cached query-response pair."""

    query: str
    query_embedding: list[float]
    response: str
    sources: list[str] = []
    metadata: dict = {}

    created_at: datetime
    expires_at: datetime | None = None
    hits: int = 0

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
```

---

## Configuration

### Pipeline Builder

```python
from agentic_rag.pipeline import PipelineBuilder

# In-memory cache (default)
pipeline = (
    PipelineBuilder()
    .with_cache(
        backend="memory",
        similarity_threshold=0.95,
        ttl_seconds=3600
    )
    .build()
)

# Redis cache for production
pipeline = (
    PipelineBuilder()
    .with_cache(
        backend="redis",
        redis_url="redis://localhost:6379/0",
        similarity_threshold=0.95,
        ttl_seconds=86400
    )
    .build()
)

# Disk cache for persistence
pipeline = (
    PipelineBuilder()
    .with_cache(
        backend="disk",
        cache_dir=".cache/agentic_rag",
        similarity_threshold=0.95
    )
    .build()
)
```

### Redis Configuration

```python
from agentic_rag.caching import RedisConfig, RedisSemanticCache

config = RedisConfig(
    host="redis.example.com",
    port=6379,
    db=0,
    password="secret",
    ssl=True,
    prefix="myapp:cache:",
    max_connections=20
)

cache = RedisSemanticCache(
    embedder=embedder,
    redis_config=config,
    similarity_threshold=0.95
)

# Or from URL
config = RedisConfig.from_url(
    "rediss://:password@redis.example.com:6379/0"
)
```

### Environment Variables

```bash
# Redis settings
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=secret
REDIS_MAX_CONNECTIONS=20

# Cache settings
CACHE_BACKEND=redis
CACHE_SIMILARITY_THRESHOLD=0.95
CACHE_TTL_SECONDS=3600
```

---

## Monitoring and Metrics

### Cache Statistics

```python
stats = cache.stats()
print(f"Entries: {stats['entries']}")
print(f"Total hits: {stats['total_hits']}")
print(f"Total misses: {stats['total_misses']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Avg hit latency: {stats['avg_hit_latency_ms']:.2f}ms")
print(f"Avg miss latency: {stats['avg_miss_latency_ms']:.2f}ms")
```

### Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| Hit rate | % of queries served from cache | > 30% |
| Hit latency | Time to return cached response | < 10ms |
| Miss latency | Time for full pipeline | Baseline |
| Cache size | Number of entries | Within limits |
| Eviction rate | Entries evicted per hour | Low |

---

## Best Practices

1. **Start with conservative threshold** (0.95) and adjust based on quality
2. **Monitor cache hit rate** - low rates suggest threshold is too high
3. **Implement cache warming** for known common queries
4. **Set appropriate TTL** based on data freshness requirements
5. **Use Redis** for production multi-node deployments
6. **Invalidate on data updates** to prevent stale responses

---

## References

1. GPTCache. "Semantic Caching for LLM Applications." [github.com/zilliztech/GPTCache](https://github.com/zilliztech/GPTCache)

2. Redis. "Redis Vector Similarity Search." [redis.io/docs/stack/search/reference/vectors](https://redis.io/docs/stack/search/reference/vectors/)

3. FAISS. "A library for efficient similarity search." [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
