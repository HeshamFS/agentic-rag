"""
Metrics collection for RAG pipelines.

Tracks key performance indicators for monitoring
and optimization.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricValue:
    """A single metric measurement."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""

    count: int = 0
    total: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    last_value: float = 0.0

    def add(self, value: float) -> None:
        """Add a value to the summary."""
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.last_value = value

    @property
    def mean(self) -> float:
        """Calculate mean value."""
        return self.total / self.count if self.count > 0 else 0.0


class RAGMetrics:
    """
    Metrics collector for RAG operations.

    Tracks:
    - Retrieval latency and quality
    - Generation latency and token usage
    - Cache hit rates
    - Error rates
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[MetricValue]] = defaultdict(list)
        self._summaries: dict[str, MetricSummary] = defaultdict(MetricSummary)

    # Counter operations
    def increment(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        self._counters[key] += value

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    # Gauge operations
    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)

    # Histogram operations
    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        self._histograms[key].append(MetricValue(value=value, labels=labels or {}))
        self._summaries[key].add(value)

    def get_summary(self, name: str, labels: dict[str, str] | None = None) -> MetricSummary:
        """Get summary for a histogram."""
        key = self._make_key(name, labels)
        return self._summaries.get(key, MetricSummary())

    # Helper methods
    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    # RAG-specific metrics
    def record_retrieval(
        self,
        latency_ms: float,
        num_chunks: int,
        retrieval_type: str,
        collection: str,
    ) -> None:
        """Record retrieval metrics."""
        labels = {"type": retrieval_type, "collection": collection}
        self.observe("rag_retrieval_latency_ms", latency_ms, labels)
        self.observe("rag_retrieval_chunks", float(num_chunks), labels)
        self.increment("rag_retrieval_total", labels=labels)

    def record_generation(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
    ) -> None:
        """Record generation metrics."""
        labels = {"provider": provider, "model": model}
        self.observe("rag_generation_latency_ms", latency_ms, labels)
        self.observe("rag_generation_input_tokens", float(input_tokens), labels)
        self.observe("rag_generation_output_tokens", float(output_tokens), labels)
        self.increment("rag_generation_total", labels=labels)

    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit."""
        self.increment("rag_cache_hits", labels={"type": cache_type})

    def record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss."""
        self.increment("rag_cache_misses", labels={"type": cache_type})

    def record_error(self, error_type: str, component: str) -> None:
        """Record an error."""
        self.increment("rag_errors", labels={"type": error_type, "component": component})

    def record_query(self, collection: str, mode: str) -> None:
        """Record a query."""
        self.increment("rag_queries_total", labels={"collection": collection, "mode": mode})

    def record_ingestion(self, collection: str, num_documents: int, num_chunks: int) -> None:
        """Record ingestion metrics."""
        labels = {"collection": collection}
        self.observe("rag_ingestion_documents", float(num_documents), labels)
        self.observe("rag_ingestion_chunks", float(num_chunks), labels)
        self.increment("rag_ingestion_total", labels=labels)

    # Export methods
    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "summaries": {
                k: {
                    "count": v.count,
                    "total": v.total,
                    "mean": v.mean,
                    "min": v.min_value if v.count > 0 else None,
                    "max": v.max_value if v.count > 0 else None,
                    "last": v.last_value,
                }
                for k, v in self._summaries.items()
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._summaries.clear()


# Global metrics instance
_metrics: RAGMetrics | None = None


def get_metrics() -> RAGMetrics:
    """Get or create global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = RAGMetrics()
    return _metrics
