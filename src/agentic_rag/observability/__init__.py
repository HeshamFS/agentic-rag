"""Observability: tracing, metrics, and debugging tools."""

from agentic_rag.observability.dashboard import (
    Dashboard,
    SimpleDashboard,
    run_dashboard,
    show_dashboard,
)
from agentic_rag.observability.metrics import (
    MetricSummary,
    MetricValue,
    RAGMetrics,
    get_metrics,
)
from agentic_rag.observability.tracer import (
    RAGTracer,
    SpanContext,
    get_tracer,
    trace_async,
    trace_function,
)

__all__ = [
    # Tracer
    "RAGTracer",
    "SpanContext",
    "get_tracer",
    "trace_function",
    "trace_async",
    # Metrics
    "RAGMetrics",
    "MetricValue",
    "MetricSummary",
    "get_metrics",
    # Dashboard
    "Dashboard",
    "SimpleDashboard",
    "show_dashboard",
    "run_dashboard",
]
