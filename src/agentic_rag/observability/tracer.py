"""
OpenTelemetry tracing for RAG pipelines.

Provides distributed tracing for monitoring and debugging
RAG pipeline executions.
"""

import functools
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from typing import Any, TypeVar

from agentic_rag.config import Settings, get_settings

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class SpanContext:
    """Context for a trace span."""

    def __init__(self, name: str, attributes: dict[str, Any] | None = None):
        self.name = name
        self.attributes = attributes or {}
        self.start_time = time.perf_counter()
        self.end_time: float | None = None
        self.status = "ok"
        self.error: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_error(self, error: Exception) -> None:
        """Mark span as error."""
        self.status = "error"
        self.error = str(error)

    def end(self) -> None:
        """End the span."""
        self.end_time = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


class RAGTracer:
    """
    Tracer for RAG pipeline operations.

    Supports OpenTelemetry when available, falls back to
    simple logging otherwise.
    """

    def __init__(self, service_name: str = "agentic-rag", settings: Settings | None = None):
        """
        Initialize tracer.

        Args:
            service_name: Service name for traces.
            settings: Settings instance.
        """
        self._service_name = service_name
        self._settings = settings or get_settings()
        self._tracer = None
        self._spans: list[SpanContext] = []

        if self._settings.enable_tracing:
            self._init_otel()

    def _init_otel(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Create resource
            resource = Resource.create({"service.name": self._service_name})

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Add exporter if URL configured
            if self._settings.trace_export_url:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    exporter = OTLPSpanExporter(endpoint=self._settings.trace_export_url)
                    provider.add_span_processor(BatchSpanProcessor(exporter))
                except ImportError:
                    pass  # OTLP exporter not installed

            # Set global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer
            self._tracer = trace.get_tracer(self._service_name)

        except ImportError:
            # OpenTelemetry not installed
            self._tracer = None

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None):
        """
        Create a trace span (sync context manager).

        Args:
            name: Span name.
            attributes: Span attributes.

        Yields:
            Span context for adding attributes.
        """
        ctx = SpanContext(name, attributes)
        self._spans.append(ctx)

        if self._tracer:
            from opentelemetry import trace

            with self._tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, str(v))
                try:
                    yield ctx
                except Exception as e:
                    ctx.set_error(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
                finally:
                    ctx.end()
        else:
            try:
                yield ctx
            except Exception as e:
                ctx.set_error(e)
                raise
            finally:
                ctx.end()

    @asynccontextmanager
    async def aspan(self, name: str, attributes: dict[str, Any] | None = None):
        """
        Create a trace span (async context manager).

        Args:
            name: Span name.
            attributes: Span attributes.

        Yields:
            Span context for adding attributes.
        """
        ctx = SpanContext(name, attributes)
        self._spans.append(ctx)

        if self._tracer:
            from opentelemetry import trace

            with self._tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, str(v))
                try:
                    yield ctx
                except Exception as e:
                    ctx.set_error(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
                finally:
                    ctx.end()
        else:
            try:
                yield ctx
            except Exception as e:
                ctx.set_error(e)
                raise
            finally:
                ctx.end()

    def get_recent_spans(self, limit: int = 100) -> list[SpanContext]:
        """Get recent spans for debugging."""
        return self._spans[-limit:]

    def clear_spans(self) -> None:
        """Clear stored spans."""
        self._spans.clear()


# Global tracer instance
_tracer: RAGTracer | None = None


def get_tracer() -> RAGTracer:
    """Get or create global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = RAGTracer()
    return _tracer


def trace_function(name: str | None = None):
    """
    Decorator to trace a sync function.

    Args:
        name: Span name. Defaults to function name.

    Returns:
        Decorated function.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.span(span_name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def trace_async(name: str | None = None):
    """
    Decorator to trace an async function.

    Args:
        name: Span name. Defaults to function name.

    Returns:
        Decorated async function.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            async with tracer.aspan(span_name):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
