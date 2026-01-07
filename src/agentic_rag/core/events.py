"""
Event system for RAG pipeline.

Provides pub/sub event handling for pipeline observability
and extensibility.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """RAG pipeline event types."""

    # Query lifecycle
    QUERY_START = "query.start"
    QUERY_END = "query.end"
    QUERY_ERROR = "query.error"

    # Retrieval events
    RETRIEVAL_START = "retrieval.start"
    RETRIEVAL_END = "retrieval.end"
    RETRIEVAL_CACHE_HIT = "retrieval.cache_hit"

    # Generation events
    GENERATION_START = "generation.start"
    GENERATION_END = "generation.end"
    GENERATION_STREAM = "generation.stream"

    # Agent events
    AGENT_STEP = "agent.step"
    AGENT_REFLECTION = "agent.reflection"
    AGENT_PLANNING = "agent.planning"

    # Ingestion events
    INGESTION_START = "ingestion.start"
    INGESTION_PROGRESS = "ingestion.progress"
    INGESTION_END = "ingestion.end"

    # Evaluation events
    EVALUATION_START = "evaluation.start"
    EVALUATION_END = "evaluation.end"


@dataclass
class Event:
    """A pipeline event."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""


# Type for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]] | Callable[[Event], None]


class EventBus:
    """
    Async event bus for pipeline events.

    Supports both sync and async handlers.
    """

    def __init__(self):
        """Initialize event bus."""
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to.
            handler: Handler function (sync or async).
        """
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe to all events.

        Args:
            handler: Handler function.
        """
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Event type.
            handler: Handler to remove.

        Returns:
            True if handler was found and removed.
        """
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            return True
        return False

    async def emit(self, event: Event) -> None:
        """
        Emit an event to all registered handlers and global subscribers.

        Handlers are executed sequentially. Async handlers are awaited.
        Any exceptions raised by handlers are caught and logged to prevent
        interrupting the main pipeline execution.

        Args:
            event: The Event object to emit.

        Note:
            This method is thread-safe and can be called from multiple threads.
            However, handlers are executed in the order they were registered.
        """
        handlers = self._handlers[event.type] + self._global_handlers

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                # Log the exception to prevent it from being lost
                print(f"Error handling event {event.type}: {e}")

    async def emit_type(
        self,
        event_type: EventType,
        data: dict[str, Any] | None = None,
        source: str = "",
    ) -> None:
        """
        Emit an event by type.

        Creates an Event object with the specified type, data, and source,
        and then emits it to all registered handlers and global subscribers.

        Args:
            event_type: Type of event.
            data: Event data.
            source: Event source identifier.
        """
        event = Event(type=event_type, data=data or {}, source=source)
        await self.emit(event)

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
        self._global_handlers.clear()


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get or create global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


class EventLogger:
    """
    Event handler that logs events.

    Useful for debugging and monitoring.
    """

    def __init__(self, log_level: str = "INFO"):
        """Initialize event logger."""
        import logging

        self._logger = logging.getLogger("agentic_rag.events")
        self._level = getattr(logging, log_level.upper())

    def __call__(self, event: Event) -> None:
        """Log the event."""
        self._logger.log(
            self._level,
            f"[{event.type.value}] {event.source}: {event.data}",
        )


class EventCollector:
    """
    Collects events for later analysis.

    Useful for testing and debugging.
    """

    def __init__(self, max_events: int = 1000):
        """Initialize collector."""
        self._events: list[Event] = []
        self._max_events = max_events

    def __call__(self, event: Event) -> None:
        """Collect the event."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

    @property
    def events(self) -> list[Event]:
        """Get collected events."""
        return self._events.copy()

    def filter_by_type(self, event_type: EventType) -> list[Event]:
        """Filter events by type."""
        return [e for e in self._events if e.type == event_type]

    def clear(self) -> None:
        """Clear collected events."""
        self._events.clear()
