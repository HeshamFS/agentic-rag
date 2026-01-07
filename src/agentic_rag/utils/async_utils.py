"""
Async utilities for concurrent operations.

Provides helpers for batching, parallel execution,
and async iteration.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def gather_with_concurrency[T](
    tasks: list[Coroutine[Any, Any, T]],
    max_concurrency: int = 10,
) -> list[T]:
    """
    Run tasks with limited concurrency.

    Args:
        tasks: List of coroutines to run.
        max_concurrency: Maximum concurrent tasks.

    Returns:
        List of results in order.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_task(task: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(*[bounded_task(t) for t in tasks])


async def map_async(
    func: Callable[[T], Coroutine[Any, Any, R]],
    items: list[T],
    max_concurrency: int = 10,
) -> list[R]:
    """
    Apply an async function to items with concurrency limit.

    Args:
        func: Async function to apply.
        items: Items to process.
        max_concurrency: Maximum concurrent calls.

    Returns:
        List of results.
    """
    tasks = [func(item) for item in items]
    return await gather_with_concurrency(tasks, max_concurrency)


async def batch_process(
    items: list[T],
    processor: Callable[[list[T]], Coroutine[Any, Any, list[R]]],
    batch_size: int = 100,
) -> list[R]:
    """
    Process items in batches.

    Args:
        items: Items to process.
        processor: Batch processor function.
        batch_size: Items per batch.

    Returns:
        Combined results.
    """
    results: list[R] = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await processor(batch)
        results.extend(batch_results)

    return results


async def first_completed[T](
    tasks: list[Coroutine[Any, Any, T]],
) -> tuple[T, list[asyncio.Task]]:
    """
    Run tasks and return first completed result.

    Args:
        tasks: List of coroutines.

    Returns:
        Tuple of (first result, pending tasks).
    """
    futures = [asyncio.ensure_future(t) for t in tasks]
    done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)

    # Get result from first completed
    result = done.pop().result()

    return result, list(pending)


async def timeout_task(
    task: Coroutine[Any, Any, T],
    timeout: float,
    default: T | None = None,
) -> T | None:
    """
    Run task with timeout.

    Args:
        task: Coroutine to run.
        timeout: Timeout in seconds.
        default: Default value if timeout.

    Returns:
        Task result or default.
    """
    try:
        return await asyncio.wait_for(task, timeout=timeout)
    except TimeoutError:
        return default


async def retry_task[T](
    task_factory: Callable[[], Coroutine[Any, Any, T]],
    max_attempts: int = 3,
    delay: float = 1.0,
) -> T:
    """
    Retry a task on failure.

    Args:
        task_factory: Factory that creates the coroutine.
        max_attempts: Maximum attempts.
        delay: Delay between attempts.

    Returns:
        Task result.

    Raises:
        Last exception if all attempts fail.
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await task_factory()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay * (2**attempt))

    raise last_exception  # type: ignore


class AsyncBatcher:
    """
    Batches async operations for efficiency.

    Collects operations and executes them in batches
    to reduce overhead.
    """

    def __init__(
        self,
        processor: Callable[[list[T]], Coroutine[Any, Any, list[R]]],
        batch_size: int = 32,
        max_wait: float = 0.1,
    ):
        """
        Initialize batcher.

        Args:
            processor: Batch processor function.
            batch_size: Maximum batch size.
            max_wait: Maximum wait time in seconds.
        """
        self._processor = processor
        self._batch_size = batch_size
        self._max_wait = max_wait
        self._queue: list[tuple[T, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._processing = False

    async def submit(self, item: T) -> R:
        """
        Submit an item for batch processing.

        Args:
            item: Item to process.

        Returns:
            Processing result.
        """
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        async with self._lock:
            self._queue.append((item, future))

            if len(self._queue) >= self._batch_size:
                await self._flush()
            elif not self._processing:
                self._processing = True
                asyncio.create_task(self._wait_and_flush())

        return await future

    async def _wait_and_flush(self) -> None:
        """Wait for max_wait then flush."""
        await asyncio.sleep(self._max_wait)
        async with self._lock:
            if self._queue:
                await self._flush()
            self._processing = False

    async def _flush(self) -> None:
        """Process the current batch."""
        if not self._queue:
            return

        batch = self._queue[: self._batch_size]
        self._queue = self._queue[self._batch_size :]

        items = [item for item, _ in batch]
        futures = [future for _, future in batch]

        try:
            results = await self._processor(items)
            for future, result in zip(futures, results, strict=False):
                future.set_result(result)
        except Exception as e:
            for future in futures:
                future.set_exception(e)


async def async_enumerate[T](
    iterable: AsyncIterator[T],
    start: int = 0,
) -> AsyncIterator[tuple[int, T]]:
    """
    Async version of enumerate.

    Args:
        iterable: Async iterable.
        start: Starting index.

    Yields:
        Tuples of (index, item).
    """
    i = start
    async for item in iterable:
        yield i, item
        i += 1
