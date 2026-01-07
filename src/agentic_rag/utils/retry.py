"""
Retry utilities for resilient API calls.

Provides decorators and utilities for retrying
failed operations with exponential backoff.
"""

import asyncio
import random
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

F = TypeVar("F", bound=Callable[..., Any])


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        """
        Initialize retry error.

        Args:
            message: Error message.
            last_exception: The last exception that caused the retry to fail.
        """
        super().__init__(message)
        self.last_exception = last_exception


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator that retries a function call using an exponential backoff strategy.

    The wait time between attempts grows exponentially (min_wait * 2^attempt)
    but is capped at max_wait.

    Args:
        max_attempts: The maximum number of times to try the operation.
        min_wait: The initial/minimum wait time in seconds.
        max_wait: The maximum wait time between any two attempts.
        exceptions: A tuple of exception types that should trigger a retry.

    Returns:
        A wrapped function that automatically retries on specified exceptions.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
    )


def retry_with_jitter(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying with random exponential backoff.

    Adds jitter to prevent thundering herd problem.

    Args:
        max_attempts: Maximum retry attempts.
        min_wait: Minimum wait time.
        max_wait: Maximum wait time.
        exceptions: Exception types to retry on.

    Returns:
        Decorated function.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
    )


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry config.

        Args:
            max_attempts: Maximum attempts.
            base_delay: Base delay in seconds.
            max_delay: Maximum delay in seconds.
            exponential_base: Base for exponential backoff.
            jitter: Add random jitter.
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for an attempt.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


async def retry_async(
    func: Callable,
    config: RetryConfig | None = None,
    *args,
    **kwargs,
) -> Any:
    """
    Retry an async function with backoff.

    Args:
        func: Async function to retry.
        config: Retry configuration.
        *args: Function arguments.
        **kwargs: Function keyword arguments.

    Returns:
        Function result.

    Raises:
        Last exception if all attempts fail.
    """
    config = config or RetryConfig()
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < config.max_attempts - 1:
                delay = config.get_delay(attempt)
                await asyncio.sleep(delay)

    raise last_exception  # type: ignore


def retry_sync(
    func: Callable,
    config: RetryConfig | None = None,
    *args,
    **kwargs,
) -> Any:
    """
    Retry a sync function with backoff.

    Args:
        func: Function to retry.
        config: Retry configuration.
        *args: Function arguments.
        **kwargs: Function keyword arguments.

    Returns:
        Function result.
    """
    import time

    config = config or RetryConfig()
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < config.max_attempts - 1:
                delay = config.get_delay(attempt)
                time.sleep(delay)

    raise last_exception  # type: ignore
