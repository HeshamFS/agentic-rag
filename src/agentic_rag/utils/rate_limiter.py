"""
Rate limiting for API calls.

Provides token bucket and sliding window rate limiters
for controlling API request rates.
"""

import asyncio
import time
from collections import deque
from typing import Any


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = 0.0):
        """
        Initialize rate limit error.

        Args:
            message: Error message.
            retry_after: Seconds until rate limit resets.
        """
        super().__init__(message)
        self.retry_after = retry_after


class TokenBucket:
    """
    Token bucket rate limiter.

    Allows bursts up to bucket capacity while maintaining
    average rate over time.
    """

    def __init__(
        self,
        rate: float,
        capacity: int | None = None,
    ):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second.
            capacity: Maximum tokens (default: rate).
        """
        self.rate = rate
        self.capacity = capacity or int(rate)
        self.tokens = float(self.capacity)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire the specified number of tokens from the bucket, waiting if
        the bucket does not have enough tokens.

        This implements a "greedy" acquire that blocks until the required
        tokens are available based on the fill rate.

        Args:
            tokens: Number of tokens to acquire (default 1).

        Returns:
            The amount of time (in seconds) spent waiting.
        """
        async with self._lock:
            self._add_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            needed = tokens - self.tokens
            wait_time = needed / self.rate

            await asyncio.sleep(wait_time)

            self._add_tokens()
            self.tokens -= tokens

            return wait_time

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            True if acquired, False otherwise.
        """
        self._add_tokens()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class SlidingWindowLimiter:
    """
    Sliding window rate limiter.

    Tracks requests in a time window for precise rate limiting.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
    ):
        """
        Initialize sliding window limiter.

        Args:
            max_requests: Maximum requests per window.
            window_seconds: Window size in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = asyncio.Lock()

    def _clean_old_requests(self) -> None:
        """Remove requests outside the window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    async def acquire(self) -> float:
        """
        Acquire a request slot, waiting if necessary.

        Returns:
            Time waited in seconds.
        """
        async with self._lock:
            self._clean_old_requests()

            if len(self.requests) < self.max_requests:
                self.requests.append(time.monotonic())
                return 0.0

            # Calculate wait time
            oldest = self.requests[0]
            wait_time = oldest + self.window_seconds - time.monotonic()

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self._clean_old_requests()
            self.requests.append(time.monotonic())

            return max(0.0, wait_time)

    def try_acquire(self) -> bool:
        """
        Try to acquire a slot without waiting.

        Returns:
            True if acquired, False otherwise.
        """
        self._clean_old_requests()

        if len(self.requests) < self.max_requests:
            self.requests.append(time.monotonic())
            return True
        return False

    @property
    def remaining(self) -> int:
        """Get remaining requests in window."""
        self._clean_old_requests()
        return max(0, self.max_requests - len(self.requests))


class RateLimitedClient:
    """
    Wrapper that adds rate limiting to any async client.

    Example:
        limiter = RateLimitedClient(my_client, rate=10)
        result = await limiter.call(my_client.some_method, arg1, arg2)
    """

    def __init__(
        self,
        client: Any,
        rate: float = 10.0,
        capacity: int | None = None,
    ):
        """
        Initialize rate limited client.

        Args:
            client: Client to wrap.
            rate: Requests per second.
            capacity: Burst capacity.
        """
        self.client = client
        self.limiter = TokenBucket(rate=rate, capacity=capacity)

    async def call(self, method: Any, *args, **kwargs) -> Any:
        """
        Call a method with rate limiting.

        Args:
            method: Method to call.
            *args: Method arguments.
            **kwargs: Method keyword arguments.

        Returns:
            Method result.
        """
        await self.limiter.acquire()
        return await method(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to wrapped client."""
        return getattr(self.client, name)
