"""
Comprehensive unit tests for utility modules.

Tests:
- Rate limiting (TokenBucket, SlidingWindow)
- Retry utilities (exponential backoff, jitter)
- Async utilities
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_rag.utils.rate_limiter import (
    RateLimitedClient,
    SlidingWindowLimiter,
    TokenBucket,
)
from agentic_rag.utils.retry import (
    RetryConfig,
    retry_async,
    retry_sync,
    retry_with_backoff,
    retry_with_jitter,
)

# =============================================================================
# TokenBucket Tests
# =============================================================================


class TestTokenBucket:
    """Tests for the TokenBucket rate limiter."""

    def test_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(rate=10.0, capacity=20)

        assert bucket.rate == 10.0
        assert bucket.capacity == 20
        assert bucket.tokens == 20.0

    def test_default_capacity(self):
        """Test default capacity equals rate."""
        bucket = TokenBucket(rate=5.0)

        assert bucket.capacity == 5

    @pytest.mark.asyncio
    async def test_acquire_immediate(self):
        """Test immediate acquisition when tokens available."""
        bucket = TokenBucket(rate=10.0, capacity=10)

        wait_time = await bucket.acquire(1)

        assert wait_time == 0.0
        assert bucket.tokens == 9.0

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens."""
        bucket = TokenBucket(rate=10.0, capacity=10)

        await bucket.acquire(5)

        assert bucket.tokens == 5.0

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self):
        """Test acquisition waits when no tokens."""
        bucket = TokenBucket(rate=10.0, capacity=2)

        # Use all tokens
        await bucket.acquire(2)

        start = time.time()
        await bucket.acquire(1)
        elapsed = time.time() - start

        # Should have waited ~0.1 seconds (1 token at 10/sec)
        assert elapsed >= 0.05

    def test_try_acquire_success(self):
        """Test non-blocking acquisition success."""
        bucket = TokenBucket(rate=10.0, capacity=10)

        result = bucket.try_acquire(1)

        assert result is True
        assert bucket.tokens == 9.0

    def test_try_acquire_fail(self):
        """Test non-blocking acquisition failure."""
        bucket = TokenBucket(rate=10.0, capacity=1)

        bucket.try_acquire(1)
        result = bucket.try_acquire(1)

        assert result is False

    @pytest.mark.asyncio
    async def test_token_replenishment(self):
        """Test tokens are replenished over time."""
        bucket = TokenBucket(rate=100.0, capacity=10)

        # Use all tokens
        await bucket.acquire(10)
        assert bucket.tokens == 0.0

        # Wait for replenishment
        await asyncio.sleep(0.1)

        # Should have some tokens now
        bucket._add_tokens()
        assert bucket.tokens > 0


# =============================================================================
# SlidingWindowLimiter Tests
# =============================================================================


class TestSlidingWindowLimiter:
    """Tests for the SlidingWindowLimiter."""

    def test_initialization(self):
        """Test sliding window initialization."""
        limiter = SlidingWindowLimiter(max_requests=10, window_seconds=1.0)

        assert limiter.max_requests == 10
        assert limiter.window_seconds == 1.0

    @pytest.mark.asyncio
    async def test_acquire_immediate(self):
        """Test immediate acquisition under limit."""
        limiter = SlidingWindowLimiter(max_requests=10, window_seconds=1.0)

        wait_time = await limiter.acquire()

        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_waits_at_limit(self):
        """Test acquisition waits when at limit."""
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=0.5)

        # Fill the window
        await limiter.acquire()
        await limiter.acquire()

        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        # Should have waited for window to slide
        assert elapsed >= 0.3

    def test_try_acquire_success(self):
        """Test non-blocking acquisition success."""
        limiter = SlidingWindowLimiter(max_requests=10, window_seconds=1.0)

        result = limiter.try_acquire()

        assert result is True

    def test_try_acquire_fail(self):
        """Test non-blocking acquisition failure at limit."""
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=10.0)

        limiter.try_acquire()
        result = limiter.try_acquire()

        assert result is False

    def test_remaining_property(self):
        """Test remaining requests property."""
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=10.0)

        assert limiter.remaining == 5

        limiter.try_acquire()
        limiter.try_acquire()

        assert limiter.remaining == 3


# =============================================================================
# RateLimitedClient Tests
# =============================================================================


class TestRateLimitedClient:
    """Tests for the RateLimitedClient wrapper."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client."""
        client = MagicMock()
        client.async_method = AsyncMock(return_value="result")
        client.attribute = "value"
        return client

    @pytest.fixture
    def rate_limited_client(self, mock_client):
        """Create rate limited client."""
        return RateLimitedClient(mock_client, rate=100.0)

    @pytest.mark.asyncio
    async def test_call_method(self, rate_limited_client, mock_client):
        """Test calling a method through the wrapper."""
        result = await rate_limited_client.call(
            mock_client.async_method,
            "arg1",
            kwarg="value",
        )

        assert result == "result"
        mock_client.async_method.assert_called_once_with("arg1", kwarg="value")

    def test_attribute_proxying(self, rate_limited_client):
        """Test attribute access is proxied."""
        assert rate_limited_client.attribute == "value"


# =============================================================================
# RetryConfig Tests
# =============================================================================


class TestRetryConfig:
    """Tests for the RetryConfig class."""

    def test_initialization(self):
        """Test retry config initialization."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0

    def test_default_values(self):
        """Test retry config defaults."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_get_delay_exponential(self):
        """Test exponential delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0

    def test_get_delay_max_cap(self):
        """Test delay is capped at max."""
        config = RetryConfig(base_delay=10.0, max_delay=20.0, jitter=False)

        delay = config.get_delay(5)  # Would be 320 without cap

        assert delay == 20.0

    def test_get_delay_with_jitter(self):
        """Test jitter adds randomness."""
        config = RetryConfig(base_delay=1.0, jitter=True)

        delays = [config.get_delay(0) for _ in range(10)]

        # Delays should vary
        assert len(set(delays)) > 1


# =============================================================================
# Retry Async Tests
# =============================================================================


class TestRetryAsync:
    """Tests for async retry functionality."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = [0]

        async def success_func():
            call_count[0] += 1
            return "success"

        result = await retry_async(success_func)

        assert result == "success"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = [0]

        async def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return "success"

        config = RetryConfig(max_attempts=5, base_delay=0.01)
        result = await retry_async(fail_then_succeed, config)

        assert result == "success"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Test exception raised when max attempts exceeded."""

        async def always_fail():
            raise ValueError("Always fails")

        config = RetryConfig(max_attempts=3, base_delay=0.01)

        with pytest.raises(ValueError, match="Always fails"):
            await retry_async(always_fail, config)

    @pytest.mark.asyncio
    async def test_retry_with_arguments(self):
        """Test retry passes arguments correctly."""

        async def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await retry_async(
            func_with_args,
            None,
            "arg1",
            "arg2",
            c="kwarg",
        )

        assert result == "arg1-arg2-kwarg"


# =============================================================================
# Retry Sync Tests
# =============================================================================


class TestRetrySync:
    """Tests for sync retry functionality."""

    def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = [0]

        def success_func():
            call_count[0] += 1
            return "success"

        result = retry_sync(success_func)

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = [0]

        def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("Temporary failure")
            return "success"

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        result = retry_sync(fail_then_succeed, config)

        assert result == "success"
        assert call_count[0] == 2


# =============================================================================
# Decorator Tests
# =============================================================================


class TestRetryDecorators:
    """Tests for retry decorators."""

    @pytest.mark.asyncio
    async def test_retry_with_backoff_decorator(self):
        """Test retry_with_backoff decorator."""
        call_count = [0]

        @retry_with_backoff(max_attempts=3, min_wait=0.01, max_wait=0.1)
        async def flaky_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("Flaky")
            return "success"

        result = await flaky_function()

        assert result == "success"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_retry_with_jitter_decorator(self):
        """Test retry_with_jitter decorator."""
        call_count = [0]

        @retry_with_jitter(max_attempts=3, min_wait=0.01, max_wait=0.1)
        async def flaky_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("Flaky")
            return "success"

        result = await flaky_function()

        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_specific_exceptions(self):
        """Test decorator only retries specified exceptions."""
        call_count = [0]

        @retry_with_backoff(
            max_attempts=3,
            min_wait=0.01,
            exceptions=(ValueError,),
        )
        async def wrong_exception():
            call_count[0] += 1
            raise TypeError("Wrong type")

        with pytest.raises(TypeError):
            await wrong_exception()

        # Should not retry for TypeError
        assert call_count[0] == 1


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestRateLimiterPerformance:
    """Performance tests for rate limiters."""

    @pytest.mark.asyncio
    async def test_token_bucket_throughput(self):
        """Test token bucket maintains rate."""
        bucket = TokenBucket(rate=100.0, capacity=10)

        # Drain initial tokens first
        await bucket.acquire(10)

        start = time.time()
        for _ in range(50):
            await bucket.acquire(1)
        elapsed = time.time() - start

        # Should complete in ~0.5 seconds for 50 requests at 100/sec
        assert elapsed >= 0.4

    @pytest.mark.asyncio
    async def test_sliding_window_throughput(self):
        """Test sliding window maintains rate."""
        limiter = SlidingWindowLimiter(max_requests=50, window_seconds=0.5)

        start = time.time()
        for _ in range(50):
            await limiter.acquire()
        elapsed = time.time() - start

        # Should complete quickly for first 50 requests
        assert elapsed < 0.5


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_zero_tokens(self):
        """Test acquiring zero tokens."""
        bucket = TokenBucket(rate=10.0, capacity=10)

        wait_time = await bucket.acquire(0)

        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_large_token_request(self):
        """Test requesting more tokens than capacity."""
        bucket = TokenBucket(rate=10.0, capacity=5)

        # Should wait for enough tokens
        start = time.time()
        await bucket.acquire(10)
        elapsed = time.time() - start

        # Would need ~0.5 seconds to generate 5 additional tokens
        assert elapsed >= 0.4

    @pytest.mark.asyncio
    async def test_concurrent_acquisition(self):
        """Test concurrent token acquisition."""
        bucket = TokenBucket(rate=100.0, capacity=10)

        async def acquire():
            await bucket.acquire(1)

        # Run 20 concurrent acquisitions
        await asyncio.gather(*[acquire() for _ in range(20)])

        # All should complete without error
