"""
Groq generator implementation for ultra-fast inference.

Groq provides the fastest inference speeds available:
- llama-3.1-8b-instant: 560 tokens/second (FREE tier: 30 RPM, 500K TPD)
- llama-3.3-70b-versatile: 280 tokens/second (FREE tier: 30 RPM, 100K TPD)
- openai/gpt-oss-20b: 1000 tokens/second (PAID, supports prompt caching)

Best for:
- Contextual chunking (fast context header generation)
- High-throughput batch processing
- Cost-sensitive applications

Rate Limits (Free Tier):
- 30 requests per minute
- 6,000-12,000 tokens per minute
- 100,000-500,000 tokens per day
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, GenerationResult
from agentic_rag.generation.base import BaseGenerator
from agentic_rag.generation.prompt_templates import RAG_SYSTEM_PROMPT

logger = logging.getLogger("agentic_rag.groq")


# Model recommendations
GROQ_MODELS = {
    # Free tier friendly (30 RPM, generous daily limits)
    "llama-3.1-8b-instant": {
        "speed": 560,  # tokens/second
        "context": 131072,
        "free_rpm": 30,
        "free_tpd": 500000,
        "best_for": "contextual_chunking",  # Fast, cheap, good enough quality
    },
    "llama-3.3-70b-versatile": {
        "speed": 280,
        "context": 131072,
        "free_rpm": 30,
        "free_tpd": 100000,
        "best_for": "quality",  # Better quality, lower daily limit
    },
    # Paid tier (faster, prompt caching)
    "openai/gpt-oss-20b": {
        "speed": 1000,  # Fastest!
        "context": 131072,
        "paid_rpm": 1000,
        "paid_tpm": 250000,
        "supports_caching": True,
        "best_for": "high_throughput",
    },
    "openai/gpt-oss-120b": {
        "speed": 500,
        "context": 131072,
        "paid_rpm": 1000,
        "paid_tpm": 250000,
        "supports_caching": True,
        "best_for": "quality_with_speed",
    },
}

# Default model for contextual chunking (fast + free tier friendly)
DEFAULT_CHUNKING_MODEL = "llama-3.1-8b-instant"


class GroqGenerator(BaseGenerator):
    """
    Groq generator for ultra-fast LLM inference.

    Optimized for high-throughput tasks like contextual chunking.
    Uses the Groq SDK which is OpenAI-compatible.

    Example:
        generator = GroqGenerator(model="llama-3.1-8b-instant")
        text = await generator.generate_text("Summarize this chunk...")
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize Groq generator.

        Args:
            model: Model ID. Defaults to "llama-3.1-8b-instant" (fast + free tier).
            api_key: Groq API key. Defaults to settings.
            max_tokens: Max output tokens. Defaults to 150 for chunking tasks.
            temperature: Sampling temperature. Defaults to 0.3 for consistency.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._model = model or DEFAULT_CHUNKING_MODEL
        self._max_tokens = max_tokens or 150  # Short outputs for context headers
        self._temperature = temperature or 0.3  # Low for consistency

        # Get API key (use get_api_key helper which handles SecretStr)
        self._api_key = api_key or self._settings.get_api_key("groq")
        if not self._api_key:
            raise ValueError("Groq API key not configured. Set RAG_GROQ_API_KEY in .env")

        # Initialize client
        self._init_client()

        # Track rate limiting
        self._request_count = 0
        self._last_request_time = 0.0

        logger.info(
            f"GROQ: Initialized with model={self._model} "
            f"(~{GROQ_MODELS.get(self._model, {}).get('speed', '?')} tok/s)"
        )

    def _init_client(self) -> None:
        """Initialize the Groq client."""
        try:
            from groq import AsyncGroq, Groq

            self._sync_client = Groq(api_key=self._api_key)
            self._async_client = AsyncGroq(api_key=self._api_key)
        except ImportError:
            raise ImportError("Groq SDK required. Install with: pip install groq")

    @property
    def provider(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate a response with RAG context.

        Args:
            query: The user's question.
            context: Retrieved chunks for grounding.
            system_prompt: Optional system prompt override.
            **kwargs: Additional parameters (temperature, max_tokens).

        Returns:
            GenerationResult with response and metadata.
        """
        start_time = time.perf_counter()

        # Build messages
        user_prompt = self._build_rag_prompt(query, context)
        messages = [
            {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Call Groq API
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract response
        response_text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"

        # Token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Check for cached tokens (prompt caching)
        cached_tokens = 0
        if hasattr(usage, "prompt_tokens_details"):
            details = usage.prompt_tokens_details
            cached_tokens = getattr(details, "cached_tokens", 0)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return GenerationResult(
            response=response_text,
            sources=context,
            confidence=0.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            provider="groq",
            model=self._model,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            metadata={
                "cached_tokens": cached_tokens,
                "tokens_per_second": output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0,
            },
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple text generation without RAG context.

        Optimized for fast, short outputs like context headers.

        Args:
            prompt: The prompt to complete.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Returns:
            Generated text string.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content or ""

    async def generate_text_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        max_concurrent: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generate text for multiple prompts concurrently.

        Respects rate limits by limiting concurrency.
        Free tier: 30 RPM = 2 requests/second max.

        Args:
            prompts: List of prompts.
            system_prompt: Shared system prompt.
            max_concurrent: Max concurrent requests (default 5).
            **kwargs: Additional parameters.

        Returns:
            List of generated texts.
        """

        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_one(prompt: str) -> str:
            async with semaphore:
                # Small delay to respect rate limits (30 RPM = 2/sec)
                await asyncio.sleep(0.1)
                return await self.generate_text(prompt, system_prompt, **kwargs)

        tasks = [generate_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    async def generate_stream(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream generation.

        Args:
            query: The user query.
            context: Retrieved chunks.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Yields:
            Generated text chunks.
        """
        user_prompt = self._build_rag_prompt(query, context)
        messages = [
            {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        stream = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def get_best_groq_model_for_chunking(paid_tier: bool = False) -> str:
    """
    Get the best Groq model for contextual chunking.

    Args:
        paid_tier: Whether user has paid Groq plan.

    Returns:
        Model ID string.
    """
    if paid_tier:
        # GPT-OSS-20B is fastest (1000 tok/s) with prompt caching
        return "openai/gpt-oss-20b"
    else:
        # Llama 3.1 8B is fast (560 tok/s) with generous free limits
        return "llama-3.1-8b-instant"
