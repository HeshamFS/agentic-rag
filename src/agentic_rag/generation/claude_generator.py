"""
Claude (Anthropic) generator implementation.

Supports Claude Sonnet 4.5, Opus 4.5, and other Claude models
via the Anthropic API.

Latest models:
- claude-sonnet-4-5-20250929 (recommended for RAG, balanced)
- claude-opus-4-5-20251101 (highest quality, most capable)
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, GenerationResult
from agentic_rag.generation.base import BaseGenerator
from agentic_rag.generation.prompt_templates import RAG_SYSTEM_PROMPT


class ClaudeGenerator(BaseGenerator):
    """
    Claude generator using the Anthropic API.

    Supports all Claude models including:
    - claude-sonnet-4-5-20250929 (recommended for RAG, latest Sonnet 4.5)
    - claude-opus-4-5-20251101 (highest quality, latest Opus 4.5)
    - claude-sonnet-4-20250514 (Sonnet 4)
    - claude-opus-4-20250514 (Opus 4)
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
        Initialize Claude generator.

        Args:
            model: Model ID. Defaults to settings.
            api_key: API key. Defaults to settings.
            max_tokens: Max output tokens. Defaults to settings.
            temperature: Sampling temperature. Defaults to settings.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._model = model or self._settings.llm_model
        self._max_tokens = max_tokens or self._settings.default_max_tokens
        self._temperature = temperature or self._settings.default_temperature

        # Get API key
        api_key = api_key or self._settings.get_api_key("claude")
        if not api_key:
            raise ValueError("Anthropic API key not configured. Set RAG_ANTHROPIC_API_KEY.")

        # Initialize client
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def provider(self) -> str:
        return "claude"

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def generate(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate a grounded response using Anthropic's Claude models.

        This method follows the standard RAG generation flow:
        1. Prompt Construction: Formats context chunks into a structured user message.
        2. API Call: Invokes the Anthropic Messages API with the provided model and parameters.
        3. Response Parsing: Extracts text content and usage statistics (tokens, latency).

        Args:
            query: The user's search question.
            context: List of retrieved context chunks for grounding.
            system_prompt: Optional custom system instructions.
            **kwargs: Generation overrides (temperature, max_tokens).

        Returns:
            GenerationResult with the response text and source attribution.
        """
        start_time = time.perf_counter()

        # Build prompt
        user_prompt = self._build_rag_prompt(query, context)

        # Prepare parameters
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        # Call API
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or RAG_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        # Extract response text
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        return GenerationResult(
            response=response_text,
            sources=context,
            confidence=0.0,  # Will be set by evaluator
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            provider="claude",
            model=self._model,
            finish_reason=response.stop_reason or "end_turn",
            latency_ms=latency_ms,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple text generation without RAG context.

        Args:
            prompt: The prompt to complete.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Returns:
            Generated text string.
        """
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "You are a helpful assistant.",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        # Extract response text
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        return response_text

    async def generate_stream(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream generation token by token.

        Args:
            query: The user query.
            context: Retrieved chunks.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Yields:
            Generated text chunks.
        """
        user_prompt = self._build_rag_prompt(query, context)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or RAG_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        ) as stream:
            async for text in stream.text_stream:
                yield text
