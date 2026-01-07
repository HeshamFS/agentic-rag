"""
OpenAI generator implementation for GPT-5 family models.

GPT-5 Family Models:
- gpt-5.2 (most capable, recommended for complex RAG)
- gpt-5.1 (high capability)
- gpt-5 (base GPT-5)
- gpt-5-mini (balanced performance and cost)
- gpt-5-nano (fast, efficient for simple queries)

GPT-5 API Parameters:
- Uses max_completion_tokens (not max_tokens)
- Uses reasoning.effort for thinking depth (none/low/medium/high/xhigh)
- Does NOT support temperature parameter
"""

import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, GenerationResult
from agentic_rag.generation.base import BaseGenerator
from agentic_rag.generation.prompt_templates import RAG_SYSTEM_PROMPT

# Type for reasoning effort levels
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


class OpenAIGenerator(BaseGenerator):
    """
    OpenAI generator for GPT-5 family models.

    GPT-5 Family:
    - gpt-5.2 (most capable, best quality)
    - gpt-5.1 (high capability)
    - gpt-5 (base model)
    - gpt-5-mini (balanced, recommended for RAG)
    - gpt-5-nano (fast, efficient)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = "medium",
        settings: Settings | None = None,
    ):
        """
        Initialize OpenAI generator for GPT-5 family.

        Args:
            model: Model ID. Defaults to "gpt-5-mini".
            api_key: API key. Defaults to settings.
            base_url: API base URL (for Azure or proxies).
            max_tokens: Max output tokens (uses max_completion_tokens).
            reasoning_effort: Thinking depth - none/low/medium/high/xhigh.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._model = model or "gpt-5-mini"
        self._max_tokens = max_tokens or self._settings.default_max_tokens
        self._reasoning_effort: ReasoningEffort = reasoning_effort
        self._current_reasoning_effort: ReasoningEffort = reasoning_effort

        # Get API key
        api_key = api_key or self._settings.get_api_key("openai")
        if not api_key:
            raise ValueError("OpenAI API key not configured. Set RAG_OPENAI_API_KEY.")

        # Get base URL
        base_url = base_url or self._settings.openai_base_url

        # Initialize client
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        """Get the current reasoning effort level."""
        return self._reasoning_effort

    def _build_reasoning_config(self, effort: ReasoningEffort | None = None) -> dict[str, Any]:
        """
        Build the reasoning configuration for GPT-5 models.

        Note: The reasoning.effort parameter is stored for future API support.
        Currently, GPT-5 models use internal reasoning without explicit API parameters.

        Args:
            effort: Override reasoning effort level.

        Returns:
            Empty dict for now (reasoning is automatic in GPT-5).
        """
        # Store effort for metadata but don't send to API yet
        # The OpenAI SDK will support this parameter when GPT-5 is released
        self._current_reasoning_effort = effort or self._reasoning_effort
        # Return empty dict - GPT-5 handles reasoning automatically
        return {}

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
        Generate a response using GPT-5 family models.

        Args:
            query: The user query.
            context: Retrieved chunks for grounding.
            system_prompt: Optional system prompt override.
            **kwargs: Additional parameters:
                - max_tokens: Max output tokens
                - reasoning_effort: Thinking depth (none/low/medium/high/xhigh)

        Returns:
            GenerationResult with response and metadata.
        """
        start_time = time.perf_counter()

        # Build prompt
        user_prompt = self._build_rag_prompt(query, context)

        # Prepare parameters for GPT-5
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        reasoning_effort = kwargs.get("reasoning_effort", self._reasoning_effort)

        # GPT-5 supports system prompts
        messages = []
        if system_prompt or RAG_SYSTEM_PROMPT:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt or RAG_SYSTEM_PROMPT,
                }
            )
        messages.append({"role": "user", "content": user_prompt})

        # Store reasoning effort for metadata
        self._build_reasoning_config(reasoning_effort)

        # Build API call kwargs for GPT-5
        # GPT-5 uses max_completion_tokens, NO temperature
        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }

        # Call API
        response = await self._client.chat.completions.create(**api_kwargs)

        # Extract response text
        response_text = response.choices[0].message.content or ""

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Get token usage (GPT-5 includes reasoning_tokens in usage)
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        # GPT-5 may include reasoning tokens separately
        reasoning_tokens = 0
        if usage and hasattr(usage, "completion_tokens_details"):
            details = usage.completion_tokens_details
            if details and hasattr(details, "reasoning_tokens"):
                reasoning_tokens = details.reasoning_tokens or 0

        return GenerationResult(
            response=response_text,
            sources=context,
            confidence=0.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            provider="openai",
            model=self._model,
            finish_reason=response.choices[0].finish_reason or "stop",
            latency_ms=latency_ms,
            metadata={
                "reasoning_tokens": reasoning_tokens,
                "reasoning_effort": self._current_reasoning_effort,
            },
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
        Simple text generation without RAG context using GPT-5.

        Args:
            prompt: The prompt to complete.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters:
                - max_tokens: Max output tokens
                - reasoning_effort: Thinking depth (none/low/medium/high/xhigh)

        Returns:
            Generated text string.
        """
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        reasoning_effort = kwargs.get("reasoning_effort", self._reasoning_effort)

        # Store reasoning effort for metadata
        self._build_reasoning_config(reasoning_effort)

        # GPT-5 supports system prompts
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # GPT-5 uses max_completion_tokens, NO temperature
        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }

        response = await self._client.chat.completions.create(**api_kwargs)

        return response.choices[0].message.content or ""

    async def generate_stream(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream generation token by token using GPT-5.

        Args:
            query: The user query.
            context: Retrieved chunks.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters:
                - max_tokens: Max output tokens
                - reasoning_effort: Thinking depth (none/low/medium/high/xhigh)

        Yields:
            Generated text chunks.
        """
        user_prompt = self._build_rag_prompt(query, context)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        reasoning_effort = kwargs.get("reasoning_effort", self._reasoning_effort)

        # Store reasoning effort for metadata
        self._build_reasoning_config(reasoning_effort)

        # GPT-5 supports system prompts
        messages = []
        if system_prompt or RAG_SYSTEM_PROMPT:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt or RAG_SYSTEM_PROMPT,
                }
            )
        messages.append({"role": "user", "content": user_prompt})

        # GPT-5 uses max_completion_tokens, NO temperature
        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "max_completion_tokens": max_tokens,
        }

        stream = await self._client.chat.completions.create(**api_kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
