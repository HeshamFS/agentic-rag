"""
Local LLM generator implementation.

Supports local inference via:
- Ollama (default, easiest setup)
- vLLM (high performance)
- Any OpenAI-compatible API
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, GenerationResult
from agentic_rag.generation.base import BaseGenerator
from agentic_rag.generation.prompt_templates import RAG_SYSTEM_PROMPT


class LocalGenerator(BaseGenerator):
    """
    Local LLM generator using Ollama or OpenAI-compatible APIs.

    Supports models like:
    - qwen2.5:7b (recommended for RAG)
    - llama3.3:70b (high quality)
    - mistral:7b (fast)

    Works with any OpenAI-compatible local server.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize local generator.

        Args:
            model: Model name (e.g., "qwen2.5:7b").
            base_url: Server URL (defaults to Ollama at localhost:11434).
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._model = model or self._settings.local_llm_model
        self._base_url = base_url or self._settings.local_llm_url
        self._max_tokens = max_tokens or self._settings.default_max_tokens
        self._temperature = temperature or self._settings.default_temperature

        # HTTP client for API calls
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=120.0,  # Local models can be slow
        )

    @property
    def provider(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._model

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

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
        Generate a response using local LLM.

        Args:
            query: The user query.
            context: Retrieved chunks for grounding.
            system_prompt: Optional system prompt override.
            **kwargs: Additional parameters.

        Returns:
            GenerationResult with response and metadata.
        """
        start_time = time.perf_counter()

        # Build prompt
        user_prompt = self._build_rag_prompt(query, context)

        # Prepare parameters
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        # Try OpenAI-compatible endpoint first (works with vLLM, LM Studio, etc.)
        try:
            result = await self._generate_openai_compat(
                user_prompt=user_prompt,
                system_prompt=system_prompt or RAG_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except httpx.HTTPStatusError:
            # Fall back to Ollama native API
            result = await self._generate_ollama(
                user_prompt=user_prompt,
                system_prompt=system_prompt or RAG_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        return GenerationResult(
            response=result["response"],
            sources=context,
            confidence=0.0,
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
            provider="local",
            model=self._model,
            finish_reason="stop",
            latency_ms=latency_ms,
        )

    async def _generate_openai_compat(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Generate using OpenAI-compatible API."""
        response = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return {
            "response": choice["message"]["content"],
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    async def _generate_ollama(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Generate using Ollama native API."""
        response = await self._client.post(
            "/api/generate",
            json={
                "model": self._model,
                "prompt": user_prompt,
                "system": system_prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        return {
            "response": data.get("response", ""),
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }

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

        try:
            result = await self._generate_openai_compat(
                user_prompt=prompt,
                system_prompt=system_prompt or "You are a helpful assistant.",
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except httpx.HTTPStatusError:
            result = await self._generate_ollama(
                user_prompt=prompt,
                system_prompt=system_prompt or "You are a helpful assistant.",
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return result["response"]

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

        # Use Ollama streaming API
        async with self._client.stream(
            "POST",
            "/api/generate",
            json={
                "model": self._model,
                "prompt": user_prompt,
                "system": system_prompt or RAG_SYSTEM_PROMPT,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
