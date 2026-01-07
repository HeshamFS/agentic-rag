"""
Google Gemini generator implementation.

Supports Gemini 3 and Gemini 2.5 models via the Google Generative AI API.

Latest models:
- gemini-3-pro-preview (most capable, with thinking)
- gemini-3-flash-preview (fast, with thinking)
- gemini-2.5-pro (stable, high quality)
- gemini-2.5-flash (fast, balanced, recommended for RAG)
- gemini-2.5-flash-lite (efficient, lightweight)

Note: Gemini 3 models use the new google.genai SDK with thinking_config support.
      Gemini 2.5 models use the google.generativeai SDK.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, GenerationResult
from agentic_rag.generation.base import BaseGenerator
from agentic_rag.generation.prompt_templates import RAG_SYSTEM_PROMPT

# Type alias for thinking level
ThinkingLevel = Literal["low", "medium", "high"]


def _is_gemini_3(model: str) -> bool:
    """Check if model is a Gemini 3 model."""
    return model.startswith("gemini-3")


class GeminiGenerator(BaseGenerator):
    """
    Gemini generator using the Google Generative AI API.

    Supports Gemini 3 and 2.5 models:
    - gemini-3-pro-preview (latest, most capable, with thinking)
    - gemini-3-flash-preview (latest, fast, with thinking)
    - gemini-2.5-pro (stable, high quality)
    - gemini-2.5-flash (recommended for RAG, fast)
    - gemini-2.5-flash-lite (efficient, lightweight)

    Gemini 3 models use the new SDK with thinking_config support.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking_level: ThinkingLevel | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize Gemini generator.

        Args:
            model: Model ID. Defaults to "gemini-2.5-flash".
            api_key: API key. Defaults to settings.
            max_tokens: Max output tokens.
            temperature: Sampling temperature (1.0 recommended for Gemini 3).
            thinking_level: Thinking level for Gemini 3 ("low", "medium", "high").
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._model = model or "gemini-2.5-flash"
        self._max_tokens = max_tokens or self._settings.default_max_tokens
        self._thinking_level = thinking_level or self._settings.gemini_thinking_level

        # Gemini 3 recommends temperature=1.0
        if temperature is not None:
            self._temperature = temperature
        elif _is_gemini_3(self._model):
            self._temperature = 1.0
        else:
            self._temperature = self._settings.default_temperature

        # Get API key
        self._api_key = api_key or self._settings.get_api_key("gemini")
        if not self._api_key:
            raise ValueError("Google API key not configured. Set RAG_GOOGLE_API_KEY.")

        # Initialize the appropriate client based on model
        self._is_gemini_3 = _is_gemini_3(self._model)

        if self._is_gemini_3:
            self._init_gemini_3_client()
        else:
            self._init_gemini_2_client()

    def _init_gemini_3_client(self) -> None:
        """Initialize Gemini 3 client using the new google.genai SDK."""
        try:
            from google import genai

            self._genai_module = genai
            self._client = genai.Client(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "Gemini 3 requires the new google-genai SDK. Install with: pip install google-genai"
            )

    def _init_gemini_2_client(self) -> None:
        """Initialize Gemini 2.5 client using google.generativeai SDK."""
        try:
            import google.generativeai as genai

            self._genai_module = genai
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(
                model_name=self._model,
                system_instruction=RAG_SYSTEM_PROMPT,
            )
        except ImportError:
            raise ImportError(
                "Gemini 2.5 requires google-generativeai SDK. "
                "Install with: pip install google-generativeai"
            )

    @property
    def provider(self) -> str:
        return "gemini"

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
        thinking_level: ThinkingLevel | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate a grounded response using Google's Gemini models.

        Supports both Gemini 2.x (stable) and Gemini 3.x (with reasoning/thinking).
        For Gemini 3, it can return thinking process details in the metadata.

        Args:
            query: The user's search question.
            context: List of retrieved context chunks for grounding.
            system_prompt: Optional custom system instructions.
            thinking_level: Reasoning depth for Gemini 3 ("low", "medium", "high").
            **kwargs: Additional generation parameters.

        Returns:
            GenerationResult with response text, sources, and optional thinking trace.
        """
        start_time = time.perf_counter()

        # Build prompt
        user_prompt = self._build_rag_prompt(query, context)

        # Prepare parameters
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        thinking = thinking_level or self._thinking_level

        thinking_text = ""
        if self._is_gemini_3:
            (
                response_text,
                input_tokens,
                output_tokens,
                thinking_text,
            ) = await self._generate_gemini_3(
                user_prompt,
                system_prompt or RAG_SYSTEM_PROMPT,
                max_tokens,
                temperature,
                thinking,
            )
        else:
            response_text, input_tokens, output_tokens = await self._generate_gemini_2(
                user_prompt,
                system_prompt,
                max_tokens,
                temperature,
            )

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        return GenerationResult(
            response=response_text,
            sources=context,
            confidence=0.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            provider="gemini",
            model=self._model,
            finish_reason="stop",
            latency_ms=latency_ms,
            metadata={"thinking": thinking_text} if thinking_text else {},
        )

    async def _generate_gemini_3(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        thinking_level: ThinkingLevel,
    ) -> tuple[str, int, int, str]:
        """Generate using Gemini 3 API with thinking support.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens, thinking_text)
        """
        from google.genai import types

        # Build config with thinking
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        )

        # Run in executor for async compatibility
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            ),
        )

        # Extract response text AND thinking separately
        response_text = ""
        thinking_text = ""
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                # This is a thinking part
                thinking_text += part.text if hasattr(part, "text") else ""
            else:
                # This is the actual response
                response_text += part.text if hasattr(part, "text") else ""

        # Get token counts
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0)
            output_tokens = getattr(usage, "candidates_token_count", 0)

        return response_text, input_tokens, output_tokens, thinking_text

    async def _generate_gemini_2(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int]:
        """Generate using Gemini 2.5 API."""
        import google.generativeai as genai

        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # Create model with custom system prompt if provided
        if system_prompt:
            client = genai.GenerativeModel(
                model_name=self._model,
                system_instruction=system_prompt,
            )
        else:
            client = self._client

        # Run in executor for async compatibility
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.generate_content(
                prompt,
                generation_config=generation_config,
            ),
        )

        # Extract response text
        response_text = response.text if response.text else ""

        # Get token counts
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0)
            output_tokens = getattr(usage, "candidates_token_count", 0)

        return response_text, input_tokens, output_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        thinking_level: ThinkingLevel | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple text generation without RAG context.

        Args:
            prompt: The prompt to complete.
            system_prompt: Optional system prompt.
            thinking_level: Override thinking level for Gemini 3.
            **kwargs: Additional parameters.

        Returns:
            Generated text string.
        """
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        thinking = thinking_level or self._thinking_level

        if self._is_gemini_3:
            response_text, _, _, _ = await self._generate_gemini_3(
                prompt,
                system_prompt or "",
                max_tokens,
                temperature,
                thinking,
            )
        else:
            response_text, _, _ = await self._generate_gemini_2(
                prompt,
                system_prompt,
                max_tokens,
                temperature,
            )

        return response_text

    async def generate_stream(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream generation.

        Note: Streaming with thinking models may include thinking tokens.
        This filters them out for clean output.

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

        if self._is_gemini_3:
            async for chunk in self._stream_gemini_3(
                user_prompt,
                system_prompt or RAG_SYSTEM_PROMPT,
                max_tokens,
                temperature,
            ):
                yield chunk
        else:
            async for chunk in self._stream_gemini_2(
                user_prompt,
                system_prompt,
                max_tokens,
                temperature,
            ):
                yield chunk

    async def _stream_gemini_3(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream using Gemini 3 API."""
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_level=self._thinking_level),
        )

        loop = asyncio.get_event_loop()

        def generate_sync():
            return self._client.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=config,
            )

        response_stream = await loop.run_in_executor(None, generate_sync)

        # Yield chunks, filtering out thinking parts
        for chunk in response_stream:
            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if not getattr(part, "thought", False) and hasattr(part, "text"):
                        yield part.text

    async def _stream_gemini_2(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream using Gemini 2.5 API."""
        import google.generativeai as genai

        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        if system_prompt:
            client = genai.GenerativeModel(
                model_name=self._model,
                system_instruction=system_prompt,
            )
        else:
            client = self._client

        loop = asyncio.get_event_loop()

        def generate_sync():
            return client.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True,
            )

        response_stream = await loop.run_in_executor(None, generate_sync)

        # Yield chunks
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
