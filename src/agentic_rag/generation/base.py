"""
Base generator class for LLM integration.

Provides the abstract interface for all LLM providers
with support for RAG-specific generation patterns.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from agentic_rag.core.models import Chunk, GenerationResult


class BaseGenerator(ABC):
    """
    Abstract base class for LLM generators.

    All provider implementations (Claude, OpenAI, Gemini, Local)
    must implement this interface.
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the LLM provider (claude, openai, gemini, local)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate a response given query and context.

        Args:
            query: The user query.
            context: Retrieved chunks for grounding.
            system_prompt: Optional system prompt override.
            **kwargs: Additional generation parameters.

        Returns:
            GenerationResult with response and metadata.
        """
        ...

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple text generation without RAG context.

        Used for HyDE hypothetical document generation,
        query expansion, and agent reasoning.

        Args:
            prompt: The prompt to complete.
            system_prompt: Optional system prompt.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text string.
        """
        ...

    async def generate_stream(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream generation token by token.

        Default implementation falls back to non-streaming.
        Override for true streaming support.

        Args:
            query: The user query.
            context: Retrieved chunks.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters.

        Yields:
            Generated text chunks.
        """
        result = await self.generate(query, context, system_prompt, **kwargs)
        yield result.response

    def _format_context(self, chunks: list[Chunk]) -> str:
        """
        Format chunks into context string.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Formatted context string for prompt.
        """
        if not chunks:
            return "No relevant context available."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = chunk.context_header or ""
            if header:
                parts.append(f"[Source {i}]\n{header}\n{chunk.content}")
            else:
                parts.append(f"[Source {i}]\n{chunk.content}")

        return "\n\n---\n\n".join(parts)

    def _build_rag_prompt(
        self,
        query: str,
        context: list[Chunk],
        custom_instructions: str | None = None,
    ) -> str:
        """
        Build the RAG prompt with query and context.

        Args:
            query: User query.
            context: Retrieved chunks.
            custom_instructions: Optional custom instructions.

        Returns:
            Formatted prompt string.
        """
        context_str = self._format_context(context)
        num_sources = len(context)

        prompt_parts = [
            "Use the following context to answer the question.",
            "If the context doesn't contain relevant information, say so.",
            f"You have exactly {num_sources} sources available. Only cite sources that exist ([Source 1] through [Source {num_sources}]).",
            "Do NOT cite sources beyond what is provided.",
            "",
            "Context:",
            context_str,
            "",
            "Question:",
            query,
        ]

        if custom_instructions:
            prompt_parts.insert(0, custom_instructions)
            prompt_parts.insert(1, "")

        return "\n".join(prompt_parts)
