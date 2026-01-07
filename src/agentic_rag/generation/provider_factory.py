"""
Factory for creating LLM generators based on provider.

Provides easy switching between Claude, OpenAI, Gemini, and local models.

Supported providers and models:
- Claude: claude-sonnet-4-5-20250929, claude-opus-4-5-20251101
- OpenAI: gpt-5.2, gpt-5-mini, gpt-5-nano
- Gemini: gemini-3-pro-preview, gemini-3-flash-preview (with thinking),
          gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
- Local: Ollama models (qwen2.5:7b, llama3.3:70b, etc.)

Note: Gemini 3 models support thinking_level parameter ("low", "medium", "high").
"""

from typing import Any, Literal

from agentic_rag.config import Settings, get_settings
from agentic_rag.generation.base import BaseGenerator
from agentic_rag.generation.claude_generator import ClaudeGenerator
from agentic_rag.generation.gemini_generator import GeminiGenerator
from agentic_rag.generation.local_generator import LocalGenerator
from agentic_rag.generation.openai_generator import OpenAIGenerator

ProviderType = Literal["claude", "openai", "gemini", "local"]


class GeneratorFactory:
    """
    Factory for creating LLM generators based on provider.

    Example:
        # Create from settings
        generator = GeneratorFactory.create()

        # Create specific provider
        generator = GeneratorFactory.create("openai", model="gpt-4o")

        # Create from config
        generator = GeneratorFactory.from_config(config)
    """

    _provider_classes: dict[str, type[BaseGenerator]] = {
        "claude": ClaudeGenerator,
        "openai": OpenAIGenerator,
        "gemini": GeminiGenerator,
        "local": LocalGenerator,
    }

    @classmethod
    def create(
        cls,
        provider: ProviderType | None = None,
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs: Any,
    ) -> BaseGenerator:
        """
        Create a generator for the specified provider.

        Args:
            provider: LLM provider (claude, openai, gemini, local).
                     Defaults to settings.llm_provider.
            model: Model ID. Defaults to settings.llm_model.
            settings: Settings instance.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Configured generator instance.

        Raises:
            ValueError: If provider is unknown or not configured.
        """
        settings = settings or get_settings()
        provider = provider or settings.llm_provider

        if provider not in cls._provider_classes:
            raise ValueError(
                f"Unknown provider: {provider}. Available: {list(cls._provider_classes.keys())}"
            )

        # Validate provider configuration
        if provider != "local" and not settings.validate_provider_config(provider):
            raise ValueError(
                f"Provider '{provider}' is not configured. "
                f"Set the appropriate API key environment variable."
            )

        # Get provider class
        generator_class = cls._provider_classes[provider]

        # Create generator
        return generator_class(
            model=model,
            settings=settings,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config: Any,  # RAGConfig
        settings: Settings | None = None,
    ) -> BaseGenerator:
        """
        Create a generator from RAGConfig.

        Args:
            config: RAGConfig instance.
            settings: Settings instance.

        Returns:
            Configured generator instance.
        """
        return cls.create(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            settings=settings,
        )

    @classmethod
    def list_providers(cls) -> list[str]:
        """List available providers."""
        return list(cls._provider_classes.keys())

    @classmethod
    def get_default_models(cls, provider: ProviderType) -> list[str]:
        """
        Get recommended models for a provider.

        Args:
            provider: LLM provider.

        Returns:
            List of recommended model IDs.
        """
        models = {
            "claude": [
                "claude-sonnet-4-5-20250929",  # Latest Sonnet 4.5
                "claude-opus-4-5-20251101",  # Latest Opus 4.5
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
            ],
            "openai": [
                "gpt-5.2",  # Latest GPT-5.2 (most capable)
                "gpt-5-mini",  # GPT-5 Mini (balanced)
                "gpt-5-nano",  # GPT-5 Nano (fast, efficient)
            ],
            "gemini": [
                "gemini-3-pro-preview",  # Latest Gemini 3 Pro
                "gemini-3-flash-preview",  # Latest Gemini 3 Flash
                "gemini-2.5-pro",  # Gemini 2.5 Pro (stable)
                "gemini-2.5-flash",  # Gemini 2.5 Flash (fast)
                "gemini-2.5-flash-lite",  # Gemini 2.5 Flash Lite (efficient)
            ],
            "local": [
                "qwen2.5:7b",
                "qwen2.5:14b",
                "llama3.3:70b",
                "mistral:7b",
            ],
        }
        return models.get(provider, [])


# Convenience function
def create_generator(
    provider: ProviderType | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BaseGenerator:
    """
    Create a generator for the specified provider.

    Convenience wrapper around GeneratorFactory.create().

    Args:
        provider: LLM provider (claude, openai, gemini, local).
        model: Model ID.
        **kwargs: Additional arguments.

    Returns:
        Configured generator instance.
    """
    return GeneratorFactory.create(provider=provider, model=model, **kwargs)
