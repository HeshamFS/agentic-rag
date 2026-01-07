"""LLM generation with multi-provider support (Claude, OpenAI, Gemini, Groq, Local)."""

from agentic_rag.generation.base import BaseGenerator
from agentic_rag.generation.claude_generator import ClaudeGenerator
from agentic_rag.generation.gemini_generator import GeminiGenerator
from agentic_rag.generation.groq_generator import GroqGenerator
from agentic_rag.generation.local_generator import LocalGenerator
from agentic_rag.generation.openai_generator import OpenAIGenerator
from agentic_rag.generation.prompt_templates import (
    TEMPLATES,
    PromptTemplate,
    get_template,
)
from agentic_rag.generation.provider_factory import (
    GeneratorFactory,
    ProviderType,
    create_generator,
)

__all__ = [
    # Base
    "BaseGenerator",
    # Providers
    "ClaudeGenerator",
    "OpenAIGenerator",
    "GeminiGenerator",
    "GroqGenerator",
    "LocalGenerator",
    # Factory
    "GeneratorFactory",
    "create_generator",
    "ProviderType",
    # Templates
    "PromptTemplate",
    "TEMPLATES",
    "get_template",
]
