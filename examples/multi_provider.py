"""
Multi-provider LLM example for RAG Optimizer.

This example demonstrates switching between different LLM providers:
- Claude (Anthropic)
- GPT-4o (OpenAI)
- Gemini (Google)
- Local models (Ollama)
"""

import asyncio
from agentic_rag.pipeline import PipelineBuilder
from agentic_rag.generation import GeneratorFactory
from agentic_rag.config import get_settings


async def query_with_provider(provider: str, model: str, question: str, collection: str):
    """Query using a specific provider."""
    settings = get_settings()

    pipeline = (
        PipelineBuilder(settings=settings)
        .with_generator(provider=provider, model=model)
        .build()
    )

    try:
        response = await pipeline.query(question, collection=collection)
        return {
            "provider": provider,
            "model": model,
            "response": response.response[:200] + "...",
            "latency_ms": response.latency_ms,
        }
    finally:
        await pipeline.close()


async def main():
    """Compare responses from different providers."""

    question = "What are the key features of modern RAG systems?"
    collection = "docs"

    # Define providers to compare
    providers = [
        ("claude", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o"),
        ("gemini", "gemini-2.0-flash"),
        # ("local", "qwen2.5:7b"),  # Uncomment if Ollama is running
    ]

    print(f"Question: {question}\n")
    print("=" * 60)

    for provider, model in providers:
        try:
            print(f"\n[{provider.upper()}] {model}")
            result = await query_with_provider(provider, model, question, collection)
            print(f"Response: {result['response']}")
            print(f"Latency: {result['latency_ms']:.0f}ms")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
