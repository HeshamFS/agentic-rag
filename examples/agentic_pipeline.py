"""
Agentic pipeline example for RAG Optimizer.

This example demonstrates the agentic pipeline with:
1. Multi-agent orchestration
2. Reflection and self-evaluation
3. CRAG (Corrective RAG) patterns
"""

import asyncio
from agentic_rag.core.models import Document
from agentic_rag.pipeline import PipelineBuilder
from agentic_rag.config import get_settings


async def main():
    """Run agentic RAG pipeline example."""

    settings = get_settings()

    # Build agentic pipeline
    pipeline = (
        PipelineBuilder(settings=settings)
        .with_embedder(model="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        .with_chunking("semantic", chunk_size=512, contextual=True)
        .with_retrieval("hybrid", use_hyde=True, use_rrf=True)
        .with_reranker("jina")
        .with_generator(provider="claude", model="claude-sonnet-4-20250514")
        .with_evaluation(enable_ragas=True, enable_self_rag=True)
        .as_agentic()  # Enable multi-agent orchestration
        .build()
    )

    # Complex multi-hop question
    question = """
    Compare the programming paradigms supported by Python with
    how machine learning algorithms process and learn from data.
    What are the key connections between them?
    """

    print("Running agentic pipeline...")
    print(f"Question: {question}\n")

    response = await pipeline.query(
        question,
        collection="example",
        enable_reflection=True,
        max_iterations=3,
    )

    print(f"Answer:\n{response.response}")
    print(f"\n--- Metadata ---")
    print(f"Iterations: {response.metadata.get('iterations', 1)}")
    print(f"Confidence: {response.metadata.get('confidence', 'N/A')}")
    print(f"Sources: {len(response.sources)}")
    print(f"Latency: {response.latency_ms:.0f}ms")

    await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
