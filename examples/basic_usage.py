"""
Basic usage example for RAG Optimizer.

This example demonstrates:
1. Setting up the pipeline
2. Ingesting documents
3. Querying the pipeline
"""

import asyncio
from agentic_rag.core.models import Document
from agentic_rag.pipeline import PipelineBuilder
from agentic_rag.config import get_settings


async def main():
    """Run basic RAG pipeline example."""

    # Initialize settings
    settings = get_settings()

    # Build pipeline with fluent API
    pipeline = (
        PipelineBuilder(settings=settings)
        .with_chunking("semantic", chunk_size=512)
        .with_retrieval("hybrid", use_hyde=True)
        .with_generator(provider="claude", model="claude-sonnet-4-20250514")
        .build()
    )

    # Sample documents
    documents = [
        Document(
            content="""
            Python is a high-level, interpreted programming language known for its
            clear syntax and readability. It supports multiple programming paradigms,
            including procedural, object-oriented, and functional programming.
            """,
            metadata={"source": "python_intro.txt"},
        ),
        Document(
            content="""
            Machine learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience without being explicitly
            programmed. It focuses on developing algorithms that can access data
            and use it to learn for themselves.
            """,
            metadata={"source": "ml_intro.txt"},
        ),
    ]

    # Ingest documents
    print("Ingesting documents...")
    result = await pipeline.ingest(documents, collection="example")
    print(f"Ingested {result.documents_count} documents, {result.chunks_created} chunks")

    # Query the pipeline
    print("\nQuerying pipeline...")
    response = await pipeline.query(
        "What is Python and how does it relate to machine learning?",
        collection="example",
    )

    print(f"\nAnswer: {response.response}")
    print(f"\nSources: {len(response.sources)} chunks used")
    print(f"Latency: {response.latency_ms:.0f}ms")

    # Cleanup
    await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
