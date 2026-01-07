"""
Benchmark runner for RAG Optimizer.

Run standard QA benchmarks to evaluate pipeline performance.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from rag_optimizer.config import get_settings
from rag_optimizer.evaluation import RAGBenchmark, load_benchmark
from rag_optimizer.pipeline import PipelineBuilder


async def run_benchmark(
    benchmark_name: str,
    collection: str,
    n_samples: int = 100,
    output_dir: Path | None = None,
):
    """Run a benchmark and save results."""

    print(f"Loading benchmark: {benchmark_name}")
    questions = load_benchmark(benchmark_name, n=n_samples)
    print(f"Loaded {len(questions)} questions")

    # Build pipeline
    settings = get_settings()
    pipeline = (
        PipelineBuilder(settings=settings)
        .with_retrieval("hybrid", use_hyde=True)
        .with_evaluation(enable_ragas=True)
        .build()
    )

    # Run benchmark
    print("Running benchmark...")
    benchmark = RAGBenchmark(pipeline=pipeline)
    results = await benchmark.run(questions=questions, collection=collection)

    # Print results
    print("\n" + "=" * 50)
    print(f"BENCHMARK RESULTS: {benchmark_name}")
    print("=" * 50)
    print(f"Total Questions: {results.total_questions}")
    print(f"Avg Context Precision: {results.avg_context_precision:.3f}")
    print(f"Avg Context Recall: {results.avg_context_recall:.3f}")
    print(f"Avg Faithfulness: {results.avg_faithfulness:.3f}")
    print(f"Avg Answer Relevancy: {results.avg_answer_relevancy:.3f}")
    print(f"Avg Latency: {results.avg_latency_ms:.1f}ms")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{benchmark_name}_{datetime.now():%Y%m%d_%H%M%S}.json"

        with open(output_file, "w") as f:
            json.dump(
                {
                    "benchmark": benchmark_name,
                    "timestamp": datetime.now().isoformat(),
                    "n_samples": n_samples,
                    "results": {
                        "total_questions": results.total_questions,
                        "avg_context_precision": results.avg_context_precision,
                        "avg_context_recall": results.avg_context_recall,
                        "avg_faithfulness": results.avg_faithfulness,
                        "avg_answer_relevancy": results.avg_answer_relevancy,
                        "avg_latency_ms": results.avg_latency_ms,
                    },
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {output_file}")

    await pipeline.close()
    return results


async def main():
    """Run all benchmarks."""
    benchmarks = ["hotpotqa", "nq", "triviaqa"]
    collection = "benchmark_docs"
    output_dir = Path("benchmark_results")

    for benchmark_name in benchmarks:
        try:
            await run_benchmark(
                benchmark_name=benchmark_name,
                collection=collection,
                n_samples=50,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"Error running {benchmark_name}: {e}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
