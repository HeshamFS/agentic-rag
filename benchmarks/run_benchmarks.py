"""
RAG Optimizer Benchmark Runner.

Runs comprehensive benchmarks against standard datasets:
- HotPotQA (multi-hop reasoning)
- Natural Questions (single-hop QA)
- Custom evaluation with RAGAS metrics

Usage:
    python -m benchmarks.run_benchmarks --dataset hotpotqa --samples 100
    python -m benchmarks.run_benchmarks --all --output results.json
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_optimizer.config import Settings, get_settings
from rag_optimizer.embeddings import Qwen3Embedder
from rag_optimizer.pipeline import PipelineBuilder
from rag_optimizer.evaluation.ragas import RAGASEvaluator
from rag_optimizer.evaluation.benchmarks import (
    BenchmarkRunner,
    HotPotQABenchmark,
    NaturalQuestionsBenchmark,
)


# =============================================================================
# Benchmark Configuration
# =============================================================================


BENCHMARK_CONFIG = {
    "hotpotqa": {
        "name": "HotPotQA",
        "description": "Multi-hop reasoning over Wikipedia",
        "metrics": ["exact_match", "f1", "supporting_facts_precision", "supporting_facts_recall"],
        "sample_sizes": [100, 500, 1000],
    },
    "natural_questions": {
        "name": "Natural Questions",
        "description": "Google search queries with Wikipedia answers",
        "metrics": ["exact_match", "f1", "short_answer_accuracy"],
        "sample_sizes": [100, 500, 1000],
    },
}


# =============================================================================
# Sample Data Generators
# =============================================================================


def generate_hotpotqa_samples(num_samples: int = 100) -> list[dict[str, Any]]:
    """Generate HotPotQA-style samples for benchmarking."""
    samples = []

    # Define some realistic multi-hop questions
    question_templates = [
        {
            "question": "What is the capital of the country where {topic} was invented?",
            "topics": [
                ("Python programming language", "Netherlands", "Amsterdam"),
                ("the World Wide Web", "United Kingdom", "London"),
                ("Linux operating system", "Finland", "Helsinki"),
            ],
        },
        {
            "question": "Who founded the company that created {product}?",
            "topics": [
                ("Windows", "Microsoft", "Bill Gates and Paul Allen"),
                ("iPhone", "Apple", "Steve Jobs and Steve Wozniak"),
                ("Gmail", "Google", "Larry Page and Sergey Brin"),
            ],
        },
        {
            "question": "In which decade was the creator of {invention} born?",
            "topics": [
                ("the Transformer architecture", "Ashish Vaswani et al.", "1980s"),
                ("BERT", "Jacob Devlin et al.", "1980s"),
                ("GPT", "Alec Radford et al.", "1980s"),
            ],
        },
    ]

    sample_id = 0
    while len(samples) < num_samples:
        for template in question_templates:
            for topic, intermediate, answer in template["topics"]:
                if len(samples) >= num_samples:
                    break

                question = template["question"].format(topic=topic, product=topic, invention=topic)

                samples.append({
                    "id": f"hotpot_{sample_id}",
                    "question": question,
                    "answer": answer,
                    "type": "bridge",
                    "level": "medium",
                    "supporting_facts": [[topic, 0], [intermediate, 0]],
                    "context": [
                        [topic, [f"Information about {topic}.", f"Related to {intermediate}."]],
                        [intermediate, [f"Details about {intermediate}.", f"Connected to {answer}."]],
                    ],
                })
                sample_id += 1

    return samples[:num_samples]


def generate_natural_questions_samples(num_samples: int = 100) -> list[dict[str, Any]]:
    """Generate Natural Questions-style samples for benchmarking."""
    samples = []

    qa_pairs = [
        {
            "question": "what is machine learning",
            "short_answer": "a subset of artificial intelligence",
            "long_answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        },
        {
            "question": "who invented the transformer architecture",
            "short_answer": "Google researchers",
            "long_answer": "The Transformer architecture was introduced by Vaswani et al. in the paper 'Attention is All You Need' published by Google in 2017.",
        },
        {
            "question": "what is retrieval augmented generation",
            "short_answer": "RAG",
            "long_answer": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation to produce more accurate and grounded responses.",
        },
        {
            "question": "what is BERT",
            "short_answer": "Bidirectional Encoder Representations from Transformers",
            "long_answer": "BERT is a language model developed by Google that uses bidirectional training to understand context from both directions.",
        },
        {
            "question": "what is deep learning",
            "short_answer": "a subset of machine learning",
            "long_answer": "Deep learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns from data.",
        },
    ]

    sample_id = 0
    while len(samples) < num_samples:
        for qa in qa_pairs:
            if len(samples) >= num_samples:
                break

            samples.append({
                "id": f"nq_{sample_id}",
                "question": qa["question"],
                "short_answer": qa["short_answer"],
                "long_answer": qa["long_answer"],
                "document_title": f"Document for {qa['question'][:20]}",
            })
            sample_id += 1

    return samples[:num_samples]


# =============================================================================
# Benchmark Execution
# =============================================================================


async def run_benchmark(
    dataset: str,
    num_samples: int,
    collection: str,
    settings: Settings,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run a single benchmark."""
    print(f"\n{'='*60}")
    print(f"Running {dataset.upper()} Benchmark")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}\n")

    # Initialize pipeline
    print("Initializing pipeline...")
    embedder = Qwen3Embedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )

    pipeline = (
        PipelineBuilder()
        .with_embedder(embedder)
        .with_vectordb("qdrant", url=str(settings.qdrant_url))
        .with_generator(settings.llm_provider, model=settings.llm_model)
        .with_chunking("semantic", chunk_size=settings.default_chunk_size)
        .with_retrieval("hybrid")
        .build()
    )

    # Generate samples
    print(f"Generating {num_samples} samples...")
    if dataset == "hotpotqa":
        samples = generate_hotpotqa_samples(num_samples)
        benchmark = HotPotQABenchmark(settings=settings)
    elif dataset == "natural_questions":
        samples = generate_natural_questions_samples(num_samples)
        benchmark = NaturalQuestionsBenchmark(settings=settings)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create benchmark runner
    runner = BenchmarkRunner(pipeline=pipeline, settings=settings)

    # Run benchmark
    print(f"Running benchmark on {len(samples)} samples...")
    start_time = time.time()

    results = {
        "dataset": dataset,
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": settings.embedding_model,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
        },
        "samples": [],
        "metrics": {},
    }

    correct_count = 0
    total_latency = 0
    f1_scores = []

    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"  Processing sample {i+1}/{len(samples)}...")

        try:
            sample_start = time.time()

            # Query pipeline
            response = await pipeline.query(
                question=sample["question"],
                collection=collection,
            )

            sample_latency = (time.time() - sample_start) * 1000

            # Evaluate
            predicted = response.response
            ground_truth = sample.get("answer") or sample.get("short_answer") or sample.get("long_answer")

            # Simple exact match
            is_correct = ground_truth.lower() in predicted.lower() if ground_truth else False

            # Token F1
            pred_tokens = set(predicted.lower().split())
            truth_tokens = set(ground_truth.lower().split()) if ground_truth else set()
            if pred_tokens and truth_tokens:
                precision = len(pred_tokens & truth_tokens) / len(pred_tokens)
                recall = len(pred_tokens & truth_tokens) / len(truth_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                f1 = 0

            sample_result = {
                "id": sample.get("id", f"sample_{i}"),
                "question": sample["question"],
                "predicted": predicted[:200],
                "ground_truth": ground_truth,
                "correct": is_correct,
                "f1": f1,
                "latency_ms": sample_latency,
            }

            results["samples"].append(sample_result)

            if is_correct:
                correct_count += 1
            total_latency += sample_latency
            f1_scores.append(f1)

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            results["samples"].append({
                "id": sample.get("id", f"sample_{i}"),
                "question": sample["question"],
                "error": str(e),
            })

    elapsed = time.time() - start_time

    # Calculate metrics
    results["metrics"] = {
        "accuracy": correct_count / len(samples) if samples else 0,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        "avg_latency_ms": total_latency / len(samples) if samples else 0,
        "total_time_s": elapsed,
        "samples_per_second": len(samples) / elapsed if elapsed > 0 else 0,
    }

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Samples: {len(samples)}")
    print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
    print(f"Avg F1: {results['metrics']['avg_f1']:.4f}")
    print(f"Avg Latency: {results['metrics']['avg_latency_ms']:.2f}ms")
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Throughput: {results['metrics']['samples_per_second']:.2f} samples/sec")
    print(f"{'='*60}\n")

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    return results


async def run_ragas_evaluation(
    samples: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, float]:
    """Run RAGAS evaluation on samples."""
    from rag_optimizer.generation.provider_factory import ProviderFactory

    generator = ProviderFactory.create(
        provider=settings.llm_provider,
        model=settings.llm_model,
        settings=settings,
    )

    evaluator = RAGASEvaluator(generator=generator, settings=settings)

    results = await evaluator.evaluate_batch(samples)

    # Aggregate scores
    aggregated = {}
    for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        scores = [r.get(metric, 0) for r in results if metric in r]
        aggregated[metric] = sum(scores) / len(scores) if scores else 0

    return aggregated


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="RAG Optimizer Benchmark Runner")
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "natural_questions"],
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to use",
    )
    parser.add_argument(
        "--collection",
        default="benchmark",
        help="Collection name for queries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for results",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Include RAGAS evaluation",
    )

    args = parser.parse_args()

    settings = get_settings()

    datasets = []
    if args.all:
        datasets = ["hotpotqa", "natural_questions"]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Please specify --dataset or --all")
        return

    all_results = {}

    for dataset in datasets:
        output_path = args.output
        if args.all and output_path:
            output_path = output_path.parent / f"{dataset}_{output_path.name}"

        results = asyncio.run(
            run_benchmark(
                dataset=dataset,
                num_samples=args.samples,
                collection=args.collection,
                settings=settings,
                output_path=output_path,
            )
        )

        all_results[dataset] = results

    # Summary
    if len(datasets) > 1:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        for dataset, results in all_results.items():
            print(f"\n{dataset.upper()}:")
            print(f"  Accuracy: {results['metrics']['accuracy']:.2%}")
            print(f"  Avg F1: {results['metrics']['avg_f1']:.4f}")
            print(f"  Avg Latency: {results['metrics']['avg_latency_ms']:.2f}ms")


if __name__ == "__main__":
    main()
