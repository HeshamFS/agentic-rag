"""
RAG Optimizer Benchmarks.

This package provides benchmark runners for evaluating RAG performance
against standard datasets:
- HotPotQA: Multi-hop reasoning over Wikipedia
- Natural Questions: Google search queries with Wikipedia answers
- Custom RAGAS evaluation suite

Usage:
    python -m benchmarks.run_benchmarks --dataset hotpotqa --samples 100
"""

from benchmarks.run_benchmarks import (
    run_benchmark,
    generate_hotpotqa_samples,
    generate_natural_questions_samples,
)

__all__ = [
    "run_benchmark",
    "generate_hotpotqa_samples",
    "generate_natural_questions_samples",
]
