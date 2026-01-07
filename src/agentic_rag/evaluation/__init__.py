"""Evaluation components including RAGAS metrics, Self-RAG reflection, and NLI verification."""

from agentic_rag.evaluation.base import (
    BaseEvaluator,
    CompositeEvaluator,
    EvaluationResult,
    EvaluationSuite,
    calculate_f1,
    calculate_overlap,
)
from agentic_rag.evaluation.benchmarks import (
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkSummary,
    HotPotQA,
    NaturalQuestions,
    RAGBenchmark,
    TriviaQA,
    load_benchmark,
)
from agentic_rag.evaluation.nli import (
    BatchNLIVerifier,
    ClaimVerification,
    HallucinationDetector,
    NLILabel,
    NLIVerifier,
    NLIVerifierOutput,
)
from agentic_rag.evaluation.ragas import (
    AnswerRelevancyEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    FaithfulnessEvaluator,
    RAGASEvaluator,
)
from agentic_rag.evaluation.self_rag import (
    IsRelEvaluator,
    IsSupEvaluator,
    IsUseEvaluator,
    ReflectionValue,
    SelfRAGEvaluator,
    SelfRAGOutput,
)

__all__ = [
    # Base
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationSuite",
    "CompositeEvaluator",
    "calculate_f1",
    "calculate_overlap",
    # RAGAS
    "RAGASEvaluator",
    "ContextPrecisionEvaluator",
    "ContextRecallEvaluator",
    "FaithfulnessEvaluator",
    "AnswerRelevancyEvaluator",
    # Self-RAG
    "SelfRAGEvaluator",
    "SelfRAGOutput",
    "IsRelEvaluator",
    "IsSupEvaluator",
    "IsUseEvaluator",
    "ReflectionValue",
    # NLI
    "NLIVerifier",
    "NLIVerifierOutput",
    "NLILabel",
    "ClaimVerification",
    "BatchNLIVerifier",
    "HallucinationDetector",
    # Benchmarks
    "RAGBenchmark",
    "BenchmarkQuestion",
    "BenchmarkResult",
    "BenchmarkSummary",
    "HotPotQA",
    "NaturalQuestions",
    "TriviaQA",
    "load_benchmark",
]
