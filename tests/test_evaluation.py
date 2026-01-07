"""
Comprehensive unit tests for evaluation functionality.

Tests:
- RAGAS evaluation metrics (Context Precision, Recall, Faithfulness, Answer Relevancy)
- Self-RAG evaluation with reflection
- NLI-based evaluation
- Benchmark datasets (HotPotQA, Natural Questions)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag.core.models import Chunk, GenerationResult, RetrievalResult
from agentic_rag.evaluation.benchmarks import (
    HotPotQA,
    NaturalQuestions,
    RAGBenchmark,
)
from agentic_rag.evaluation.nli import NLIVerifier
from agentic_rag.evaluation.ragas import RAGASEvaluator
from agentic_rag.evaluation.self_rag import SelfRAGEvaluator

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_retrieval_result() -> RetrievalResult:
    """Create sample retrieval result for evaluation."""
    return RetrievalResult(
        chunks=[
            Chunk(
                id="c1",
                content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                document_id="doc1",
            ),
            Chunk(
                id="c2",
                content="Deep learning uses neural networks with multiple layers to process complex patterns.",
                document_id="doc2",
            ),
            Chunk(
                id="c3",
                content="Natural language processing helps computers understand human language.",
                document_id="doc3",
            ),
        ],
        scores=[0.95, 0.85, 0.75],
        retrieval_type="dense",
    )


@pytest.fixture
def sample_generation_result() -> GenerationResult:
    """Create sample generation result for evaluation."""
    return GenerationResult(
        response="Machine learning is a type of AI that learns from data. It uses algorithms to identify patterns and make predictions.",
        sources=[
            Chunk(id="c1", content="ML is AI that learns from data.", document_id="doc1"),
        ],
        confidence=0.85,
        provider="mock",
        model="test-model",
    )


@pytest.fixture
def mock_generator_for_eval():
    """Create mock generator for evaluation."""
    generator = MagicMock()

    async def mock_generate_text(prompt, **kwargs):
        # Simulate evaluation responses
        if "faithfulness" in prompt.lower():
            return "Score: 0.9"
        elif "relevance" in prompt.lower():
            return "Score: 0.85"
        else:
            return "Evaluation complete"

    generator.generate_text = AsyncMock(side_effect=mock_generate_text)
    return generator


# =============================================================================
# RAGAS Evaluator Tests
# =============================================================================


class TestRAGASEvaluator:
    """Tests for the RAGAS evaluation framework."""

    @pytest.fixture
    def ragas_evaluator(self, mock_generator_for_eval, test_settings_minimal):
        """Create RAGAS evaluator with mock generator."""
        return RAGASEvaluator(
            generator=mock_generator_for_eval,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_evaluate_context_precision(self, ragas_evaluator, sample_retrieval_result):
        """Test context precision evaluation."""
        question = "What is machine learning?"
        ground_truth = "Machine learning is a subset of AI that learns from data."

        score = await ragas_evaluator.evaluate_context_precision(
            question=question,
            retrieval_result=sample_retrieval_result,
            ground_truth=ground_truth,
        )

        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_context_recall(self, ragas_evaluator, sample_retrieval_result):
        """Test context recall evaluation."""
        question = "What is machine learning?"
        ground_truth = "Machine learning is AI that learns from data."

        score = await ragas_evaluator.evaluate_context_recall(
            question=question,
            retrieval_result=sample_retrieval_result,
            ground_truth=ground_truth,
        )

        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness(
        self, ragas_evaluator, sample_generation_result, sample_retrieval_result
    ):
        """Test faithfulness evaluation."""
        score = await ragas_evaluator.evaluate_faithfulness(
            generation_result=sample_generation_result,
            retrieval_result=sample_retrieval_result,
        )

        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_answer_relevancy(self, ragas_evaluator, sample_generation_result):
        """Test answer relevancy evaluation."""
        question = "What is machine learning?"

        score = await ragas_evaluator.evaluate_answer_relevancy(
            question=question,
            generation_result=sample_generation_result,
        )

        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_full_ragas(
        self, ragas_evaluator, sample_generation_result, sample_retrieval_result
    ):
        """Test full RAGAS evaluation."""
        question = "What is machine learning?"
        ground_truth = "Machine learning is AI that learns from data."

        results = await ragas_evaluator.evaluate(
            question=question,
            retrieval_result=sample_retrieval_result,
            generation_result=sample_generation_result,
            ground_truth=ground_truth,
        )

        assert "context_precision" in results
        assert "context_recall" in results
        assert "faithfulness" in results
        assert "answer_relevancy" in results

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, ragas_evaluator):
        """Test batch evaluation."""
        samples = [
            {
                "question": "What is ML?",
                "contexts": ["ML is AI that learns."],
                "answer": "ML is machine learning.",
                "ground_truth": "Machine learning.",
            },
            {
                "question": "What is DL?",
                "contexts": ["DL uses neural networks."],
                "answer": "DL is deep learning.",
                "ground_truth": "Deep learning.",
            },
        ]

        results = await ragas_evaluator.evaluate_batch(samples)

        assert len(results) == 2


# =============================================================================
# Self-RAG Evaluator Tests
# =============================================================================


class TestSelfRAGEvaluator:
    """Tests for the Self-RAG evaluation framework."""

    @pytest.fixture
    def self_rag_evaluator(self, mock_generator_for_eval, test_settings_minimal):
        """Create Self-RAG evaluator."""
        return SelfRAGEvaluator(
            generator=mock_generator_for_eval,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_evaluate_retrieval_necessity(self, self_rag_evaluator):
        """Test retrieval necessity evaluation."""
        question = "What year did World War II end?"

        needs_retrieval = await self_rag_evaluator.evaluate_retrieval_necessity(
            question=question,
        )

        assert isinstance(needs_retrieval, bool)

    @pytest.mark.asyncio
    async def test_evaluate_relevance(self, self_rag_evaluator, sample_retrieval_result):
        """Test relevance evaluation."""
        question = "What is machine learning?"

        relevance_scores = await self_rag_evaluator.evaluate_relevance(
            question=question,
            retrieval_result=sample_retrieval_result,
        )

        assert len(relevance_scores) == len(sample_retrieval_result.chunks)
        assert all(0.0 <= s <= 1.0 for s in relevance_scores)

    @pytest.mark.asyncio
    async def test_evaluate_support(
        self, self_rag_evaluator, sample_generation_result, sample_retrieval_result
    ):
        """Test support evaluation."""
        support_score = await self_rag_evaluator.evaluate_support(
            generation_result=sample_generation_result,
            retrieval_result=sample_retrieval_result,
        )

        assert 0.0 <= support_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_utility(self, self_rag_evaluator, sample_generation_result):
        """Test utility evaluation."""
        question = "What is machine learning?"

        utility_score = await self_rag_evaluator.evaluate_utility(
            question=question,
            generation_result=sample_generation_result,
        )

        assert 0.0 <= utility_score <= 1.0

    @pytest.mark.asyncio
    async def test_full_self_rag_evaluation(
        self, self_rag_evaluator, sample_generation_result, sample_retrieval_result
    ):
        """Test full Self-RAG evaluation."""
        question = "What is machine learning?"

        results = await self_rag_evaluator.evaluate(
            question=question,
            retrieval_result=sample_retrieval_result,
            generation_result=sample_generation_result,
        )

        assert "needs_retrieval" in results or "relevance" in results
        assert "support" in results
        assert "utility" in results


# =============================================================================
# NLI Evaluator Tests
# =============================================================================


class TestNLIEvaluator:
    """Tests for the NLI-based evaluation."""

    @pytest.fixture
    def nli_evaluator(self, test_settings_minimal):
        """Create NLI evaluator."""
        with (
            patch("agentic_rag.evaluation.nli.AutoModelForSequenceClassification"),
            patch("agentic_rag.evaluation.nli.AutoTokenizer"),
        ):
            return NLIEvaluator(settings=test_settings_minimal)

    @pytest.mark.asyncio
    async def test_evaluate_entailment(self, nli_evaluator):
        """Test entailment evaluation."""
        premise = "Machine learning is a subset of AI."
        hypothesis = "ML is related to artificial intelligence."

        score = await nli_evaluator.evaluate_entailment(
            premise=premise,
            hypothesis=hypothesis,
        )

        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_contradiction(self, nli_evaluator):
        """Test contradiction detection."""
        premise = "Python is a programming language."
        hypothesis = "Python is an animal species."

        result = await nli_evaluator.evaluate_contradiction(
            premise=premise,
            hypothesis=hypothesis,
        )

        assert isinstance(result, (float, bool))

    @pytest.mark.asyncio
    async def test_evaluate_factual_consistency(
        self, nli_evaluator, sample_generation_result, sample_retrieval_result
    ):
        """Test factual consistency evaluation."""
        score = await nli_evaluator.evaluate_factual_consistency(
            generation_result=sample_generation_result,
            retrieval_result=sample_retrieval_result,
        )

        assert 0.0 <= score <= 1.0


# =============================================================================
# Benchmark Runner Tests
# =============================================================================


class TestBenchmarkRunner:
    """Tests for the benchmark runner."""

    @pytest.fixture
    def mock_pipeline_for_benchmark(self):
        """Create mock pipeline for benchmarks."""
        pipeline = MagicMock()

        async def mock_query(question, collection, **kwargs):
            return GenerationResult(
                response=f"Answer to: {question}",
                sources=[Chunk(id="c1", content="Source", document_id="d1")],
                confidence=0.8,
                provider="mock",
            )

        pipeline.query = AsyncMock(side_effect=mock_query)
        return pipeline

    @pytest.fixture
    def benchmark_runner(self, mock_pipeline_for_benchmark, test_settings_minimal):
        """Create benchmark runner."""
        return BenchmarkRunner(
            pipeline=mock_pipeline_for_benchmark,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_run_single_sample(self, benchmark_runner, hotpotqa_sample):
        """Test running single benchmark sample."""
        sample = hotpotqa_sample[0]

        result = await benchmark_runner.run_sample(
            question=sample["question"],
            ground_truth=sample["answer"],
            collection="test",
        )

        assert "response" in result
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_run_benchmark(self, benchmark_runner, hotpotqa_sample):
        """Test running full benchmark."""
        results = await benchmark_runner.run_benchmark(
            samples=hotpotqa_sample,
            collection="test",
        )

        assert "samples" in results
        assert "metrics" in results
        assert len(results["samples"]) == len(hotpotqa_sample)

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, benchmark_runner):
        """Test metrics calculation."""
        sample_results = [
            {"correct": True, "latency_ms": 100},
            {"correct": True, "latency_ms": 150},
            {"correct": False, "latency_ms": 200},
        ]

        metrics = benchmark_runner.calculate_metrics(sample_results)

        assert "accuracy" in metrics or "recall" in metrics
        assert "avg_latency_ms" in metrics or "latency" in metrics


# =============================================================================
# HotPotQA Benchmark Tests
# =============================================================================


class TestHotPotQABenchmark:
    """Tests for HotPotQA benchmark evaluation."""

    @pytest.fixture
    def hotpotqa_benchmark(self, test_settings_minimal):
        """Create HotPotQA benchmark."""
        return HotPotQABenchmark(settings=test_settings_minimal)

    def test_load_samples(self, hotpotqa_benchmark, hotpotqa_sample):
        """Test loading HotPotQA samples."""
        # Using fixture data
        assert len(hotpotqa_sample) > 0
        assert "question" in hotpotqa_sample[0]
        assert "answer" in hotpotqa_sample[0]

    def test_parse_sample(self, hotpotqa_benchmark, hotpotqa_sample):
        """Test parsing HotPotQA sample."""
        sample = hotpotqa_sample[0]
        parsed = hotpotqa_benchmark.parse_sample(sample)

        assert "question" in parsed
        assert "answer" in parsed
        assert "supporting_facts" in parsed or "context" in parsed

    def test_evaluate_answer(self, hotpotqa_benchmark):
        """Test answer evaluation."""
        # Exact match
        assert hotpotqa_benchmark.evaluate_answer("Python", "Python") == 1.0

        # Partial match
        score = hotpotqa_benchmark.evaluate_answer("Python language", "Python")
        assert 0.0 <= score <= 1.0

    def test_evaluate_supporting_facts(self, hotpotqa_benchmark):
        """Test supporting facts evaluation."""
        predicted_facts = [("Doc1", 0), ("Doc2", 1)]
        ground_truth_facts = [("Doc1", 0), ("Doc3", 2)]

        precision, recall, f1 = hotpotqa_benchmark.evaluate_supporting_facts(
            predicted=predicted_facts,
            ground_truth=ground_truth_facts,
        )

        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0


# =============================================================================
# Natural Questions Benchmark Tests
# =============================================================================


class TestNaturalQuestionsBenchmark:
    """Tests for Natural Questions benchmark evaluation."""

    @pytest.fixture
    def nq_benchmark(self, test_settings_minimal):
        """Create Natural Questions benchmark."""
        return NaturalQuestionsBenchmark(settings=test_settings_minimal)

    def test_load_samples(self, nq_benchmark, natural_questions_sample):
        """Test loading NQ samples."""
        assert len(natural_questions_sample) > 0
        assert "question" in natural_questions_sample[0]

    def test_parse_sample(self, nq_benchmark, natural_questions_sample):
        """Test parsing NQ sample."""
        sample = natural_questions_sample[0]
        parsed = nq_benchmark.parse_sample(sample)

        assert "question" in parsed
        assert "short_answer" in parsed or "long_answer" in parsed

    def test_evaluate_short_answer(self, nq_benchmark):
        """Test short answer evaluation."""
        # Exact match
        assert nq_benchmark.evaluate_short_answer("Python", ["Python"]) == 1.0

        # No match
        assert nq_benchmark.evaluate_short_answer("Java", ["Python"]) == 0.0

    def test_evaluate_long_answer(self, nq_benchmark):
        """Test long answer evaluation (token F1)."""
        predicted = "Python is a programming language"
        ground_truth = "Python is a high-level programming language"

        score = nq_benchmark.evaluate_long_answer(predicted, ground_truth)

        assert 0.0 <= score <= 1.0


# =============================================================================
# Evaluation Metrics Tests
# =============================================================================


class TestEvaluationMetrics:
    """Tests for evaluation metric calculations."""

    def test_exact_match(self):
        """Test exact match metric."""
        from agentic_rag.evaluation.base import exact_match

        assert exact_match("Python", "Python") == 1.0
        assert exact_match("Python", "python") == 1.0  # Case insensitive
        assert exact_match("Python", "Java") == 0.0

    def test_f1_score(self):
        """Test F1 score calculation."""
        from agentic_rag.evaluation.base import token_f1_score

        pred = "machine learning is great"
        truth = "machine learning is useful"

        score = token_f1_score(pred, truth)
        assert 0.0 <= score <= 1.0

    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        from agentic_rag.evaluation.base import recall_at_k

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc6"]

        recall_5 = recall_at_k(retrieved, relevant, k=5)
        recall_3 = recall_at_k(retrieved, relevant, k=3)

        assert recall_5 >= recall_3

    def test_mrr(self):
        """Test Mean Reciprocal Rank calculation."""
        from agentic_rag.evaluation.base import mean_reciprocal_rank

        # First result is relevant
        rankings1 = [["doc1", "doc2", "doc3"]]
        relevant1 = [["doc1"]]
        assert mean_reciprocal_rank(rankings1, relevant1) == 1.0

        # Third result is relevant
        rankings2 = [["doc1", "doc2", "doc3"]]
        relevant2 = [["doc3"]]
        assert mean_reciprocal_rank(rankings2, relevant2) == pytest.approx(1 / 3, abs=0.01)

    def test_ndcg(self):
        """Test NDCG calculation."""
        from agentic_rag.evaluation.base import ndcg_score

        relevance_scores = [3, 2, 3, 0, 1, 2]
        ndcg = ndcg_score(relevance_scores, k=5)

        assert 0.0 <= ndcg <= 1.0


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestEvaluationIntegration:
    """Integration tests for evaluation components."""

    @pytest.fixture
    def full_evaluator(self, mock_generator_for_eval, test_settings_minimal):
        """Create full evaluator stack."""
        return {
            "ragas": RAGASEvaluator(
                generator=mock_generator_for_eval, settings=test_settings_minimal
            ),
            "self_rag": SelfRAGEvaluator(
                generator=mock_generator_for_eval, settings=test_settings_minimal
            ),
        }

    @pytest.mark.asyncio
    async def test_combined_evaluation(
        self,
        full_evaluator,
        sample_generation_result,
        sample_retrieval_result,
    ):
        """Test combined evaluation from multiple frameworks."""
        question = "What is machine learning?"
        ground_truth = "Machine learning is AI that learns from data."

        ragas_results = await full_evaluator["ragas"].evaluate(
            question=question,
            retrieval_result=sample_retrieval_result,
            generation_result=sample_generation_result,
            ground_truth=ground_truth,
        )

        self_rag_results = await full_evaluator["self_rag"].evaluate(
            question=question,
            retrieval_result=sample_retrieval_result,
            generation_result=sample_generation_result,
        )

        # Both should complete
        assert len(ragas_results) > 0
        assert len(self_rag_results) > 0


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestEvaluationPerformance:
    """Performance tests for evaluation."""

    @pytest.fixture
    def fast_evaluator(self, mock_generator_for_eval, test_settings_minimal):
        """Create evaluator for performance testing."""
        return RAGASEvaluator(
            generator=mock_generator_for_eval,
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_batch_evaluation_speed(self, fast_evaluator):
        """Test batch evaluation speed."""
        import time

        samples = [
            {
                "question": f"Question {i}",
                "contexts": [f"Context {i}"],
                "answer": f"Answer {i}",
                "ground_truth": f"Truth {i}",
            }
            for i in range(10)
        ]

        start = time.time()
        await fast_evaluator.evaluate_batch(samples)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 30.0

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, fast_evaluator):
        """Test concurrent evaluation handling."""
        import time

        tasks = []
        for i in range(5):
            tasks.append(
                fast_evaluator.evaluate_context_precision(
                    question=f"Question {i}",
                    retrieval_result=RetrievalResult(
                        chunks=[Chunk(id=f"c{i}", content=f"Content {i}", document_id=f"d{i}")],
                        scores=[0.9],
                        retrieval_type="test",
                    ),
                    ground_truth=f"Truth {i}",
                )
            )

        start = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        # Concurrent should be efficient
        assert elapsed < 10.0
