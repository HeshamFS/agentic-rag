"""
RAGAS (Retrieval-Augmented Generation Assessment) evaluation metrics.

Implements the core RAGAS metrics:
- Context Precision: Are retrieved contexts relevant?
- Context Recall: Are all needed facts retrieved?
- Faithfulness: Is response grounded in context?
- Answer Relevancy: Does response address the query?
"""

import re

from agentic_rag.core.models import Chunk
from agentic_rag.core.protocols import Generator
from agentic_rag.evaluation.base import BaseEvaluator, EvaluationResult


class ContextPrecisionEvaluator(BaseEvaluator):
    """
    Context Precision: Measures the proportion of retrieved contexts
    that are relevant to the query.

    High precision = Few irrelevant contexts retrieved.
    """

    def __init__(self, generator: Generator):
        """
        Initialize context precision evaluator.

        Args:
            generator: LLM for relevance assessment.
        """
        super().__init__("context_precision")
        self._generator = generator

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate context precision.

        Context precision measures the signal-to-noise ratio of the retrieved
        chunks. It identifies how many of the top-k results are actually
        relevant to the user's query.

        Args:
            query: The user search query.
            response: The generated response (not used for this metric).
            contexts: List of retrieved context chunks to evaluate.
            ground_truth: Not used for this metric.

        Returns:
            EvaluationResult with the precision score (0.0 to 1.0).
        """
        if not contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                reasoning="No contexts to evaluate",
            )

        relevant_count = 0
        relevance_scores = []

        for _i, chunk in enumerate(contexts):
            is_relevant = await self._is_context_relevant(query, chunk.content)
            relevance_scores.append(is_relevant)
            if is_relevant:
                relevant_count += 1

        precision = relevant_count / len(contexts)

        return EvaluationResult(
            metric_name=self.name,
            score=precision,
            details={
                "relevant_count": relevant_count,
                "total_count": len(contexts),
                "relevance_per_chunk": relevance_scores,
            },
            reasoning=f"{relevant_count}/{len(contexts)} contexts are relevant to the query",
        )

    async def _is_context_relevant(self, query: str, context: str) -> bool:
        """Check if a context is relevant to the query."""
        prompt = f"""Is this context relevant to answering the query?

Query: "{query}"

Context: "{context[:500]}"

Answer only 'yes' or 'no'."""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=10,
        )

        return "yes" in response.lower()


class ContextRecallEvaluator(BaseEvaluator):
    """
    Context Recall: Measures how much of the ground truth
    can be attributed to the retrieved contexts.

    High recall = All needed information was retrieved.
    Requires ground truth answer.
    """

    def __init__(self, generator: Generator):
        """
        Initialize context recall evaluator.

        Args:
            generator: LLM for attribution assessment.
        """
        super().__init__("context_recall")
        self._generator = generator

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate context recall.

        Context recall measures whether the retrieved chunks contain all the
        necessary factual claims present in the ground truth answer. This
        metric is critical for ensuring the retriever captures complete information.

        Args:
            query: The user search query.
            response: The generated response (not used for this metric).
            contexts: List of retrieved context chunks.
            ground_truth: The reference correct answer (required).

        Returns:
            EvaluationResult with the recall score (0.0 to 1.0).
        """
        if not ground_truth:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                reasoning="Ground truth required for context recall",
            )

        if not contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                reasoning="No contexts to evaluate",
            )

        # Extract claims from ground truth
        claims = await self._extract_claims(ground_truth)

        if not claims:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                reasoning="No claims to verify in ground truth",
            )

        # Check how many claims are supported by contexts
        context_text = "\n\n".join(c.content for c in contexts)
        supported_claims = 0
        claim_results = []

        for claim in claims:
            is_supported = await self._is_claim_supported(claim, context_text)
            claim_results.append({"claim": claim, "supported": is_supported})
            if is_supported:
                supported_claims += 1

        recall = supported_claims / len(claims)

        return EvaluationResult(
            metric_name=self.name,
            score=recall,
            details={
                "supported_claims": supported_claims,
                "total_claims": len(claims),
                "claim_results": claim_results,
            },
            reasoning=f"{supported_claims}/{len(claims)} ground truth claims are supported by retrieved contexts",
        )

    async def _extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from text."""
        prompt = f"""Extract the key factual claims from this text.
Output each claim on a separate line.

Text: "{text}"

Claims:"""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.2,
            max_tokens=512,
        )

        claims = []
        for line in response.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line:
                claims.append(line)

        return claims

    async def _is_claim_supported(self, claim: str, context: str) -> bool:
        """Check if a claim is supported by the context."""
        prompt = f"""Is this claim supported by the context?

Claim: "{claim}"

Context: "{context[:2000]}"

Answer only 'yes' or 'no'."""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=10,
        )

        return "yes" in response.lower()


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Faithfulness: Measures how much of the response
    is grounded in the retrieved contexts.

    High faithfulness = No hallucinations, all claims supported.
    """

    def __init__(self, generator: Generator):
        """
        Initialize faithfulness evaluator.

        Args:
            generator: LLM for claim verification.
        """
        super().__init__("faithfulness")
        self._generator = generator

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate response faithfulness.

        Args:
            query: User query.
            response: Generated response to verify.
            contexts: Retrieved contexts.
            ground_truth: Not used for this metric.

        Returns:
            Faithfulness score.
        """
        if not response:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                reasoning="Empty response",
            )

        if not contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                reasoning="No context to verify against",
            )

        # Extract claims from response
        claims = await self._extract_claims(response)

        if not claims:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                reasoning="No verifiable claims in response",
            )

        # Verify each claim against contexts
        context_text = "\n\n".join(c.content for c in contexts)
        supported_claims = 0
        hallucinations = []
        claim_results = []

        for claim in claims:
            verdict = await self._verify_claim(claim, context_text)
            claim_results.append(
                {
                    "claim": claim,
                    "verdict": verdict,
                }
            )

            if verdict == "supported":
                supported_claims += 1
            elif verdict == "contradicted":
                hallucinations.append(claim)

        faithfulness = supported_claims / len(claims)

        return EvaluationResult(
            metric_name=self.name,
            score=faithfulness,
            details={
                "supported_claims": supported_claims,
                "total_claims": len(claims),
                "hallucinations": hallucinations,
                "claim_results": claim_results,
            },
            reasoning=f"{supported_claims}/{len(claims)} response claims are supported by context. "
            f"Found {len(hallucinations)} potential hallucinations.",
        )

    async def _extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from response."""
        prompt = f"""Extract all factual claims from this response.
Each claim should be a single, verifiable statement.
Output one claim per line.

Response: "{text}"

Claims:"""

        result = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.2,
            max_tokens=512,
        )

        claims = []
        for line in result.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line:
                claims.append(line)

        return claims

    async def _verify_claim(self, claim: str, context: str) -> str:
        """Verify if a claim is supported, contradicted, or neutral."""
        prompt = f"""Verify this claim against the context.

Claim: "{claim}"

Context: "{context[:2000]}"

Is the claim:
- SUPPORTED: The context contains information that supports this claim
- CONTRADICTED: The context contains information that contradicts this claim
- NEUTRAL: The context neither supports nor contradicts this claim

Answer with one word: SUPPORTED, CONTRADICTED, or NEUTRAL."""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=20,
        )

        response_lower = response.lower()
        if "supported" in response_lower:
            return "supported"
        elif "contradicted" in response_lower:
            return "contradicted"
        else:
            return "neutral"


class AnswerRelevancyEvaluator(BaseEvaluator):
    """
    Answer Relevancy: Measures how well the response
    addresses the original query.

    High relevancy = Response directly answers the question.
    """

    def __init__(self, generator: Generator):
        """
        Initialize answer relevancy evaluator.

        Args:
            generator: LLM for relevancy assessment.
        """
        super().__init__("answer_relevancy")
        self._generator = generator

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate answer relevancy.

        Args:
            query: User query.
            response: Generated response.
            contexts: Not used for this metric.
            ground_truth: Not used for this metric.

        Returns:
            Answer relevancy score.
        """
        if not response:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                reasoning="Empty response",
            )

        # Method: Generate questions that the response would answer,
        # then compare with original query
        prompt = f"""Given this response, what question(s) is it answering?

Response: "{response}"

List the main question(s) this response addresses, one per line:"""

        result = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=256,
        )

        generated_questions = []
        for line in result.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line:
                generated_questions.append(line)

        if not generated_questions:
            return EvaluationResult(
                metric_name=self.name,
                score=0.5,
                reasoning="Could not extract questions from response",
            )

        # Calculate semantic similarity between original query and generated questions
        max_similarity = 0.0
        best_match = ""

        for gen_q in generated_questions:
            similarity = await self._calculate_similarity(query, gen_q)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = gen_q

        return EvaluationResult(
            metric_name=self.name,
            score=max_similarity,
            details={
                "original_query": query,
                "generated_questions": generated_questions,
                "best_match": best_match,
            },
            reasoning=f"Response addresses: '{best_match}' (similarity: {max_similarity:.2f})",
        )

    async def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between two queries."""
        prompt = f"""Rate the semantic similarity between these two questions from 0.0 to 1.0.
1.0 means they ask exactly the same thing.
0.0 means they are completely unrelated.

Question 1: "{query1}"
Question 2: "{query2}"

Output only a number between 0.0 and 1.0:"""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=10,
        )

        try:
            # Extract number from response
            match = re.search(r"(\d+\.?\d*)", response)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
        except (ValueError, AttributeError):
            pass

        return 0.5  # Default


class RAGASEvaluator:
    """
    Complete RAGAS evaluation suite.

    Combines all four core RAGAS metrics:
    - Context Precision
    - Context Recall
    - Faithfulness
    - Answer Relevancy
    """

    def __init__(self, generator: Generator):
        """
        Initialize RAGAS evaluator.

        Args:
            generator: LLM for all evaluations.
        """
        self._generator = generator
        self.evaluators = [
            ContextPrecisionEvaluator(generator),
            ContextRecallEvaluator(generator),
            FaithfulnessEvaluator(generator),
            AnswerRelevancyEvaluator(generator),
        ]

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> dict[str, EvaluationResult]:
        """
        Run all RAGAS evaluations.

        Args:
            query: User query.
            response: Generated response.
            contexts: Retrieved contexts.
            ground_truth: Optional ground truth.

        Returns:
            Dict of metric name to evaluation result.
        """
        results = {}

        for evaluator in self.evaluators:
            result = await evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts,
                ground_truth=ground_truth,
            )
            results[result.metric_name] = result

        return results

    def aggregate_scores(
        self,
        results: dict[str, EvaluationResult],
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Calculate weighted aggregate score.

        Args:
            results: Individual metric results.
            weights: Optional weights per metric.

        Returns:
            Aggregate score.
        """
        if weights is None:
            weights = {
                "context_precision": 0.2,
                "context_recall": 0.2,
                "faithfulness": 0.3,
                "answer_relevancy": 0.3,
            }

        total_weight = 0.0
        weighted_sum = 0.0

        for metric_name, result in results.items():
            weight = weights.get(metric_name, 0.25)
            weighted_sum += result.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0
