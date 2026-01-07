"""
Self-RAG reflection token evaluation.

Implements the Self-RAG reflection tokens:
- ISREL: Is the retrieved context relevant?
- ISSUP: Is the response supported by the context?
- ISUSE: Is the response useful to the user?

Based on the paper: "Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection"
"""

from enum import Enum

from pydantic import BaseModel, Field

from agentic_rag.core.models import Chunk, ReflectionToken
from agentic_rag.core.protocols import Generator
from agentic_rag.evaluation.base import BaseEvaluator, EvaluationResult


class ReflectionValue(str, Enum):
    """Possible values for reflection tokens."""

    FULLY = "fully"
    PARTIALLY = "partially"
    NOT = "not"


class SelfRAGOutput(BaseModel):
    """Complete Self-RAG evaluation output."""

    isrel: ReflectionValue = Field(description="Context relevance")
    issup: ReflectionValue = Field(description="Response support")
    isuse: ReflectionValue = Field(description="Response usefulness")

    isrel_score: float = Field(ge=0.0, le=1.0, description="Numeric relevance score")
    issup_score: float = Field(ge=0.0, le=1.0, description="Numeric support score")
    isuse_score: float = Field(ge=0.0, le=1.0, description="Numeric usefulness score")

    overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    should_regenerate: bool = Field(description="Whether to regenerate response")

    reflection_tokens: list[ReflectionToken] = Field(
        default_factory=list, description="Structured reflection tokens"
    )


class IsRelEvaluator(BaseEvaluator):
    """
    ISREL: Is the retrieved context relevant to the query?

    Evaluates whether each piece of retrieved context is relevant
    to answering the user's query.
    """

    def __init__(self, generator: Generator):
        """
        Initialize ISREL evaluator.

        Args:
            generator: LLM for relevance assessment.
        """
        super().__init__("isrel")
        self._generator = generator

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate context relevance.

        Args:
            query: User query.
            response: Not used for ISREL.
            contexts: Retrieved contexts to evaluate.
            ground_truth: Not used.

        Returns:
            ISREL evaluation result.
        """
        if not contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                details={"value": ReflectionValue.NOT.value},
                reasoning="No contexts retrieved",
            )

        context_text = "\n\n---\n\n".join(c.content[:500] for c in contexts[:5])

        prompt = f"""Evaluate if the retrieved context is relevant to answering this query.

Query: "{query}"

Retrieved Context:
{context_text}

Rate the relevance:
- FULLY: Context directly addresses the query with specific, useful information
- PARTIALLY: Context has some relevant information but is incomplete
- NOT: Context is unrelated or doesn't help answer the query

Also provide a score from 0.0 to 1.0.

Output format:
VERDICT: [FULLY/PARTIALLY/NOT]
SCORE: [0.0-1.0]
REASONING: [brief explanation]"""

        result = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.1,
            max_tokens=200,
        )

        # Parse result
        verdict = ReflectionValue.PARTIALLY
        score = 0.5
        reasoning = ""

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                val = line.split(":", 1)[1].strip().lower()
                if "fully" in val:
                    verdict = ReflectionValue.FULLY
                    score = 0.9 if score == 0.5 else score
                elif "not" in val:
                    verdict = ReflectionValue.NOT
                    score = 0.1 if score == 0.5 else score
            elif line.startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    score = min(max(score, 0.0), 1.0)
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            details={
                "value": verdict.value,
                "num_contexts": len(contexts),
            },
            reasoning=reasoning,
        )


class IsSupEvaluator(BaseEvaluator):
    """
    ISSUP: Is the response supported by the context?

    Evaluates whether claims in the response are grounded
    in the retrieved context (no hallucinations).
    """

    def __init__(self, generator: Generator):
        """
        Initialize ISSUP evaluator.

        Args:
            generator: LLM for support assessment.
        """
        super().__init__("issup")
        self._generator = generator

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate response support.

        Args:
            query: Not used for ISSUP.
            response: Response to verify.
            contexts: Context to verify against.
            ground_truth: Not used.

        Returns:
            ISSUP evaluation result.
        """
        if not response:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                details={"value": ReflectionValue.NOT.value},
                reasoning="Empty response",
            )

        if not contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                details={"value": ReflectionValue.NOT.value},
                reasoning="No context to verify against",
            )

        context_text = "\n\n---\n\n".join(c.content[:500] for c in contexts[:5])

        prompt = f"""Evaluate if this response is supported by the given context.
Check for hallucinations or unsupported claims.

Response:
{response[:1000]}

Context:
{context_text}

Rate the support:
- FULLY: Every claim in the response is supported by the context
- PARTIALLY: Some claims are supported, some are not
- NOT: Response contains significant unsupported claims or contradicts context

Output format:
VERDICT: [FULLY/PARTIALLY/NOT]
SCORE: [0.0-1.0]
UNSUPPORTED: [list any unsupported claims, or "none"]
REASONING: [brief explanation]"""

        result = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.1,
            max_tokens=300,
        )

        # Parse result
        verdict = ReflectionValue.PARTIALLY
        score = 0.5
        unsupported = []
        reasoning = ""

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                val = line.split(":", 1)[1].strip().lower()
                if "fully" in val:
                    verdict = ReflectionValue.FULLY
                    score = 0.95 if score == 0.5 else score
                elif "not" in val:
                    verdict = ReflectionValue.NOT
                    score = 0.1 if score == 0.5 else score
            elif line.startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    score = min(max(score, 0.0), 1.0)
                except ValueError:
                    pass
            elif line.startswith("UNSUPPORTED:"):
                unsupported_text = line.split(":", 1)[1].strip()
                if unsupported_text.lower() != "none":
                    unsupported = [unsupported_text]
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            details={
                "value": verdict.value,
                "unsupported_claims": unsupported,
            },
            reasoning=reasoning,
        )


class IsUseEvaluator(BaseEvaluator):
    """
    ISUSE: Is the response useful to the user?

    Evaluates whether the response actually helps answer
    the user's query in a clear, actionable way.
    """

    def __init__(self, generator: Generator):
        """
        Initialize ISUSE evaluator.

        Args:
            generator: LLM for usefulness assessment.
        """
        super().__init__("isuse")
        self._generator = generator

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate response usefulness.

        Args:
            query: User query.
            response: Response to evaluate.
            contexts: Not used for ISUSE.
            ground_truth: Not used.

        Returns:
            ISUSE evaluation result.
        """
        if not response:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                details={"value": ReflectionValue.NOT.value},
                reasoning="Empty response",
            )

        prompt = f"""Evaluate if this response is useful for answering the user's query.

Query: "{query}"

Response:
{response[:1000]}

Rate the usefulness:
- FULLY: Response completely and clearly answers the query
- PARTIALLY: Response addresses the query but is incomplete or unclear
- NOT: Response doesn't help answer the query or is confusing

Consider:
- Does it directly address what was asked?
- Is it clear and understandable?
- Is the information actionable/complete?

Output format:
VERDICT: [FULLY/PARTIALLY/NOT]
SCORE: [0.0-1.0]
MISSING: [what's missing or could be improved, or "nothing"]
REASONING: [brief explanation]"""

        result = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.1,
            max_tokens=250,
        )

        # Parse result
        verdict = ReflectionValue.PARTIALLY
        score = 0.5
        missing = ""
        reasoning = ""

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                val = line.split(":", 1)[1].strip().lower()
                if "fully" in val:
                    verdict = ReflectionValue.FULLY
                    score = 0.95 if score == 0.5 else score
                elif "not" in val:
                    verdict = ReflectionValue.NOT
                    score = 0.1 if score == 0.5 else score
            elif line.startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    score = min(max(score, 0.0), 1.0)
                except ValueError:
                    pass
            elif line.startswith("MISSING:"):
                missing = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            details={
                "value": verdict.value,
                "missing": missing if missing.lower() != "nothing" else None,
            },
            reasoning=reasoning,
        )


class SelfRAGEvaluator:
    """
    Complete Self-RAG evaluation combining ISREL, ISSUP, and ISUSE.

    Provides comprehensive quality assessment with actionable feedback.
    """

    def __init__(
        self,
        generator: Generator,
        regenerate_threshold: float = 0.5,
    ):
        """
        Initialize Self-RAG evaluator.

        Args:
            generator: LLM for all evaluations.
            regenerate_threshold: Score below which to suggest regeneration.
        """
        self._generator = generator
        self.regenerate_threshold = regenerate_threshold

        self._isrel = IsRelEvaluator(generator)
        self._issup = IsSupEvaluator(generator)
        self._isuse = IsUseEvaluator(generator)

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
    ) -> SelfRAGOutput:
        """
        Perform a full Self-RAG evaluation of a response.

        Args:
            query: The user's search query.
            response: The generated response to evaluate.
            contexts: The retrieved context chunks used for generation.

        Returns:
            SelfRAGOutput containing three-dimensional reflection tokens (ISREL, ISSUP, ISUSE)
            and a recommendation on whether to regenerate.
        """
        # Run all evaluations
        isrel_result = await self._isrel.evaluate(query, response, contexts)
        issup_result = await self._issup.evaluate(query, response, contexts)
        isuse_result = await self._isuse.evaluate(query, response, contexts)

        # Calculate overall score
        overall = (isrel_result.score + issup_result.score + isuse_result.score) / 3

        # Create reflection tokens
        tokens = [
            ReflectionToken(
                token_type="ISREL",
                value=isrel_result.details.get("value", "partially"),
                score=isrel_result.score,
            ),
            ReflectionToken(
                token_type="ISSUP",
                value=issup_result.details.get("value", "partially"),
                score=issup_result.score,
            ),
            ReflectionToken(
                token_type="ISUSE",
                value=isuse_result.details.get("value", "partially"),
                score=isuse_result.score,
            ),
        ]

        return SelfRAGOutput(
            isrel=ReflectionValue(isrel_result.details.get("value", "partially")),
            issup=ReflectionValue(issup_result.details.get("value", "partially")),
            isuse=ReflectionValue(isuse_result.details.get("value", "partially")),
            isrel_score=isrel_result.score,
            issup_score=issup_result.score,
            isuse_score=isuse_result.score,
            overall_score=overall,
            should_regenerate=overall < self.regenerate_threshold,
            reflection_tokens=tokens,
        )

    def create_feedback(self, output: SelfRAGOutput) -> str:
        """
        Create human-readable feedback from evaluation.

        Args:
            output: Self-RAG evaluation output.

        Returns:
            Feedback string.
        """
        feedback_parts = []

        if output.isrel == ReflectionValue.NOT:
            feedback_parts.append("Retrieved context is not relevant to the query.")
        elif output.isrel == ReflectionValue.PARTIALLY:
            feedback_parts.append("Retrieved context is only partially relevant.")

        if output.issup == ReflectionValue.NOT:
            feedback_parts.append("Response contains unsupported claims (hallucinations).")
        elif output.issup == ReflectionValue.PARTIALLY:
            feedback_parts.append("Some claims in the response lack context support.")

        if output.isuse == ReflectionValue.NOT:
            feedback_parts.append("Response doesn't effectively answer the query.")
        elif output.isuse == ReflectionValue.PARTIALLY:
            feedback_parts.append("Response could be more complete or clearer.")

        if not feedback_parts:
            return "Response quality is good."

        return " ".join(feedback_parts)
