"""
Evaluator Agent implementing Self-RAG reflection.

Evaluates generated responses using reflection tokens:
- ISREL: Is the retrieved context relevant?
- ISSUP: Is the response supported by the context?
- ISUSE: Is the response useful to the user?
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.agents.base import AgentState, BaseAgent
from agentic_rag.config import Settings
from agentic_rag.core.models import Chunk, ReflectionToken
from agentic_rag.core.protocols import Generator


class ReflectionScore(str, Enum):
    """Scores for reflection tokens."""

    FULLY = "fully"  # Fully relevant/supported/useful
    PARTIALLY = "partially"  # Partially relevant/supported/useful
    NOT = "not"  # Not relevant/supported/useful


class EvaluatorOutput(BaseModel):
    """Output from the evaluator agent."""

    # Self-RAG reflection tokens
    isrel: ReflectionScore = Field(description="Is context relevant?")
    issup: ReflectionScore = Field(description="Is response supported by context?")
    isuse: ReflectionScore = Field(description="Is response useful?")

    # Detailed scores
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    support_score: float = Field(ge=0.0, le=1.0, description="Support/faithfulness score")
    usefulness_score: float = Field(ge=0.0, le=1.0, description="Usefulness score")

    # Overall assessment
    overall_quality: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    pass_threshold: bool = Field(description="Whether response passes quality threshold")

    # Feedback
    issues: list[str] = Field(default_factory=list, description="Identified issues")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    reasoning: str = Field(default="", description="Evaluation reasoning")


class EvaluatorAgent(BaseAgent[EvaluatorOutput]):
    """
    Evaluator Agent with Self-RAG reflection capabilities.

    Responsibilities:
    1. Evaluate context relevance (ISREL)
    2. Evaluate response faithfulness (ISSUP)
    3. Evaluate response usefulness (ISUSE)
    4. Provide actionable feedback
    5. Decide if response should be regenerated
    """

    def __init__(
        self,
        generator: Generator,
        settings: Settings | None = None,
        quality_threshold: float = 0.7,
    ):
        """
        Initialize evaluator agent.

        Args:
            generator: LLM for evaluation.
            settings: Configuration settings.
            quality_threshold: Minimum quality to pass.
        """
        super().__init__(
            generator=generator,
            settings=settings,
            name="EvaluatorAgent",
        )
        self.quality_threshold = quality_threshold

    def _get_default_system_prompt(self) -> str:
        """Get evaluator-specific system prompt."""
        return """You are a response quality evaluator for a RAG system.
You evaluate responses using three criteria:

1. ISREL (Relevance): Is the retrieved context relevant to the query?
   - FULLY: Context directly addresses the query
   - PARTIALLY: Context has some relevant information
   - NOT: Context is unrelated

2. ISSUP (Support): Is the response supported by the context?
   - FULLY: Every claim is supported by the context
   - PARTIALLY: Some claims are supported, some are not
   - NOT: Response contradicts or ignores the context

3. ISUSE (Usefulness): Is the response useful to the user?
   - FULLY: Response completely answers the query
   - PARTIALLY: Response partially addresses the query
   - NOT: Response doesn't help the user

Be critical but fair in your assessment."""

    async def execute(self, state: AgentState) -> EvaluatorOutput:
        """
        Evaluate the response quality using Self-RAG reflection.

        Performs a three-dimensional evaluation:
        1. Relevance (ISREL): How well the context matches the query.
        2. Support (ISSUP): How well the response is grounded in the context.
        3. Usefulness (ISUSE): How well the response addresses the user's need.

        Args:
            state: AgentState containing query, retrieved chunks, and the generated response.

        Returns:
            EvaluatorOutput with scores, identified issues, and improvement suggestions.
        """
        query = state.query
        context_chunks = state.context.get("chunks", [])
        response = state.context.get("response", "")

        if not response:
            return EvaluatorOutput(
                isrel=ReflectionScore.NOT,
                issup=ReflectionScore.NOT,
                isuse=ReflectionScore.NOT,
                relevance_score=0.0,
                support_score=0.0,
                usefulness_score=0.0,
                overall_quality=0.0,
                pass_threshold=False,
                issues=["No response to evaluate"],
                reasoning="Empty response",
            )

        # Evaluate each dimension
        relevance = await self._evaluate_relevance(query, context_chunks)
        support = await self._evaluate_support(response, context_chunks)
        usefulness = await self._evaluate_usefulness(query, response)

        # Calculate overall quality
        overall = (relevance["score"] + support["score"] + usefulness["score"]) / 3

        return EvaluatorOutput(
            isrel=relevance["token"],
            issup=support["token"],
            isuse=usefulness["token"],
            relevance_score=relevance["score"],
            support_score=support["score"],
            usefulness_score=usefulness["score"],
            overall_quality=overall,
            pass_threshold=overall >= self.quality_threshold,
            issues=relevance["issues"] + support["issues"] + usefulness["issues"],
            suggestions=self._generate_suggestions(relevance, support, usefulness),
            reasoning=f"Relevance: {relevance['score']:.2f}, Support: {support['score']:.2f}, Usefulness: {usefulness['score']:.2f}",
        )

    async def _evaluate_relevance(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> dict[str, Any]:
        """
        Evaluate context relevance (ISREL).

        Args:
            query: User query.
            chunks: Retrieved context chunks.

        Returns:
            Relevance assessment.
        """
        if not chunks:
            return {
                "token": ReflectionScore.NOT,
                "score": 0.0,
                "issues": ["No context retrieved"],
            }

        context_text = "\n\n".join(c.content[:300] for c in chunks[:5])

        prompt = f"""Evaluate the relevance of this context to the query.

Query: "{query}"

Context:
{context_text}

Rate the relevance from 0.0 to 1.0 and identify any issues.
Output JSON: {{"score": 0.0-1.0, "issues": ["issue1", ...]}}"""

        response = await self.think(prompt, temperature=0.2)

        try:
            import json
            import re

            json_match = re.search(r"\{[\s\S]*\}", str(response))
            if json_match:
                data = json.loads(json_match.group())
                score = data.get("score", 0.5)
                issues = data.get("issues", [])

                if score >= 0.7:
                    token = ReflectionScore.FULLY
                elif score >= 0.4:
                    token = ReflectionScore.PARTIALLY
                else:
                    token = ReflectionScore.NOT

                return {"token": token, "score": score, "issues": issues}
        except Exception:
            pass

        # Fallback
        return {
            "token": ReflectionScore.PARTIALLY,
            "score": 0.5,
            "issues": ["Could not fully assess relevance"],
        }

    async def _evaluate_support(
        self,
        response: str,
        chunks: list[Chunk],
    ) -> dict[str, Any]:
        """
        Evaluate response faithfulness (ISSUP).

        Args:
            response: Generated response.
            chunks: Context chunks.

        Returns:
            Support assessment.
        """
        if not chunks:
            return {
                "token": ReflectionScore.NOT,
                "score": 0.0,
                "issues": ["No context to support response"],
            }

        context_text = "\n\n".join(c.content[:300] for c in chunks[:5])

        prompt = f"""Evaluate if this response is supported by the context.
Check for hallucinations or unsupported claims.

Context:
{context_text}

Response:
{response[:1000]}

Rate the support from 0.0 to 1.0 and list any unsupported claims.
Output JSON: {{"score": 0.0-1.0, "issues": ["unsupported claim 1", ...]}}"""

        result = await self.think(prompt, temperature=0.2)

        try:
            import json
            import re

            json_match = re.search(r"\{[\s\S]*\}", str(result))
            if json_match:
                data = json.loads(json_match.group())
                score = data.get("score", 0.5)
                issues = data.get("issues", [])

                if score >= 0.7:
                    token = ReflectionScore.FULLY
                elif score >= 0.4:
                    token = ReflectionScore.PARTIALLY
                else:
                    token = ReflectionScore.NOT

                return {"token": token, "score": score, "issues": issues}
        except Exception:
            pass

        return {
            "token": ReflectionScore.PARTIALLY,
            "score": 0.5,
            "issues": ["Could not fully assess support"],
        }

    async def _evaluate_usefulness(
        self,
        query: str,
        response: str,
    ) -> dict[str, Any]:
        """
        Evaluate response usefulness (ISUSE).

        Args:
            query: User query.
            response: Generated response.

        Returns:
            Usefulness assessment.
        """
        prompt = f"""Evaluate if this response is useful for answering the query.

Query: "{query}"

Response:
{response[:1000]}

Consider:
- Does it directly answer the question?
- Is it clear and understandable?
- Is it complete or does it miss important aspects?

Rate usefulness from 0.0 to 1.0 and list any issues.
Output JSON: {{"score": 0.0-1.0, "issues": ["issue1", ...]}}"""

        result = await self.think(prompt, temperature=0.2)

        try:
            import json
            import re

            json_match = re.search(r"\{[\s\S]*\}", str(result))
            if json_match:
                data = json.loads(json_match.group())
                score = data.get("score", 0.5)
                issues = data.get("issues", [])

                if score >= 0.7:
                    token = ReflectionScore.FULLY
                elif score >= 0.4:
                    token = ReflectionScore.PARTIALLY
                else:
                    token = ReflectionScore.NOT

                return {"token": token, "score": score, "issues": issues}
        except Exception:
            pass

        return {
            "token": ReflectionScore.PARTIALLY,
            "score": 0.5,
            "issues": ["Could not fully assess usefulness"],
        }

    def _generate_suggestions(
        self,
        relevance: dict[str, Any],
        support: dict[str, Any],
        usefulness: dict[str, Any],
    ) -> list[str]:
        """
        Generate improvement suggestions.

        Args:
            relevance: Relevance assessment.
            support: Support assessment.
            usefulness: Usefulness assessment.

        Returns:
            List of suggestions.
        """
        suggestions = []

        if relevance["score"] < 0.5:
            suggestions.append("Retrieve more relevant context using different query")

        if support["score"] < 0.5:
            suggestions.append("Regenerate response with stricter grounding to context")

        if usefulness["score"] < 0.5:
            suggestions.append("Ensure response directly addresses the user's question")

        if not suggestions:
            suggestions.append("Response quality is acceptable")

        return suggestions

    def create_reflection_tokens(
        self,
        output: EvaluatorOutput,
    ) -> list[ReflectionToken]:
        """
        Convert output to ReflectionToken models.

        Args:
            output: Evaluator output.

        Returns:
            List of reflection tokens.
        """
        return [
            ReflectionToken(
                token_type="ISREL",
                value=output.isrel.value,
                score=output.relevance_score,
            ),
            ReflectionToken(
                token_type="ISSUP",
                value=output.issup.value,
                score=output.support_score,
            ),
            ReflectionToken(
                token_type="ISUSE",
                value=output.isuse.value,
                score=output.usefulness_score,
            ),
        ]


class CriticAgent(EvaluatorAgent):
    """
    Enhanced evaluator that can provide detailed critique and rewriting suggestions.
    """

    async def critique_and_improve(
        self,
        query: str,
        response: str,
        chunks: list[Chunk],
    ) -> dict[str, Any]:
        """
        Provide detailed critique with improvement suggestions.

        Args:
            query: User query.
            response: Generated response.
            chunks: Context chunks.

        Returns:
            Critique with suggested improvements.
        """
        context_text = "\n\n".join(c.content[:300] for c in chunks[:5])

        prompt = f"""Critically evaluate this RAG response and suggest improvements.

Query: "{query}"

Context:
{context_text}

Response:
{response}

Provide:
1. Specific issues with the response
2. Factual errors or hallucinations (if any)
3. A rewritten version that addresses the issues

Output JSON:
{{
    "issues": ["issue1", ...],
    "hallucinations": ["hallucination1", ...],
    "improved_response": "The improved response text..."
}}"""

        result = await self.think(prompt, temperature=0.3)

        try:
            import json
            import re

            json_match = re.search(r"\{[\s\S]*\}", str(result))
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return {
            "issues": [],
            "hallucinations": [],
            "improved_response": response,
        }
