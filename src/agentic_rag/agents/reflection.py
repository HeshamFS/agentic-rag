"""
Reflection patterns for agentic RAG.

Implements self-reflection capabilities for agents to
evaluate and improve their outputs.
"""

from enum import Enum

from pydantic import BaseModel, Field

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk
from agentic_rag.generation import BaseGenerator


class ReflectionType(str, Enum):
    """Types of reflection."""

    RELEVANCE = "relevance"  # Is context relevant to query?
    SUPPORT = "support"  # Is response supported by context?
    USEFULNESS = "usefulness"  # Is response useful to user?
    COMPLETENESS = "completeness"  # Is response complete?
    ACCURACY = "accuracy"  # Is response accurate?


class ReflectionResult(BaseModel):
    """Result of a reflection."""

    reflection_type: ReflectionType
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    suggestions: list[str] = Field(default_factory=list)


class Reflector:
    """
    Agent reflection capabilities.

    Enables agents to evaluate their own outputs and
    identify areas for improvement.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        settings: Settings | None = None,
    ):
        """
        Initialize reflector.

        Args:
            generator: LLM for reflection.
            settings: Settings instance.
        """
        self._generator = generator
        self._settings = settings or get_settings()

    async def reflect_relevance(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> ReflectionResult:
        """
        Reflect on context relevance.

        Args:
            query: User query.
            chunks: Retrieved chunks.

        Returns:
            ReflectionResult on relevance.
        """
        context = "\n\n".join([f"[{i + 1}] {c.content[:500]}" for i, c in enumerate(chunks[:5])])

        prompt = f"""Evaluate if the retrieved context is relevant to the query.

Query: {query}

Context:
{context}

Assess:
1. Does the context contain information related to the query?
2. Would this context help answer the query?
3. Is there irrelevant or misleading information?

Respond in JSON:
{{"passed": true/false, "confidence": 0.0-1.0, "reasoning": "explanation", "suggestions": ["suggestion1", ...]}}"""

        response = await self._generator.generate_text(prompt, max_tokens=300)
        return self._parse_reflection(response, ReflectionType.RELEVANCE)

    async def reflect_support(
        self,
        query: str,
        response: str,
        chunks: list[Chunk],
    ) -> ReflectionResult:
        """
        Reflect on response support.

        Args:
            query: User query.
            response: Generated response.
            chunks: Source chunks.

        Returns:
            ReflectionResult on support.
        """
        context = "\n\n".join([f"[{i + 1}] {c.content[:400]}" for i, c in enumerate(chunks[:5])])

        prompt = f"""Evaluate if the response is supported by the context.

Query: {query}

Response: {response}

Context:
{context}

Assess:
1. Are claims in the response backed by the context?
2. Are there unsupported statements?
3. Is information from the context accurately represented?

Respond in JSON:
{{"passed": true/false, "confidence": 0.0-1.0, "reasoning": "explanation", "suggestions": ["suggestion1", ...]}}"""

        response_text = await self._generator.generate_text(prompt, max_tokens=300)
        return self._parse_reflection(response_text, ReflectionType.SUPPORT)

    async def reflect_usefulness(
        self,
        query: str,
        response: str,
    ) -> ReflectionResult:
        """
        Reflect on response usefulness.

        Args:
            query: User query.
            response: Generated response.

        Returns:
            ReflectionResult on usefulness.
        """
        prompt = f"""Evaluate if the response is useful for the user.

Query: {query}

Response: {response}

Assess:
1. Does the response directly address the query?
2. Is the response clear and understandable?
3. Does it provide actionable or informative content?

Respond in JSON:
{{"passed": true/false, "confidence": 0.0-1.0, "reasoning": "explanation", "suggestions": ["suggestion1", ...]}}"""

        response_text = await self._generator.generate_text(prompt, max_tokens=300)
        return self._parse_reflection(response_text, ReflectionType.USEFULNESS)

    async def reflect_completeness(
        self,
        query: str,
        response: str,
    ) -> ReflectionResult:
        """
        Reflect on response completeness.

        Args:
            query: User query.
            response: Generated response.

        Returns:
            ReflectionResult on completeness.
        """
        prompt = f"""Evaluate if the response is complete.

Query: {query}

Response: {response}

Assess:
1. Does the response cover all aspects of the query?
2. Are there missing details that should be included?
3. Is additional context needed?

Respond in JSON:
{{"passed": true/false, "confidence": 0.0-1.0, "reasoning": "explanation", "suggestions": ["suggestion1", ...]}}"""

        response_text = await self._generator.generate_text(prompt, max_tokens=300)
        return self._parse_reflection(response_text, ReflectionType.COMPLETENESS)

    async def full_reflection(
        self,
        query: str,
        response: str,
        chunks: list[Chunk],
    ) -> list[ReflectionResult]:
        """
        Perform full reflection across all types.

        Args:
            query: User query.
            response: Generated response.
            chunks: Source chunks.

        Returns:
            List of ReflectionResults.
        """
        import asyncio

        results = await asyncio.gather(
            self.reflect_relevance(query, chunks),
            self.reflect_support(query, response, chunks),
            self.reflect_usefulness(query, response),
            self.reflect_completeness(query, response),
        )

        return list(results)

    def _parse_reflection(
        self,
        response: str,
        reflection_type: ReflectionType,
    ) -> ReflectionResult:
        """Parse JSON reflection response."""
        import json

        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return ReflectionResult(
                    reflection_type=reflection_type,
                    passed=data.get("passed", False),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    suggestions=data.get("suggestions", []),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Default to failed if parsing fails
        return ReflectionResult(
            reflection_type=reflection_type,
            passed=False,
            confidence=0.5,
            reasoning="Could not parse reflection",
        )


class SelfCritiqueChain:
    """
    Self-critique chain for iterative improvement.

    Generates response, critiques it, then improves.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        max_iterations: int = 3,
        settings: Settings | None = None,
    ):
        """Initialize self-critique chain."""
        self._generator = generator
        self._reflector = Reflector(generator, settings)
        self._max_iterations = max_iterations
        self._settings = settings or get_settings()

    async def run(
        self,
        query: str,
        chunks: list[Chunk],
        initial_response: str | None = None,
    ) -> tuple[str, list[ReflectionResult]]:
        """
        Run self-critique loop.

        Args:
            query: User query.
            chunks: Source chunks.
            initial_response: Optional initial response.

        Returns:
            Tuple of (final_response, reflection_history).
        """

        # Generate initial response if not provided
        if initial_response is None:
            result = await self._generator.generate(query, chunks)
            response = result.response
        else:
            response = initial_response

        all_reflections: list[ReflectionResult] = []

        for _ in range(self._max_iterations):
            # Reflect on current response
            reflections = await self._reflector.full_reflection(query, response, chunks)
            all_reflections.extend(reflections)

            # Check if all passed
            if all(r.passed for r in reflections):
                break

            # Collect suggestions
            suggestions = []
            for r in reflections:
                if not r.passed:
                    suggestions.extend(r.suggestions)

            if not suggestions:
                break

            # Improve response based on suggestions
            improve_prompt = f"""Improve this response based on the feedback.

Original Query: {query}

Current Response: {response}

Feedback to address:
{chr(10).join(f"- {s}" for s in suggestions[:5])}

Improved Response:"""

            response = await self._generator.generate_text(improve_prompt, max_tokens=1000)

        return response, all_reflections
