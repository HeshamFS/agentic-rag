"""
Query planning for complex questions.

Decomposes complex queries into executable sub-plans
for multi-step reasoning.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.config import Settings, get_settings
from agentic_rag.generation import BaseGenerator


class PlanStepType(str, Enum):
    """Types of plan steps."""

    RETRIEVE = "retrieve"  # Search for information
    ANALYZE = "analyze"  # Analyze retrieved information
    COMPARE = "compare"  # Compare multiple pieces of info
    SYNTHESIZE = "synthesize"  # Combine information
    VERIFY = "verify"  # Verify a claim or result
    RESPOND = "respond"  # Generate final response


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    step_id: int
    step_type: PlanStepType
    description: str
    query: str = ""  # Search query for retrieve steps
    depends_on: list[int] = Field(default_factory=list)
    completed: bool = False
    result: Any = None


class ExecutionPlan(BaseModel):
    """A complete execution plan."""

    original_query: str
    steps: list[PlanStep]
    reasoning: str = ""


class QueryPlanner:
    """
    Plans execution strategy for complex queries.

    Analyzes the query and creates a step-by-step plan
    for answering it.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        settings: Settings | None = None,
    ):
        """
        Initialize planner.

        Args:
            generator: LLM for planning.
            settings: Settings instance.
        """
        self._generator = generator
        self._settings = settings or get_settings()

    async def create_plan(self, query: str) -> ExecutionPlan:
        """
        Create an execution plan for a query.

        Args:
            query: User query.

        Returns:
            ExecutionPlan with steps.
        """
        prompt = f"""Analyze this question and create a step-by-step plan to answer it.

Question: {query}

For each step, specify:
- Type: retrieve (search), analyze, compare, synthesize, verify, or respond
- Description: What this step does
- Query: Search query (for retrieve steps only)
- Dependencies: Which previous steps this depends on

Output as JSON:
{{
  "reasoning": "Why this plan structure",
  "steps": [
    {{"step_id": 1, "step_type": "retrieve", "description": "...", "query": "...", "depends_on": []}},
    {{"step_id": 2, "step_type": "analyze", "description": "...", "depends_on": [1]}},
    ...
  ]
}}

Keep it simple - use 2-5 steps for most queries."""

        response = await self._generator.generate_text(prompt, max_tokens=500)
        return self._parse_plan(query, response)

    def _parse_plan(self, query: str, response: str) -> ExecutionPlan:
        """Parse JSON plan response."""
        import json

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                steps = []
                for step_data in data.get("steps", []):
                    step = PlanStep(
                        step_id=step_data.get("step_id", len(steps) + 1),
                        step_type=PlanStepType(step_data.get("step_type", "retrieve")),
                        description=step_data.get("description", ""),
                        query=step_data.get("query", ""),
                        depends_on=step_data.get("depends_on", []),
                    )
                    steps.append(step)

                return ExecutionPlan(
                    original_query=query,
                    steps=steps,
                    reasoning=data.get("reasoning", ""),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Default to simple single-step plan
        return ExecutionPlan(
            original_query=query,
            steps=[
                PlanStep(
                    step_id=1,
                    step_type=PlanStepType.RETRIEVE,
                    description="Search for relevant information",
                    query=query,
                ),
                PlanStep(
                    step_id=2,
                    step_type=PlanStepType.RESPOND,
                    description="Generate response from retrieved information",
                    depends_on=[1],
                ),
            ],
        )

    async def should_plan(self, query: str) -> bool:
        """
        Determine if a query needs planning.

        Simple factual queries don't need planning.
        Complex, multi-part, or analytical queries do.

        Args:
            query: User query.

        Returns:
            True if planning is recommended.
        """
        prompt = f"""Analyze this question and determine if it requires multi-step planning.

Question: {query}

A question needs planning if it:
- Asks for comparison between multiple things
- Requires gathering information from multiple sources
- Involves analysis or synthesis
- Has multiple sub-questions
- Needs verification or fact-checking

Simple factual questions that can be answered with a single search do NOT need planning.

Respond with only: YES or NO"""

        response = await self._generator.generate_text(prompt, max_tokens=10)
        return "YES" in response.upper()


class PlanExecutor:
    """
    Executes a plan step by step.

    Coordinates retrieval and analysis for each step.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        settings: Settings | None = None,
    ):
        """Initialize executor."""
        self._generator = generator
        self._settings = settings or get_settings()

    async def execute_step(
        self,
        step: PlanStep,
        context: dict[int, Any],
        retriever: Any = None,  # BaseRetriever
        collection: str = "",
    ) -> Any:
        """
        Execute a single plan step.

        Args:
            step: Step to execute.
            context: Results from previous steps.
            retriever: Retriever for search steps.
            collection: Collection to search.

        Returns:
            Step result.
        """
        # Gather dependency results
        dep_results = [context.get(dep_id, "") for dep_id in step.depends_on]
        dep_context = "\n\n".join(str(r) for r in dep_results if r)

        if step.step_type == PlanStepType.RETRIEVE:
            if retriever:
                result = await retriever.retrieve(
                    query=step.query or step.description,
                    collection=collection,
                )
                return result.chunks
            return []

        elif step.step_type == PlanStepType.ANALYZE:
            prompt = f"""Analyze the following information:

{dep_context}

Task: {step.description}

Analysis:"""
            return await self._generator.generate_text(prompt, max_tokens=500)

        elif step.step_type == PlanStepType.COMPARE:
            prompt = f"""Compare and contrast the following:

{dep_context}

Task: {step.description}

Comparison:"""
            return await self._generator.generate_text(prompt, max_tokens=500)

        elif step.step_type == PlanStepType.SYNTHESIZE:
            prompt = f"""Synthesize the following information:

{dep_context}

Task: {step.description}

Synthesis:"""
            return await self._generator.generate_text(prompt, max_tokens=500)

        elif step.step_type == PlanStepType.VERIFY:
            prompt = f"""Verify the following:

{dep_context}

Task: {step.description}

Verification:"""
            return await self._generator.generate_text(prompt, max_tokens=300)

        elif step.step_type == PlanStepType.RESPOND:
            prompt = f"""Based on the following analysis:

{dep_context}

Generate a comprehensive response.

Response:"""
            return await self._generator.generate_text(prompt, max_tokens=1000)

        return None

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        retriever: Any = None,
        collection: str = "",
    ) -> dict[int, Any]:
        """
        Execute all steps in a plan.

        Args:
            plan: Execution plan.
            retriever: Retriever for search steps.
            collection: Collection to search.

        Returns:
            Dict mapping step_id to result.
        """
        results: dict[int, Any] = {}

        for step in plan.steps:
            # Check dependencies are complete
            for dep_id in step.depends_on:
                if dep_id not in results:
                    continue  # Skip if dependency not met

            result = await self.execute_step(
                step=step,
                context=results,
                retriever=retriever,
                collection=collection,
            )
            results[step.step_id] = result
            step.completed = True
            step.result = result

        return results
