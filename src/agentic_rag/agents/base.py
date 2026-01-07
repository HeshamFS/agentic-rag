"""
Base agent class for the agentic RAG system.

All agents inherit from this base class which provides:
- LLM integration
- Structured output parsing
- Retry logic
- Logging
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.protocols import Generator

T = TypeVar("T", bound=BaseModel)


class AgentState(BaseModel):
    """Base state for agents."""

    query: str
    context: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    iteration: int = 0
    max_iterations: int = 3


class AgentOutput(BaseModel):
    """Base output from agents."""

    success: bool = True
    message: str = ""
    data: dict[str, Any] = {}
    reasoning: str = ""


class BaseAgent[T: BaseModel](ABC):
    """
    Base class for all agents in the agentic RAG system.

    Provides common functionality:
    - LLM-based reasoning
    - Structured output parsing
    - State management
    - Error handling
    """

    def __init__(
        self,
        generator: Generator,
        settings: Settings | None = None,
        name: str = "BaseAgent",
        system_prompt: str | None = None,
    ):
        """
        Initialize agent.

        Args:
            generator: LLM for reasoning.
            settings: Configuration settings.
            name: Agent name for logging.
            system_prompt: System prompt for the agent.
        """
        self._generator = generator
        self._settings = settings or get_settings()
        self.name = name
        self._system_prompt = system_prompt or self._get_default_system_prompt()

    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for this agent."""
        ...

    @abstractmethod
    async def execute(self, state: AgentState) -> T:
        """
        Execute the agent's main task.

        Args:
            state: Current agent state.

        Returns:
            Agent-specific output.
        """
        ...

    async def think(
        self,
        prompt: str,
        output_schema: type[BaseModel] | None = None,
        temperature: float = 0.3,
    ) -> str | BaseModel:
        """
        Have the agent think and produce output.

        Args:
            prompt: The prompt/question for the agent.
            output_schema: Optional Pydantic model for structured output.
            temperature: Generation temperature.

        Returns:
            String response or parsed Pydantic model.
        """
        full_prompt = f"{self._system_prompt}\n\n{prompt}"

        response = await self._generator.generate_text(
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=1024,
        )

        if output_schema:
            return self._parse_structured_output(response, output_schema)

        return response

    def _parse_structured_output(
        self,
        response: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        """
        Parse LLM response into structured output.

        Args:
            response: Raw LLM response.
            schema: Pydantic model to parse into.

        Returns:
            Parsed Pydantic model.
        """
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return schema.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        # Try parsing entire response as JSON
        try:
            data = json.loads(response)
            return schema.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

        # Return default instance with error
        return schema.model_validate(
            {
                "success": False,
                "message": f"Failed to parse response: {response[:200]}",
            }
        )

    async def reflect(
        self,
        action: str,
        result: Any,
        goal: str,
    ) -> str:
        """
        Reflect on an action and its result.

        Args:
            action: What action was taken.
            result: What happened.
            goal: What we're trying to achieve.

        Returns:
            Reflection/analysis.
        """
        prompt = f"""Reflect on this action and its result:

Action taken: {action}
Result: {result}
Goal: {goal}

Provide a brief analysis:
1. Was this action successful in moving toward the goal?
2. What should be the next step?
3. Are there any issues or improvements needed?"""

        reflection = await self.think(prompt, temperature=0.5)
        return str(reflection)

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message.

        Args:
            message: Message to log.
            level: Log level.
        """
        # Simple print logging - replace with proper logger in production
        prefix = f"[{self.name}]"
        print(f"{prefix} {message}")


class ReactAgent(BaseAgent[T]):
    """
    ReAct (Reasoning + Acting) agent pattern.

    Alternates between:
    - Thought: Reasoning about the current state
    - Action: Taking an action
    - Observation: Observing the result
    """

    @abstractmethod
    async def get_available_actions(self, state: AgentState) -> list[str]:
        """Get available actions for current state."""
        ...

    @abstractmethod
    async def execute_action(
        self,
        action: str,
        state: AgentState,
    ) -> tuple[Any, AgentState]:
        """Execute an action and return result + updated state."""
        ...

    async def react_loop(
        self,
        state: AgentState,
        max_iterations: int | None = None,
    ) -> T:
        """
        Run the ReAct loop.

        Args:
            state: Initial state.
            max_iterations: Maximum iterations.

        Returns:
            Final output.
        """
        max_iter = max_iterations or state.max_iterations

        while state.iteration < max_iter:
            # Thought
            actions = await self.get_available_actions(state)
            thought_prompt = f"""Current state:
Query: {state.query}
Context: {state.context}
Previous actions: {state.history}

Available actions: {actions}

What should be the next action and why?
Format: THOUGHT: [your reasoning] ACTION: [action name]"""

            thought = await self.think(thought_prompt, temperature=0.3)
            thought_str = str(thought)

            # Parse action from thought
            action_match = re.search(r"ACTION:\s*(\w+)", thought_str)
            if not action_match:
                # Default to first available action
                action = actions[0] if actions else "stop"
            else:
                action = action_match.group(1)

            # Action
            if action == "stop" or action == "finish":
                break

            result, state = await self.execute_action(action, state)

            # Observation
            state.history.append(
                {
                    "thought": thought_str,
                    "action": action,
                    "result": str(result),
                    "iteration": state.iteration,
                }
            )

            state.iteration += 1

        return await self.execute(state)
