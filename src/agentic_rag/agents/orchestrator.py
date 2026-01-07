"""
Orchestrator Agent - Master planner for the agentic RAG system.

Coordinates all other agents and manages the overall workflow.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.agents.base import AgentState, BaseAgent
from agentic_rag.agents.evaluator import EvaluatorAgent
from agentic_rag.agents.generator import GeneratorAgent
from agentic_rag.agents.retriever import (
    CorrectionAction,
    RetrieverAgent,
)
from agentic_rag.agents.router import (
    RetrievalStrategy,
    RouterAgent,
    RouterOutput,
)
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, GenerationResult, RetrievalResult
from agentic_rag.core.protocols import Embedder, Generator, VectorDB
from agentic_rag.retrieval import BaseRetriever, HybridRetriever, HyDERetriever


class WorkflowStep(str, Enum):
    """Steps in the orchestration workflow."""

    ROUTE = "route"
    RETRIEVE = "retrieve"
    EVALUATE_RETRIEVAL = "evaluate_retrieval"
    GENERATE = "generate"
    EVALUATE_RESPONSE = "evaluate_response"
    COMPLETE = "complete"
    ERROR = "error"


class OrchestratorOutput(BaseModel):
    """Output from the orchestrator."""

    response: str = Field(description="Final response")
    sources: list[Chunk] = Field(default_factory=list, description="Source chunks used")
    workflow_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Workflow execution history"
    )
    quality_scores: dict[str, float] = Field(default_factory=dict, description="Quality metrics")
    iterations: int = Field(default=1, description="Number of iterations")
    success: bool = Field(default=True, description="Whether workflow succeeded")
    error_message: str = Field(default="", description="Error message if failed")


class OrchestratorAgent(BaseAgent[OrchestratorOutput]):
    """
    Orchestrator Agent - Master planner for the agentic RAG system.

    Coordinates the multi-agent workflow:
    1. RouterAgent: Classifies query intent and retrieval strategy.
    2. RetrieverAgent: Evaluates initial retrieval and applies CRAG corrections.
    3. GeneratorAgent: Synthesizes final response with Self-RAG reflection.
    4. EvaluatorAgent: Performs quality assessment and triggers iterations.

    Implements:
    - Multi-agent collaboration using a shared state.
    - Iterative refinement based on quality scores.
    - Context-aware routing and correction.
    """

    def __init__(
        self,
        generator: Generator,
        embedder: Embedder,
        vectordb: VectorDB,
        settings: Settings | None = None,
        max_iterations: int = 3,
    ):
        """
        Initialize orchestrator.

        Args:
            generator: LLM for all agents.
            embedder: Embedding model.
            vectordb: Vector database.
            settings: Configuration settings.
            max_iterations: Maximum refinement iterations.
        """
        super().__init__(
            generator=generator,
            settings=settings,
            name="OrchestratorAgent",
        )
        self._embedder = embedder
        self._vectordb = vectordb
        self.max_iterations = max_iterations

        # Initialize sub-agents
        self._router = RouterAgent(generator, settings)
        self._retriever_agent = RetrieverAgent(generator, settings)
        self._evaluator = EvaluatorAgent(generator, settings)
        self._generator_agent = GeneratorAgent(generator, settings)

        # Initialize retrievers
        self._hybrid_retriever = HybridRetriever(
            embedder=embedder,
            vectordb=vectordb,
            settings=settings,
        )

    def _get_default_system_prompt(self) -> str:
        """Get orchestrator-specific system prompt."""
        return """You are the master orchestrator of an agentic RAG system.
You coordinate multiple specialized agents to deliver high-quality responses.
Make decisions about when to iterate, when to accept results, and how to handle failures."""

    async def execute(self, state: AgentState) -> OrchestratorOutput:
        """
        Execute the full multi-agent orchestration workflow.

        Follows these steps:
        1. ROUTE: Classify query and determine strategy.
        2. RETRIEVE: Execute initial retrieval.
        3. EVALUATE_RETRIEVAL: Assess quality and apply CRAG if needed.
        4. GENERATE: Produce a response based on context.
        5. EVALUATE_RESPONSE: Self-RAG quality check.
        6. COMPLETE: Return final result or iterate.

        Args:
            state: AgentState containing the query and context.

        Returns:
            OrchestratorOutput with response, sources, and workflow history.
        """
        workflow_history = []
        current_step = WorkflowStep.ROUTE
        iteration = 0

        collection = state.context.get("collection", "default")
        query = state.query
        chunks: list[Chunk] = []
        response = ""

        while iteration < self.max_iterations:
            iteration += 1

            try:
                # Step 1: Route query
                if current_step == WorkflowStep.ROUTE:
                    router_output = await self._route_query(query)
                    workflow_history.append(
                        {
                            "step": "route",
                            "output": router_output.model_dump(),
                        }
                    )

                    # Handle conversational queries without retrieval
                    if router_output.retrieval_strategy == RetrievalStrategy.NONE:
                        response = await self._direct_response(query)
                        return OrchestratorOutput(
                            response=response,
                            sources=[],
                            workflow_history=workflow_history,
                            quality_scores={},
                            iterations=iteration,
                        )

                    current_step = WorkflowStep.RETRIEVE

                # Step 2: Retrieve
                if current_step == WorkflowStep.RETRIEVE:
                    retrieval_result = await self._retrieve(
                        query=query,
                        collection=collection,
                        strategy=router_output.retrieval_strategy,
                        top_k=router_output.top_k_recommendation,
                    )
                    chunks = retrieval_result.chunks
                    workflow_history.append(
                        {
                            "step": "retrieve",
                            "num_chunks": len(chunks),
                            "strategy": router_output.retrieval_strategy.value,
                        }
                    )

                    current_step = WorkflowStep.EVALUATE_RETRIEVAL

                # Step 3: Evaluate retrieval
                if current_step == WorkflowStep.EVALUATE_RETRIEVAL:
                    state.context["retrieval_result"] = retrieval_result
                    retriever_output = await self._retriever_agent.execute(state)
                    workflow_history.append(
                        {
                            "step": "evaluate_retrieval",
                            "output": retriever_output.model_dump(),
                        }
                    )

                    # Handle corrective actions
                    if retriever_output.action == CorrectionAction.REFINE_QUERY:
                        query = await self._retriever_agent.refine_query(query, retrieval_result)
                        current_step = WorkflowStep.RETRIEVE
                        continue

                    if retriever_output.action == CorrectionAction.EXPAND_SEARCH:
                        retrieval_result = await self._retrieve(
                            query=query,
                            collection=collection,
                            strategy=RetrievalStrategy.HYDE,
                            top_k=router_output.top_k_recommendation * 2,
                        )
                        chunks = retrieval_result.chunks

                    # Filter to relevant chunks
                    if retriever_output.relevant_chunks:
                        chunks = self._retriever_agent.filter_relevant_chunks(
                            chunks, retriever_output.relevant_chunks
                        )

                    current_step = WorkflowStep.GENERATE

                # Step 4: Generate response
                if current_step == WorkflowStep.GENERATE:
                    state.context["chunks"] = chunks
                    generator_output = await self._generator_agent.execute(state)
                    response = generator_output.response
                    workflow_history.append(
                        {
                            "step": "generate",
                            "output": {
                                "confidence": generator_output.confidence,
                                "citations": generator_output.citations,
                            },
                        }
                    )

                    current_step = WorkflowStep.EVALUATE_RESPONSE

                # Step 5: Evaluate response
                if current_step == WorkflowStep.EVALUATE_RESPONSE:
                    state.context["response"] = response
                    eval_output = await self._evaluator.execute(state)
                    workflow_history.append(
                        {
                            "step": "evaluate_response",
                            "output": eval_output.model_dump(),
                        }
                    )

                    if eval_output.pass_threshold:
                        current_step = WorkflowStep.COMPLETE
                    else:
                        # Need to iterate
                        if iteration >= self.max_iterations:
                            current_step = WorkflowStep.COMPLETE
                        else:
                            # Adjust strategy and retry
                            current_step = WorkflowStep.RETRIEVE
                            continue

                # Completion
                if current_step == WorkflowStep.COMPLETE:
                    return OrchestratorOutput(
                        response=response,
                        sources=chunks,
                        workflow_history=workflow_history,
                        quality_scores={
                            "relevance": eval_output.relevance_score,
                            "support": eval_output.support_score,
                            "usefulness": eval_output.usefulness_score,
                            "overall": eval_output.overall_quality,
                        },
                        iterations=iteration,
                    )

            except Exception as e:
                workflow_history.append(
                    {
                        "step": "error",
                        "error": str(e),
                    }
                )
                return OrchestratorOutput(
                    response=f"An error occurred: {str(e)}",
                    sources=[],
                    workflow_history=workflow_history,
                    quality_scores={},
                    iterations=iteration,
                    success=False,
                    error_message=str(e),
                )

        # Max iterations reached
        return OrchestratorOutput(
            response=response or "Could not generate a satisfactory response.",
            sources=chunks,
            workflow_history=workflow_history,
            quality_scores={},
            iterations=iteration,
        )

    async def _route_query(self, query: str) -> RouterOutput:
        """Route the query."""
        state = AgentState(query=query)
        return await self._router.execute(state)

    async def _retrieve(
        self,
        query: str,
        collection: str,
        strategy: RetrievalStrategy,
        top_k: int,
    ) -> RetrievalResult:
        """
        Perform retrieval with specified strategy.

        Args:
            query: Search query.
            collection: Collection name.
            strategy: Retrieval strategy.
            top_k: Number of results.

        Returns:
            Retrieval result.
        """
        if strategy == RetrievalStrategy.HYDE:
            hyde_retriever = HyDERetriever(
                generator=self._generator,
                embedder=self._embedder,
                vectordb=self._vectordb,
                settings=self._settings,
            )
            return await hyde_retriever.retrieve(
                query=query,
                collection=collection,
                top_k=top_k,
            )
        else:
            return await self._hybrid_retriever.retrieve(
                query=query,
                collection=collection,
                top_k=top_k,
            )

    async def _direct_response(self, query: str) -> str:
        """Generate direct response without retrieval."""
        return await self._generator.generate_text(
            prompt=query,
            temperature=0.7,
            max_tokens=512,
        )

    async def close(self) -> None:
        """Clean up resources."""
        await self._hybrid_retriever.close()


class SimpleOrchestrator:
    """
    Simplified orchestrator for non-agentic pipelines.

    Provides a straightforward retrieve-then-generate flow
    without the multi-agent complexity.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: Generator,
        settings: Settings | None = None,
    ):
        """
        Initialize simple orchestrator.

        Args:
            retriever: Retriever to use.
            generator: LLM for generation.
            settings: Configuration settings.
        """
        self._retriever = retriever
        self._generator = generator
        self._settings = settings or get_settings()

    async def query(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
    ) -> GenerationResult:
        """
        Execute simple retrieve-then-generate.

        Args:
            query: User query.
            collection: Collection to search.
            top_k: Number of chunks to retrieve.

        Returns:
            Generation result.
        """
        import time

        start_time = time.time()

        # Retrieve
        retrieval_result = await self._retriever.retrieve(
            query=query,
            collection=collection,
            top_k=top_k,
        )

        # Build context
        context = "\n\n---\n\n".join(c.content for c in retrieval_result.chunks)

        # Generate
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
        )

        latency_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            response=response,
            sources=retrieval_result.chunks,
            model=getattr(self._generator, "model", "unknown"),
            provider=self._generator.provider,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=latency_ms,
        )

    async def close(self) -> None:
        """Clean up resources."""
        await self._retriever.close()
