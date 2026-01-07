"""
Router Agent for query intent classification.

Analyzes incoming queries and routes them to appropriate handlers:
- Simple factual queries → Direct retrieval
- Complex analytical queries → Multi-step retrieval with HyDE
- Conversational queries → Direct LLM response
- Ambiguous queries → Clarification request
"""

from enum import Enum

from pydantic import BaseModel, Field

from agentic_rag.agents.base import AgentState, BaseAgent
from agentic_rag.config import Settings
from agentic_rag.core.protocols import Generator


class QueryType(str, Enum):
    """Types of queries the router can identify."""

    FACTUAL = "factual"  # Simple fact lookup
    ANALYTICAL = "analytical"  # Complex reasoning required
    COMPARISON = "comparison"  # Compare multiple items
    PROCEDURAL = "procedural"  # How-to questions
    CONVERSATIONAL = "conversational"  # Chitchat, no retrieval needed
    AMBIGUOUS = "ambiguous"  # Needs clarification


class RetrievalStrategy(str, Enum):
    """Retrieval strategies based on query type."""

    DIRECT = "direct"  # Standard dense retrieval
    HYDE = "hyde"  # Use hypothetical document embeddings
    MULTI_QUERY = "multi_query"  # Generate multiple query variants
    ITERATIVE = "iterative"  # Multiple retrieval rounds
    NONE = "none"  # No retrieval needed


class RouterOutput(BaseModel):
    """Output from the router agent."""

    query_type: QueryType = Field(description="Classified query type")
    retrieval_strategy: RetrievalStrategy = Field(description="Recommended retrieval strategy")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    sub_queries: list[str] = Field(
        default_factory=list, description="Decomposed sub-queries if applicable"
    )
    reasoning: str = Field(default="", description="Explanation of routing decision")
    requires_clarification: bool = Field(
        default=False, description="Whether to ask for clarification"
    )
    clarification_question: str = Field(default="", description="Question to ask user if needed")
    top_k_recommendation: int = Field(
        default=10, description="Recommended number of chunks to retrieve"
    )


class RouterAgent(BaseAgent[RouterOutput]):
    """
    Router Agent for intelligent query routing.

    Responsibilities:
    1. Classify query intent
    2. Determine optimal retrieval strategy
    3. Decompose complex queries into sub-queries
    4. Estimate retrieval parameters
    """

    def __init__(
        self,
        generator: Generator,
        settings: Settings | None = None,
    ):
        """
        Initialize router agent.

        Args:
            generator: LLM for classification.
            settings: Configuration settings.
        """
        super().__init__(
            generator=generator,
            settings=settings,
            name="RouterAgent",
        )

    def _get_default_system_prompt(self) -> str:
        """Get router-specific system prompt."""
        return """You are a query routing specialist for a RAG system.
Your job is to analyze user queries and determine:
1. The type of query (factual, analytical, comparison, procedural, conversational, ambiguous)
2. The best retrieval strategy (direct, hyde, multi_query, iterative, none)
3. Whether the query should be decomposed into sub-queries
4. How many documents to retrieve (top_k)

Always respond with a JSON object containing your analysis."""

    async def execute(self, state: AgentState) -> RouterOutput:
        """
        Execute the query routing logic.

        1. Performs quick greeting detection for casual conversational queries.
        2. Uses the LLM to classify query type and recommend a retrieval strategy.
        3. Falls back to heuristic-based default routing if LLM classification fails.

        Args:
            state: AgentState containing the user query.

        Returns:
            RouterOutput containing classification, recommended strategy, and reasoning.
        """
        query = state.query

        # Quick classification for obvious cases
        if self._is_greeting(query):
            return RouterOutput(
                query_type=QueryType.CONVERSATIONAL,
                retrieval_strategy=RetrievalStrategy.NONE,
                confidence=0.95,
                reasoning="Query is a greeting or casual conversation",
                top_k_recommendation=0,
            )

        # Use LLM for classification
        prompt = self._build_classification_prompt(query)
        result = await self.think(prompt, output_schema=RouterOutput)

        if isinstance(result, RouterOutput):
            return result

        # Fallback to default routing
        return self._default_routing(query)

    def _is_greeting(self, query: str) -> bool:
        """Check if query is a greeting."""
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        query_lower = query.lower().strip()
        return query_lower in greetings or query_lower.startswith(("hi ", "hello ", "hey "))

    def _build_classification_prompt(self, query: str) -> str:
        """Build the classification prompt."""
        return f"""Analyze this query and provide routing information.

Query: "{query}"

Classify the query and respond with JSON:
{{
    "query_type": "factual|analytical|comparison|procedural|conversational|ambiguous",
    "retrieval_strategy": "direct|hyde|multi_query|iterative|none",
    "confidence": 0.0-1.0,
    "sub_queries": ["sub-query 1", "sub-query 2"] (if query should be decomposed),
    "reasoning": "Brief explanation of your decision",
    "requires_clarification": true/false,
    "clarification_question": "Question to ask if ambiguous",
    "top_k_recommendation": 5-20
}}

Guidelines:
- FACTUAL: Simple fact questions → DIRECT retrieval, top_k=5-10
- ANALYTICAL: Complex reasoning → HYDE or MULTI_QUERY, top_k=10-15
- COMPARISON: Multiple items → MULTI_QUERY with sub-queries, top_k=15-20
- PROCEDURAL: How-to → DIRECT or HYDE, top_k=10
- CONVERSATIONAL: No domain knowledge needed → NONE
- AMBIGUOUS: Unclear intent → require clarification"""

    def _default_routing(self, query: str) -> RouterOutput:
        """Provide default routing when LLM fails."""
        query_lower = query.lower()

        # Simple heuristics
        if any(w in query_lower for w in ["how to", "how do", "how can"]):
            return RouterOutput(
                query_type=QueryType.PROCEDURAL,
                retrieval_strategy=RetrievalStrategy.DIRECT,
                confidence=0.6,
                reasoning="Query appears to be a how-to question",
                top_k_recommendation=10,
            )

        if any(w in query_lower for w in ["compare", "difference", "versus", " vs "]):
            return RouterOutput(
                query_type=QueryType.COMPARISON,
                retrieval_strategy=RetrievalStrategy.MULTI_QUERY,
                confidence=0.6,
                reasoning="Query appears to be a comparison",
                top_k_recommendation=15,
            )

        if any(w in query_lower for w in ["why", "explain", "analyze"]):
            return RouterOutput(
                query_type=QueryType.ANALYTICAL,
                retrieval_strategy=RetrievalStrategy.HYDE,
                confidence=0.6,
                reasoning="Query requires analytical reasoning",
                top_k_recommendation=12,
            )

        # Default: factual
        return RouterOutput(
            query_type=QueryType.FACTUAL,
            retrieval_strategy=RetrievalStrategy.DIRECT,
            confidence=0.5,
            reasoning="Default routing for factual query",
            top_k_recommendation=10,
        )


class MultiQueryGenerator:
    """
    Generates multiple query variants for improved retrieval.

    Useful for comparison queries or when a single query might miss relevant documents.
    """

    def __init__(self, generator: Generator):
        """
        Initialize multi-query generator.

        Args:
            generator: LLM for query generation.
        """
        self._generator = generator

    async def generate_variants(
        self,
        query: str,
        num_variants: int = 3,
    ) -> list[str]:
        """
        Generate query variants.

        Args:
            query: Original query.
            num_variants: Number of variants to generate.

        Returns:
            List of query variants including original.
        """
        prompt = f"""Generate {num_variants} different ways to ask this question.
Each variant should potentially match different relevant documents.

Original query: "{query}"

Output {num_variants} variants, one per line, without numbering or bullets.
Just the query text, nothing else."""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
        )

        variants = [query]  # Always include original
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and line != query:
                variants.append(line)

        return variants[: num_variants + 1]  # Original + variants

    async def decompose_query(
        self,
        query: str,
    ) -> list[str]:
        """
        Decompose a complex query into sub-queries.

        Args:
            query: Complex query.

        Returns:
            List of sub-queries.
        """
        prompt = f"""Break down this complex query into simpler sub-queries that can be answered independently.

Query: "{query}"

Output each sub-query on a new line.
Only output the sub-queries, nothing else."""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.5,
            max_tokens=256,
        )

        sub_queries = []
        for line in response.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line:
                sub_queries.append(line)

        return sub_queries if sub_queries else [query]
