"""
Generator Agent for response generation.

Generates high-quality responses grounded in retrieved context.
"""

from pydantic import BaseModel, Field

from agentic_rag.agents.base import AgentState, BaseAgent
from agentic_rag.config import Settings
from agentic_rag.core.models import Chunk
from agentic_rag.core.protocols import Generator


class GeneratorOutput(BaseModel):
    """Output from the generator agent."""

    response: str = Field(description="Generated response")
    citations: list[int] = Field(default_factory=list, description="Indices of cited chunks")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in response")
    reasoning: str = Field(default="", description="Generation reasoning")
    follow_up_questions: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )


class GeneratorAgent(BaseAgent[GeneratorOutput]):
    """
    Generator Agent for response generation.

    Responsibilities:
    1. Generate grounded responses from context
    2. Add citations to sources
    3. Handle different response formats
    4. Suggest follow-up questions
    """

    def __init__(
        self,
        generator: Generator,
        settings: Settings | None = None,
        include_citations: bool = True,
    ):
        """
        Initialize generator agent.

        Args:
            generator: LLM for generation.
            settings: Configuration settings.
            include_citations: Whether to include citations.
        """
        super().__init__(
            generator=generator,
            settings=settings,
            name="GeneratorAgent",
        )
        self.include_citations = include_citations

    def _get_default_system_prompt(self) -> str:
        """Get generator-specific system prompt."""
        return """You are a helpful assistant that generates accurate, well-structured responses.

Guidelines:
1. Base your response ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Be concise but comprehensive
4. Use natural language, not robotic responses
5. If citing sources, use [1], [2], etc.
6. Never make up information not in the context"""

    async def execute(self, state: AgentState) -> GeneratorOutput:
        """
        Execute the response generation logic.

        1. Formats the retrieved context chunks into a structured string.
        2. Constructs a RAG-specific prompt using the system prompt and context.
        3. Calls the LLM to generate a grounded response.
        4. Extracts citations and estimates confidence based on context coverage.

        Args:
            state: AgentState containing the user query and retrieved context chunks.

        Returns:
            GeneratorOutput containing the response, citations, and confidence score.
        """
        query = state.query
        chunks = state.context.get("chunks", [])

        if not chunks:
            return GeneratorOutput(
                response="I don't have enough information to answer this question.",
                confidence=0.0,
                reasoning="No context available",
            )

        # Build context with optional citations
        context_text = self._build_context(chunks)

        # Generate response
        response = await self._generate_response(query, context_text, chunks)

        return response

    def _build_context(self, chunks: list[Chunk]) -> str:
        """
        Build context string from chunks.

        Args:
            chunks: Context chunks.

        Returns:
            Formatted context string.
        """
        parts = []

        for i, chunk in enumerate(chunks):
            if self.include_citations:
                parts.append(f"[Source {i + 1}]\n{chunk.content}")
            else:
                parts.append(chunk.content)

        return "\n\n---\n\n".join(parts)

    async def _generate_response(
        self,
        query: str,
        context: str,
        chunks: list[Chunk],
    ) -> GeneratorOutput:
        """
        Generate the actual response.

        Args:
            query: User query.
            context: Formatted context.
            chunks: Original chunks.

        Returns:
            Generator output.
        """
        citation_instruction = ""
        if self.include_citations:
            citation_instruction = """
When referencing information from the sources, cite using [1], [2], etc.
"""

        prompt = f"""{self._system_prompt}
{citation_instruction}

Context:
{context}

User Question: {query}

Please provide a helpful, accurate response based on the context above."""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
        )

        # Extract citations from response
        citations = self._extract_citations(response, len(chunks))

        # Estimate confidence based on context coverage
        confidence = min(len(chunks) / 5, 1.0) * 0.8

        return GeneratorOutput(
            response=response,
            citations=citations,
            confidence=confidence,
            reasoning=f"Generated from {len(chunks)} context chunks",
        )

    def _extract_citations(self, response: str, num_chunks: int) -> list[int]:
        """
        Extract citation indices from response.

        Args:
            response: Generated response.
            num_chunks: Total number of chunks.

        Returns:
            List of cited chunk indices.
        """
        import re

        citations = set()
        # Match [1], [2], etc.
        matches = re.findall(r"\[(\d+)\]", response)

        for match in matches:
            idx = int(match) - 1  # Convert to 0-indexed
            if 0 <= idx < num_chunks:
                citations.add(idx)

        return sorted(citations)

    async def generate_with_format(
        self,
        query: str,
        chunks: list[Chunk],
        format_type: str = "paragraph",
    ) -> GeneratorOutput:
        """
        Generate response in specific format.

        Args:
            query: User query.
            chunks: Context chunks.
            format_type: Output format (paragraph, bullets, table, step_by_step).

        Returns:
            Formatted response.
        """
        context = self._build_context(chunks)

        format_instructions = {
            "paragraph": "Write your response in clear paragraphs.",
            "bullets": "Format your response as bullet points.",
            "table": "Format your response as a markdown table if appropriate.",
            "step_by_step": "Provide a numbered step-by-step response.",
        }

        instruction = format_instructions.get(format_type, format_instructions["paragraph"])

        prompt = f"""{self._system_prompt}

{instruction}

Context:
{context}

Question: {query}"""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
        )

        return GeneratorOutput(
            response=response,
            citations=self._extract_citations(response, len(chunks)),
            confidence=0.7,
            reasoning=f"Generated in {format_type} format",
        )

    async def generate_with_follow_ups(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> GeneratorOutput:
        """
        Generate response with follow-up question suggestions.

        Args:
            query: User query.
            chunks: Context chunks.

        Returns:
            Response with follow-up questions.
        """
        context = self._build_context(chunks)

        prompt = f"""{self._system_prompt}

Context:
{context}

Question: {query}

After your response, suggest 2-3 relevant follow-up questions the user might ask.
Format follow-up questions on separate lines starting with "Follow-up: """

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1200,
        )

        # Extract follow-up questions
        follow_ups = []
        lines = response.split("\n")
        main_response_lines = []

        for line in lines:
            if line.strip().lower().startswith("follow-up:"):
                follow_up = line.split(":", 1)[1].strip()
                follow_ups.append(follow_up)
            else:
                main_response_lines.append(line)

        main_response = "\n".join(main_response_lines).strip()

        return GeneratorOutput(
            response=main_response,
            citations=self._extract_citations(main_response, len(chunks)),
            confidence=0.7,
            reasoning="Generated with follow-up suggestions",
            follow_up_questions=follow_ups,
        )


class StreamingGeneratorAgent(GeneratorAgent):
    """
    Generator agent with streaming support.
    """

    async def stream_response(
        self,
        query: str,
        chunks: list[Chunk],
    ):
        """
        Stream response tokens.

        Args:
            query: User query.
            chunks: Context chunks.

        Yields:
            Response tokens.
        """
        context = self._build_context(chunks)

        prompt = f"""{self._system_prompt}

Context:
{context}

Question: {query}"""

        # Use streaming if generator supports it
        if hasattr(self._generator, "stream_text"):
            async for token in self._generator.stream_text(prompt):
                yield token
        else:
            # Fallback to non-streaming
            response = await self._generator.generate_text(prompt)
            yield response


class ChainOfThoughtGenerator(GeneratorAgent):
    """
    Generator that uses chain-of-thought reasoning.
    """

    async def execute(self, state: AgentState) -> GeneratorOutput:
        """
        Generate with chain-of-thought.

        Args:
            state: Agent state.

        Returns:
            Generated response.
        """
        query = state.query
        chunks = state.context.get("chunks", [])

        if not chunks:
            return GeneratorOutput(
                response="I don't have enough information to answer this question.",
                confidence=0.0,
                reasoning="No context available",
            )

        context = self._build_context(chunks)

        prompt = f"""{self._system_prompt}

Use chain-of-thought reasoning to answer the question.
First, think through the problem step by step.
Then, provide your final answer.

Context:
{context}

Question: {query}

Let me think through this step by step:"""

        response = await self._generator.generate_text(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1500,
        )

        # Extract reasoning and final answer
        reasoning = ""
        final_answer = response

        if "Therefore" in response or "In conclusion" in response:
            parts = response.split("Therefore", 1)
            if len(parts) == 2:
                reasoning = parts[0].strip()
                final_answer = "Therefore" + parts[1]

        return GeneratorOutput(
            response=final_answer,
            citations=self._extract_citations(response, len(chunks)),
            confidence=0.8,
            reasoning=reasoning[:500] if reasoning else "Chain-of-thought reasoning applied",
        )
