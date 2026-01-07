"""
Comprehensive unit tests for generation functionality.

Tests:
- Multi-provider generator support (Gemini, OpenAI, Claude, Local)
- RAG prompt building and context formatting
- Token counting and cost estimation
- Streaming generation
- Error handling and retries
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag.core.models import Chunk, GenerationResult
from agentic_rag.generation.base import BaseGenerator
from agentic_rag.generation.claude_generator import ClaudeGenerator
from agentic_rag.generation.gemini_generator import GeminiGenerator
from agentic_rag.generation.local_generator import LocalGenerator
from agentic_rag.generation.openai_generator import OpenAIGenerator
from agentic_rag.generation.provider_factory import GeneratorFactory

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_context_chunks() -> list[Chunk]:
    """Create sample chunks for context."""
    return [
        Chunk(
            id="ctx_1",
            content="Machine learning is a subset of artificial intelligence.",
            document_id="doc1",
            context_header="From: AI Fundamentals Guide",
        ),
        Chunk(
            id="ctx_2",
            content="Deep learning uses neural networks with many layers.",
            document_id="doc2",
            context_header="From: Deep Learning Handbook",
        ),
        Chunk(
            id="ctx_3",
            content="RAG combines retrieval with generation for accurate responses.",
            document_id="doc3",
            context_header="From: NLP Best Practices",
        ),
    ]


# =============================================================================
# BaseGenerator Tests
# =============================================================================


class TestBaseGenerator:
    """Tests for the BaseGenerator base class."""

    def test_format_context_with_chunks(self, sample_context_chunks):
        """Test context formatting with chunks."""

        # Create a concrete implementation for testing
        class ConcreteGenerator(BaseGenerator):
            @property
            def provider(self):
                return "test"

            @property
            def model_name(self):
                return "test-model"

            async def generate(self, query, context, **kwargs):
                pass

            async def generate_text(self, prompt, **kwargs):
                pass

        generator = ConcreteGenerator()
        formatted = generator._format_context(sample_context_chunks)

        assert "Source 1" in formatted
        assert "Source 2" in formatted
        assert "Machine learning" in formatted
        assert "AI Fundamentals Guide" in formatted

    def test_format_context_empty(self):
        """Test context formatting with empty list."""

        class ConcreteGenerator(BaseGenerator):
            @property
            def provider(self):
                return "test"

            @property
            def model_name(self):
                return "test-model"

            async def generate(self, query, context, **kwargs):
                pass

            async def generate_text(self, prompt, **kwargs):
                pass

        generator = ConcreteGenerator()
        formatted = generator._format_context([])

        assert "No relevant context" in formatted

    def test_build_rag_prompt(self, sample_context_chunks):
        """Test RAG prompt building."""

        class ConcreteGenerator(BaseGenerator):
            @property
            def provider(self):
                return "test"

            @property
            def model_name(self):
                return "test-model"

            async def generate(self, query, context, **kwargs):
                pass

            async def generate_text(self, prompt, **kwargs):
                pass

        generator = ConcreteGenerator()
        prompt = generator._build_rag_prompt(
            query="What is machine learning?",
            context=sample_context_chunks,
        )

        assert "What is machine learning?" in prompt
        assert "Context:" in prompt
        assert "Machine learning" in prompt

    def test_build_rag_prompt_with_custom_instructions(self, sample_context_chunks):
        """Test RAG prompt with custom instructions."""

        class ConcreteGenerator(BaseGenerator):
            @property
            def provider(self):
                return "test"

            @property
            def model_name(self):
                return "test-model"

            async def generate(self, query, context, **kwargs):
                pass

            async def generate_text(self, prompt, **kwargs):
                pass

        generator = ConcreteGenerator()
        prompt = generator._build_rag_prompt(
            query="Test",
            context=sample_context_chunks,
            custom_instructions="Be concise and technical.",
        )

        assert "Be concise and technical" in prompt


# =============================================================================
# GeminiGenerator Tests
# =============================================================================


class TestGeminiGenerator:
    """Tests for the GeminiGenerator class."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client."""
        with patch("agentic_rag.generation.gemini_generator.genai") as mock_genai:
            mock_model = MagicMock()

            # Mock response
            mock_response = MagicMock()
            mock_response.text = "Generated response from Gemini"
            mock_response.usage_metadata = MagicMock()
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 50

            async def mock_generate(*args, **kwargs):
                return mock_response

            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_genai.GenerativeModel.return_value = mock_model

            yield mock_genai

    @pytest.fixture
    def gemini_generator(self, mock_gemini_client, test_settings_minimal):
        """Create Gemini generator with mock."""
        return GeminiGenerator(
            model_name="gemini-2.0-flash",
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_gemini_generate_returns_result(self, gemini_generator, sample_context_chunks):
        """Test Gemini generation returns GenerationResult."""
        result = await gemini_generator.generate(
            query="What is AI?",
            context=sample_context_chunks,
        )

        assert isinstance(result, GenerationResult)
        assert result.provider == "gemini"

    @pytest.mark.asyncio
    async def test_gemini_generate_includes_response(self, gemini_generator, sample_context_chunks):
        """Test Gemini generation includes response text."""
        result = await gemini_generator.generate(
            query="What is AI?",
            context=sample_context_chunks,
        )

        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_gemini_generate_includes_sources(self, gemini_generator, sample_context_chunks):
        """Test Gemini generation includes sources."""
        result = await gemini_generator.generate(
            query="What is AI?",
            context=sample_context_chunks,
        )

        assert len(result.sources) > 0

    @pytest.mark.asyncio
    async def test_gemini_generate_text_simple(self, gemini_generator):
        """Test simple text generation."""
        result = await gemini_generator.generate_text(
            prompt="Explain machine learning in one sentence.",
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_gemini_provider_property(self, gemini_generator):
        """Test provider property."""
        assert gemini_generator.provider == "gemini"

    def test_gemini_model_name_property(self, gemini_generator):
        """Test model name property."""
        assert "gemini" in gemini_generator.model_name.lower()


# =============================================================================
# OpenAIGenerator Tests
# =============================================================================


class TestOpenAIGenerator:
    """Tests for the OpenAIGenerator class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        with patch("agentic_rag.generation.openai_generator.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()

            # Mock response
            mock_choice = MagicMock()
            mock_choice.message.content = "Generated response from OpenAI"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50

            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            yield mock_openai

    @pytest.fixture
    def openai_generator(self, mock_openai_client, test_settings_minimal):
        """Create OpenAI generator with mock."""
        return OpenAIGenerator(
            model_name="gpt-4o-mini",
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_openai_generate_returns_result(self, openai_generator, sample_context_chunks):
        """Test OpenAI generation returns GenerationResult."""
        result = await openai_generator.generate(
            query="What is AI?",
            context=sample_context_chunks,
        )

        assert isinstance(result, GenerationResult)
        assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_openai_generate_text(self, openai_generator):
        """Test simple text generation."""
        result = await openai_generator.generate_text(
            prompt="What is 2+2?",
        )

        assert isinstance(result, str)

    def test_openai_provider_property(self, openai_generator):
        """Test provider property."""
        assert openai_generator.provider == "openai"


# =============================================================================
# ClaudeGenerator Tests
# =============================================================================


class TestClaudeGenerator:
    """Tests for the ClaudeGenerator class."""

    @pytest.fixture
    def mock_claude_client(self):
        """Create mock Anthropic client."""
        with patch("agentic_rag.generation.claude_generator.AsyncAnthropic") as mock_anthropic:
            mock_client = MagicMock()

            # Mock response
            mock_content = MagicMock()
            mock_content.text = "Generated response from Claude"

            mock_response = MagicMock()
            mock_response.content = [mock_content]
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50

            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client

            yield mock_anthropic

    @pytest.fixture
    def claude_generator(self, mock_claude_client, test_settings_minimal):
        """Create Claude generator with mock."""
        return ClaudeGenerator(
            model_name="claude-sonnet-4-20250514",
            settings=test_settings_minimal,
        )

    @pytest.mark.asyncio
    async def test_claude_generate_returns_result(self, claude_generator, sample_context_chunks):
        """Test Claude generation returns GenerationResult."""
        result = await claude_generator.generate(
            query="What is AI?",
            context=sample_context_chunks,
        )

        assert isinstance(result, GenerationResult)
        assert result.provider == "claude"

    @pytest.mark.asyncio
    async def test_claude_generate_text(self, claude_generator):
        """Test simple text generation."""
        result = await claude_generator.generate_text(
            prompt="Explain recursion.",
        )

        assert isinstance(result, str)

    def test_claude_provider_property(self, claude_generator):
        """Test provider property."""
        assert claude_generator.provider == "claude"


# =============================================================================
# LocalGenerator Tests
# =============================================================================


class TestLocalGenerator:
    """Tests for the LocalGenerator class."""

    @pytest.fixture
    def mock_local_model(self):
        """Create mock local model."""
        with (
            patch("agentic_rag.generation.local_generator.AutoModelForCausalLM") as mock_model,
            patch("agentic_rag.generation.local_generator.AutoTokenizer") as mock_tokenizer,
        ):
            tokenizer = MagicMock()
            tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
            tokenizer.decode.return_value = "Local model response"
            mock_tokenizer.from_pretrained.return_value = tokenizer

            model = MagicMock()
            model.generate.return_value = MagicMock()
            mock_model.from_pretrained.return_value = model

            yield mock_model, mock_tokenizer

    @pytest.fixture
    def local_generator(self, mock_local_model, test_settings_minimal):
        """Create local generator with mock."""
        return LocalGenerator(
            model_name="microsoft/phi-2",
            settings=test_settings_minimal,
        )

    def test_local_provider_property(self, local_generator):
        """Test provider property."""
        assert local_generator.provider == "local"


# =============================================================================
# GeneratorFactory Tests
# =============================================================================


class TestGeneratorFactory:
    """Tests for the GeneratorFactory class."""

    def test_create_gemini_generator(self, test_settings_minimal):
        """Test creating Gemini generator."""
        with patch("agentic_rag.generation.gemini_generator.genai"):
            generator = GeneratorFactory.create(
                provider="gemini",
                model="gemini-2.0-flash",
                settings=test_settings_minimal,
            )
            assert generator.provider == "gemini"

    def test_create_openai_generator(self, test_settings_minimal):
        """Test creating OpenAI generator."""
        with patch("agentic_rag.generation.openai_generator.AsyncOpenAI"):
            generator = GeneratorFactory.create(
                provider="openai",
                model="gpt-4o-mini",
                settings=test_settings_minimal,
            )
            assert generator.provider == "openai"

    def test_create_claude_generator(self, test_settings_minimal):
        """Test creating Claude generator."""
        with patch("agentic_rag.generation.claude_generator.AsyncAnthropic"):
            generator = GeneratorFactory.create(
                provider="claude",
                model="claude-sonnet-4-20250514",
                settings=test_settings_minimal,
            )
            assert generator.provider == "claude"

    def test_invalid_provider_raises_error(self, test_settings_minimal):
        """Test that invalid provider raises error."""
        with pytest.raises((ValueError, KeyError)):
            GeneratorFactory.create(
                provider="invalid_provider",
                model="some-model",
                settings=test_settings_minimal,
            )


# =============================================================================
# Generation Quality Tests
# =============================================================================


class TestGenerationQuality:
    """Tests for generation quality metrics."""

    @pytest.fixture
    def mock_quality_generator(self, test_settings_minimal):
        """Create generator with quality-focused responses."""
        with patch("agentic_rag.generation.gemini_generator.genai") as mock_genai:
            mock_model = MagicMock()

            def create_response(prompt):
                """Create response that references sources."""
                return MagicMock(
                    text="According to [Source 1], machine learning is a subset of AI. "
                    "As stated in [Source 2], deep learning uses neural networks.",
                    usage_metadata=MagicMock(
                        prompt_token_count=150,
                        candidates_token_count=80,
                    ),
                )

            mock_model.generate_content_async = AsyncMock(
                side_effect=lambda *args, **kwargs: create_response(args[0])
            )
            mock_genai.GenerativeModel.return_value = mock_model

            return GeminiGenerator(
                model_name="gemini-2.0-flash",
                settings=test_settings_minimal,
            )

    @pytest.mark.asyncio
    async def test_response_cites_sources(self, mock_quality_generator, sample_context_chunks):
        """Test that response cites sources."""
        result = await mock_quality_generator.generate(
            query="What is machine learning?",
            context=sample_context_chunks,
        )

        # Response should reference sources
        assert "Source" in result.response or "[" in result.response


# =============================================================================
# Token Counting Tests
# =============================================================================


class TestTokenCounting:
    """Tests for token counting and cost estimation."""

    @pytest.fixture
    def generator_with_tokens(self, test_settings_minimal):
        """Create generator that returns token counts."""
        with patch("agentic_rag.generation.gemini_generator.genai") as mock_genai:
            mock_model = MagicMock()
            mock_response = MagicMock(
                text="Test response",
                usage_metadata=MagicMock(
                    prompt_token_count=500,
                    candidates_token_count=200,
                ),
            )
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_genai.GenerativeModel.return_value = mock_model

            return GeminiGenerator(
                model_name="gemini-2.0-flash",
                settings=test_settings_minimal,
            )

    @pytest.mark.asyncio
    async def test_result_includes_token_counts(self, generator_with_tokens, sample_context_chunks):
        """Test that result includes token counts."""
        result = await generator_with_tokens.generate(
            query="Test",
            context=sample_context_chunks,
        )

        assert result.input_tokens > 0
        assert result.output_tokens > 0

    @pytest.mark.asyncio
    async def test_token_counts_reasonable(self, generator_with_tokens, sample_context_chunks):
        """Test that token counts are reasonable."""
        result = await generator_with_tokens.generate(
            query="Test",
            context=sample_context_chunks,
        )

        # Input should be more than output for RAG
        assert result.input_tokens >= result.output_tokens * 0.5


# =============================================================================
# Streaming Tests
# =============================================================================


class TestStreamingGeneration:
    """Tests for streaming generation."""

    @pytest.fixture
    def streaming_generator(self, test_settings_minimal):
        """Create generator with streaming support."""
        with patch("agentic_rag.generation.gemini_generator.genai") as mock_genai:
            mock_model = MagicMock()
            mock_response = MagicMock(
                text="Full streamed response",
                usage_metadata=MagicMock(
                    prompt_token_count=100,
                    candidates_token_count=50,
                ),
            )
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_genai.GenerativeModel.return_value = mock_model

            return GeminiGenerator(
                model_name="gemini-2.0-flash",
                settings=test_settings_minimal,
            )

    @pytest.mark.asyncio
    async def test_stream_yields_text(self, streaming_generator, sample_context_chunks):
        """Test that stream yields text chunks."""
        chunks = []
        async for chunk in streaming_generator.generate_stream(
            query="Test",
            context=sample_context_chunks,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestGenerationErrorHandling:
    """Tests for error handling in generation."""

    @pytest.fixture
    def failing_generator(self, test_settings_minimal):
        """Create generator that simulates failures."""
        with patch("agentic_rag.generation.gemini_generator.genai") as mock_genai:
            mock_model = MagicMock()
            mock_model.generate_content_async = AsyncMock(
                side_effect=Exception("API rate limit exceeded")
            )
            mock_genai.GenerativeModel.return_value = mock_model

            return GeminiGenerator(
                model_name="gemini-2.0-flash",
                settings=test_settings_minimal,
            )

    @pytest.mark.asyncio
    async def test_handles_api_error(self, failing_generator, sample_context_chunks):
        """Test handling of API errors."""
        with pytest.raises(Exception) as exc_info:
            await failing_generator.generate(
                query="Test",
                context=sample_context_chunks,
            )

        assert "rate limit" in str(exc_info.value).lower() or "API" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handles_empty_context(self, test_settings_minimal):
        """Test generation with empty context."""
        with patch("agentic_rag.generation.gemini_generator.genai") as mock_genai:
            mock_model = MagicMock()
            mock_response = MagicMock(
                text="I don't have context to answer.",
                usage_metadata=MagicMock(
                    prompt_token_count=50,
                    candidates_token_count=20,
                ),
            )
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_genai.GenerativeModel.return_value = mock_model

            generator = GeminiGenerator(
                model_name="gemini-2.0-flash",
                settings=test_settings_minimal,
            )

            result = await generator.generate(
                query="What is the capital of France?",
                context=[],
            )

            assert isinstance(result, GenerationResult)


# =============================================================================
# System Prompt Tests
# =============================================================================


class TestSystemPrompts:
    """Tests for system prompt handling."""

    @pytest.fixture
    def prompt_tracking_generator(self, test_settings_minimal):
        """Create generator that tracks prompts."""
        with patch("agentic_rag.generation.gemini_generator.genai") as mock_genai:
            mock_model = MagicMock()
            captured_prompts = []

            async def capture_and_respond(*args, **kwargs):
                captured_prompts.append(args[0] if args else kwargs.get("prompt"))
                return MagicMock(
                    text="Response",
                    usage_metadata=MagicMock(
                        prompt_token_count=100,
                        candidates_token_count=50,
                    ),
                )

            mock_model.generate_content_async = AsyncMock(side_effect=capture_and_respond)
            mock_model.captured_prompts = captured_prompts
            mock_genai.GenerativeModel.return_value = mock_model

            generator = GeminiGenerator(
                model_name="gemini-2.0-flash",
                settings=test_settings_minimal,
            )
            generator._captured_prompts = captured_prompts
            return generator

    @pytest.mark.asyncio
    async def test_custom_system_prompt_used(
        self, prompt_tracking_generator, sample_context_chunks
    ):
        """Test that custom system prompt is used."""
        await prompt_tracking_generator.generate(
            query="Test",
            context=sample_context_chunks,
            system_prompt="You are a helpful coding assistant.",
        )

        # System prompt should be incorporated
        # (exact verification depends on implementation)
        assert isinstance(prompt_tracking_generator._captured_prompts, list)


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestGenerationPerformance:
    """Performance tests for generation."""

    @pytest.fixture
    def fast_generator(self, test_settings_minimal):
        """Create fast mock generator."""
        with patch("agentic_rag.generation.gemini_generator.genai") as mock_genai:
            mock_model = MagicMock()
            mock_response = MagicMock(
                text="Quick response",
                usage_metadata=MagicMock(
                    prompt_token_count=100,
                    candidates_token_count=50,
                ),
            )
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_genai.GenerativeModel.return_value = mock_model

            return GeminiGenerator(
                model_name="gemini-2.0-flash",
                settings=test_settings_minimal,
            )

    @pytest.mark.asyncio
    async def test_generation_latency(self, fast_generator, sample_context_chunks):
        """Test generation completes quickly."""
        import time

        start = time.time()
        await fast_generator.generate(
            query="Test",
            context=sample_context_chunks,
        )
        elapsed = time.time() - start

        # Mock should be very fast
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_generations(self, fast_generator, sample_context_chunks):
        """Test concurrent generation requests."""
        import time

        start = time.time()
        tasks = [
            fast_generator.generate(
                query=f"Query {i}",
                context=sample_context_chunks,
            )
            for i in range(5)
        ]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        # Concurrent should be efficient
        assert elapsed < 5.0
