# Response Generation

> **Multi-Provider LLM Integration for RAG**
>
> This document covers LLM providers (Claude, OpenAI, Gemini, Local), prompt templates, and generation strategies.

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Providers](#supported-providers)
3. [Prompt Templates](#prompt-templates)
4. [Generation Strategies](#generation-strategies)
5. [Configuration](#configuration)

---

## Overview

The generation module handles the final step of RAG: synthesizing a response from retrieved context using an LLM.

### Generation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Generation Pipeline                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Inputs:                                                             │
│  • Query: "What is the Transformer architecture?"                   │
│  • Context: [Chunk1, Chunk2, Chunk3, ...]                          │
│  • System Prompt: (optional override)                               │
└───────────────────────────────────────┬─────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. Format Context                                                   │
│     [Source 1]                                                      │
│     Content from chunk 1...                                         │
│     ---                                                             │
│     [Source 2]                                                      │
│     Content from chunk 2...                                         │
└───────────────────────────────────────┬─────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Build RAG Prompt                                                 │
│     System: RAG guidelines, citation instructions                   │
│     User: Context + Question                                        │
└───────────────────────────────────────┬─────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. LLM Generation                                                   │
│     Provider: Claude / OpenAI / Gemini / Local                      │
└───────────────────────────────────────┬─────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Output:                                                             │
│  • Response text                                                    │
│  • Source citations                                                 │
│  • Token usage                                                      │
│  • Latency metrics                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Base Generator Protocol

```python
class BaseGenerator(ABC):
    """Abstract base class for LLM generators."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the LLM provider (claude, openai, gemini, local)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: list[Chunk],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response given query and context."""
        ...

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Simple text generation (for HyDE, query expansion, etc.)."""
        ...
```

---

## Supported Providers

### Claude (Anthropic)

| Model | Use Case | Context | Best For |
|-------|----------|---------|----------|
| `claude-sonnet-4-5-20250929` | Latest | 200K | Highest quality RAG |
| `claude-3-5-sonnet-20241022` | Previous | 200K | General RAG |
| `claude-3-5-haiku-20241022` | Fast | 200K | Speed + low cost |

```python
from agentic_rag.generation import ClaudeGenerator

generator = ClaudeGenerator(
    model="claude-sonnet-4-5-20250929",
    temperature=0.3,
    max_tokens=4096,
)
```

### OpenAI

| Model | Use Case | Context | Best For |
|-------|----------|---------|----------|
| `gpt-4o` | Latest | 128K | General purpose |
| `gpt-4o-mini` | Fast | 128K | Cost efficiency |
| `o1-preview` | Reasoning | 128K | Complex analysis |

```python
from agentic_rag.generation import OpenAIGenerator

generator = OpenAIGenerator(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=4096,
)
```

### Gemini (Google)

| Model | Use Case | Context | Best For |
|-------|----------|---------|----------|
| `gemini-2.0-flash` | Latest | 1M | Best speed/quality |
| `gemini-1.5-pro` | Quality | 2M | Complex tasks |
| `gemini-1.5-flash` | Fast | 1M | Speed + cost |

```python
from agentic_rag.generation import GeminiGenerator

generator = GeminiGenerator(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=4096,
)
```

### Local (Ollama)

| Model | Size | Best For |
|-------|------|----------|
| `qwen2.5:7b` | 7B | Fast local inference |
| `qwen2.5:14b` | 14B | Better quality |
| `llama3.3:70b` | 70B | Highest local quality |
| `mistral:7b` | 7B | Efficient |

```python
from agentic_rag.generation import LocalGenerator

generator = LocalGenerator(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
)
```

### Generator Factory

```python
from agentic_rag.generation import create_generator, GeneratorFactory

# Auto-detect from settings
generator = create_generator()

# Specify provider
generator = create_generator("gemini", model="gemini-2.5-flash")

# List available providers
providers = GeneratorFactory.list_providers()
# ['claude', 'openai', 'gemini', 'local']

# Get recommended models
models = GeneratorFactory.get_default_models("gemini")
# ['gemini-2.5-pro', 'gemini-2.5-flash', ...]
```

---

## Prompt Templates

### RAG System Prompt

```python
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

Guidelines:
1. Base your answers ONLY on the information provided in the context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer that"
3. Cite your sources using [Source N] notation
4. Be concise but comprehensive
5. If you're uncertain about something, express that uncertainty

Never make up information that isn't in the context."""
```

### RAG User Template

```python
RAG_USER_TEMPLATE = """Context:
{context}

---

Question: {query}

Please provide a comprehensive answer based on the context above. Cite your sources."""
```

### HyDE Template

For generating hypothetical documents:

```python
HYDE_SYSTEM_PROMPT = """You are an AI assistant that generates detailed, authoritative passages.

Your task is to write a comprehensive passage that would perfectly answer the given question.
Write as if you are an expert source document, not as an AI assistant.

Guidelines:
1. Write in a formal, authoritative tone
2. Include specific details and technical information
3. Structure the passage logically
4. Do NOT say "I" or address the reader - write as a reference document
5. Aim for 100-200 words"""

HYDE_USER_TEMPLATE = """Write a detailed passage that would perfectly answer this question:

{query}

Write as an authoritative source document:"""
```

### Query Expansion Template

For generating query variations:

```python
QUERY_EXPANSION_SYSTEM_PROMPT = """You are an AI assistant that helps improve search queries.

Your task is to generate alternative phrasings of the given query that might retrieve different relevant information.

Guidelines:
1. Generate 3-5 alternative queries
2. Use different vocabulary and phrasing
3. Consider different aspects of the question
4. Keep queries focused and specific
5. Output one query per line"""
```

### Self-RAG Evaluation Template

```python
SELF_RAG_EVALUATION_TEMPLATE = """Evaluate this RAG response:

Query: {query}

Retrieved Context:
{context}

Generated Response:
{response}

Evaluate:
1. ISREL: Is the retrieved context relevant to the query?
2. ISSUP: Is the response supported by the context?
3. ISUSE: Is the response useful for answering the query?

For each, answer YES or NO with a brief explanation."""
```

### Using Templates

```python
from agentic_rag.generation.prompt_templates import get_template, TEMPLATES

# Get a specific template
template = get_template("rag_default")

# Format user prompt
user_prompt = template.format_user(
    context="[Source 1]\nTransformers use attention...",
    query="What is the Transformer architecture?"
)

# List available templates
available = list(TEMPLATES.keys())
# ['rag_default', 'rag_analytical', 'hyde', 'query_expansion', 'self_rag']
```

---

## Generation Strategies

### Standard RAG Generation

```python
async def generate(
    self,
    query: str,
    context: list[Chunk],
    system_prompt: str | None = None,
) -> GenerationResult:
    # Format context
    context_str = self._format_context(context)

    # Build prompt
    prompt = self._build_rag_prompt(query, context)

    # Generate
    response = await self._llm_call(
        system_prompt or RAG_SYSTEM_PROMPT,
        prompt,
    )

    return GenerationResult(
        response=response,
        sources=context,
        provider=self.provider,
        model=self.model_name,
    )
```

### Streaming Generation

```python
async def generate_stream(
    self,
    query: str,
    context: list[Chunk],
) -> AsyncIterator[str]:
    """Stream tokens as they're generated."""
    prompt = self._build_rag_prompt(query, context)

    async for chunk in self._stream_llm_call(prompt):
        yield chunk
```

### Multi-Query Generation

Generate responses from multiple query perspectives:

```python
async def multi_query_generate(
    query: str,
    context: list[Chunk],
    num_queries: int = 4,
) -> GenerationResult:
    # Generate query variations
    variations = await generate_query_variations(query, num_queries)

    # Generate intermediate answers
    intermediate_answers = []
    for variation in variations:
        answer = await generate(variation, context)
        intermediate_answers.append(answer)

    # Synthesize final answer
    final = await synthesize_answers(
        query,
        intermediate_answers,
        context
    )

    return final
```

---

## Configuration

### Pipeline Builder

```python
from agentic_rag.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .with_generator(
        provider="claude",
        model="claude-sonnet-4-5-20250929",
        temperature=0.3,
        max_tokens=4096,
    )
    .build()
)
```

### Environment Variables

```bash
# Claude
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
OPENAI_API_KEY=sk-...

# Gemini
GOOGLE_API_KEY=...

# Local (Ollama)
OLLAMA_BASE_URL=http://localhost:11434

# Default provider
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=4096
```

### Generation Result Model

```python
class GenerationResult(BaseModel):
    """Result from LLM generation."""

    response: str
    sources: list[Chunk] = []
    confidence: float = 0.0

    # Token usage for cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Provider info
    provider: str = ""
    model: str = ""

    # Performance
    finish_reason: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = {}
```

### Cost Estimation

| Provider | Model | Input (per 1M) | Output (per 1M) |
|----------|-------|----------------|-----------------|
| Claude | Sonnet 4 | $3.00 | $15.00 |
| Claude | Opus 4 | $15.00 | $75.00 |
| OpenAI | GPT-4o | $2.50 | $10.00 |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 |
| Gemini | 2.5 Flash | $0.075 | $0.30 |
| Gemini | 2.5 Pro | $1.25 | $5.00 |
| Local | Any | Free | Free |

---

## Best Practices

### 1. Temperature Settings

| Task | Temperature | Reason |
|------|-------------|--------|
| Factual QA | 0.0-0.3 | Deterministic, accurate |
| Analysis | 0.3-0.5 | Some creativity |
| Creative | 0.7-1.0 | More varied |
| HyDE | 0.7 | Diverse hypotheticals |

### 2. Context Formatting

```python
def _format_context(self, chunks: list[Chunk]) -> str:
    """Format chunks for LLM consumption."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Include context header if available
        header = chunk.context_header or ""
        if header:
            parts.append(f"[Source {i}]\n{header}\n{chunk.content}")
        else:
            parts.append(f"[Source {i}]\n{chunk.content}")

    return "\n\n---\n\n".join(parts)
```

### 3. Citation Instructions

Always include citation count in prompts:

```python
f"You have exactly {num_sources} sources available. "
f"Only cite sources that exist ([Source 1] through [Source {num_sources}])."
```

### 4. Error Handling

```python
try:
    result = await generator.generate(query, context)
except RateLimitError:
    # Retry with backoff
    await asyncio.sleep(exponential_backoff)
    result = await generator.generate(query, context)
except APIError as e:
    # Log and return graceful error
    return GenerationResult(
        response="I encountered an error generating the response.",
        metadata={"error": str(e)},
    )
```

---

## References

1. Anthropic. "Claude API Documentation." [docs.anthropic.com](https://docs.anthropic.com)

2. OpenAI. "API Reference." [platform.openai.com/docs](https://platform.openai.com/docs)

3. Google AI. "Gemini API Documentation." [ai.google.dev](https://ai.google.dev)

4. Ollama. "Running LLMs Locally." [ollama.ai](https://ollama.ai)

