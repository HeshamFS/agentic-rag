"""
Prompt templates for RAG generation.

Provider-agnostic templates that work across
Claude, OpenAI, Gemini, and local models.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptTemplate:
    """A named prompt template."""

    name: str
    system_prompt: str
    user_template: str

    def format_user(self, **kwargs: Any) -> str:
        """Format the user prompt with variables."""
        return self.user_template.format(**kwargs)


# =============================================================================
# System Prompts
# =============================================================================

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

Guidelines:
1. Base your answers ONLY on the information provided in the context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer that"
3. Cite your sources using [Source N] notation
4. Be concise but comprehensive
5. If you're uncertain about something, express that uncertainty

Never make up information that isn't in the context."""


ANALYTICAL_SYSTEM_PROMPT = """You are an analytical AI assistant that provides in-depth analysis based on provided context.

Guidelines:
1. Analyze the information systematically
2. Identify patterns, relationships, and key insights
3. Support conclusions with evidence from the context
4. Consider multiple perspectives when applicable
5. Cite sources using [Source N] notation
6. Clearly distinguish between what's in the context and logical inferences

Be thorough in your analysis while remaining grounded in the provided information."""


HYDE_SYSTEM_PROMPT = """You are an AI assistant that generates detailed, authoritative passages.

Your task is to write a comprehensive passage that would perfectly answer the given question.
Write as if you are an expert source document, not as an AI assistant.

Guidelines:
1. Write in a formal, authoritative tone
2. Include specific details and technical information
3. Structure the passage logically
4. Do NOT say "I" or address the reader - write as a reference document
5. Aim for 100-200 words"""


QUERY_EXPANSION_SYSTEM_PROMPT = """You are an AI assistant that helps improve search queries.

Your task is to generate alternative phrasings of the given query that might retrieve different relevant information.

Guidelines:
1. Generate 3-5 alternative queries
2. Use different vocabulary and phrasing
3. Consider different aspects of the question
4. Keep queries focused and specific
5. Output one query per line"""


SELF_RAG_SYSTEM_PROMPT = """You are an AI assistant evaluating the quality of RAG responses.

Evaluate the following aspects:
1. ISREL (Relevance): Is the retrieved context relevant to the query?
2. ISSUP (Support): Is the response fully supported by the context?
3. ISUSE (Usefulness): Is the response useful for answering the query?

For each aspect, respond with YES or NO followed by a brief explanation."""


# =============================================================================
# User Prompt Templates
# =============================================================================

RAG_USER_TEMPLATE = """Context:
{context}

---

Question: {query}

Please provide a comprehensive answer based on the context above. Cite your sources."""


HYDE_USER_TEMPLATE = """Write a detailed passage that would perfectly answer this question:

{query}

Write as an authoritative source document:"""


QUERY_EXPANSION_USER_TEMPLATE = """Original query: {query}

Generate 3-5 alternative phrasings of this query that might retrieve different relevant information:"""


CLAIM_EXTRACTION_USER_TEMPLATE = """Extract all factual claims from the following response.
List each claim on a separate line.

Response:
{response}

Claims (one per line):"""


CLAIM_VERIFICATION_USER_TEMPLATE = """Determine if the following claim is supported by the context.

Claim: {claim}

Context:
{context}

Is this claim supported by the context? Answer YES or NO, then explain briefly."""


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


# =============================================================================
# Pre-built Templates
# =============================================================================

TEMPLATES = {
    "rag_default": PromptTemplate(
        name="rag_default",
        system_prompt=RAG_SYSTEM_PROMPT,
        user_template=RAG_USER_TEMPLATE,
    ),
    "rag_analytical": PromptTemplate(
        name="rag_analytical",
        system_prompt=ANALYTICAL_SYSTEM_PROMPT,
        user_template=RAG_USER_TEMPLATE,
    ),
    "hyde": PromptTemplate(
        name="hyde",
        system_prompt=HYDE_SYSTEM_PROMPT,
        user_template=HYDE_USER_TEMPLATE,
    ),
    "query_expansion": PromptTemplate(
        name="query_expansion",
        system_prompt=QUERY_EXPANSION_SYSTEM_PROMPT,
        user_template=QUERY_EXPANSION_USER_TEMPLATE,
    ),
    "self_rag": PromptTemplate(
        name="self_rag",
        system_prompt=SELF_RAG_SYSTEM_PROMPT,
        user_template=SELF_RAG_EVALUATION_TEMPLATE,
    ),
}


def get_template(name: str) -> PromptTemplate:
    """Get a prompt template by name."""
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]
