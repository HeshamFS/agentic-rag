# RAG Evaluation

> **Measuring Quality Without Ground Truth**
>
> This document covers RAGAS metrics, Self-RAG reflection tokens, and evaluation strategies for RAG systems.

---

## Table of Contents

1. [Overview](#overview)
2. [RAGAS Framework](#ragas-framework)
3. [Self-RAG Reflection Tokens](#self-rag-reflection-tokens)
4. [Component-Level Evaluation](#component-level-evaluation)
5. [Configuration](#configuration)

---

## Overview

RAG evaluation is challenging because it involves multiple components (retrieval + generation) and lacks ground truth for most queries.

> **Reference**: Es, S., et al. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)

### Evaluation Challenges

| Challenge | Traditional | RAG-Specific |
|-----------|-------------|--------------|
| Ground truth | Expensive to create | Often unavailable |
| Metrics | BLEU, ROUGE | Faithfulness, relevance |
| Components | Single model | Retriever + Generator |
| Failure modes | Wrong output | Wrong retrieval OR wrong generation |

### Evaluation Dimensions

```
                    ┌──────────────────────────────────────────┐
                    │         RAG Evaluation Dimensions         │
                    └─────────────────────┬────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼                             ▼                             ▼
    ┌───────────────┐            ┌───────────────┐            ┌───────────────┐
    │   Retrieval   │            │  Generation   │            │   End-to-End  │
    │   Quality     │            │   Quality     │            │    Quality    │
    └───────┬───────┘            └───────┬───────┘            └───────┬───────┘
            │                             │                             │
            ▼                             ▼                             ▼
    • Context Precision          • Faithfulness               • Answer Relevancy
    • Context Recall             • No Hallucinations          • Usefulness
    • Context Relevancy          • Citation Accuracy          • Completeness
```

---

## RAGAS Framework

**RAGAS** (Retrieval-Augmented Generation Assessment) provides reference-free metrics for evaluating RAG pipelines.

### Core Metrics

#### 1. Context Precision

**Measures**: Signal-to-noise ratio of retrieved contexts.

$$\text{Context Precision} = \frac{\text{Relevant contexts retrieved}}{\text{Total contexts retrieved}}$$

**High precision**: Few irrelevant contexts retrieved.

```python
class ContextPrecisionEvaluator(BaseEvaluator):
    """Measures proportion of relevant retrieved contexts."""

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
    ) -> EvaluationResult:
        relevant_count = 0
        for chunk in contexts:
            is_relevant = await self._is_context_relevant(query, chunk.content)
            if is_relevant:
                relevant_count += 1

        precision = relevant_count / len(contexts)
        return EvaluationResult(
            metric_name="context_precision",
            score=precision,
        )
```

#### 2. Context Recall

**Measures**: Coverage of ground truth information.

$$\text{Context Recall} = \frac{\text{Ground truth claims in contexts}}{\text{Total ground truth claims}}$$

**High recall**: All needed information was retrieved.

**Note**: Requires ground truth answer.

```python
class ContextRecallEvaluator(BaseEvaluator):
    """Measures coverage of ground truth in retrieved contexts."""

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
        ground_truth: str,  # Required
    ) -> EvaluationResult:
        # Extract claims from ground truth
        claims = await self._extract_claims(ground_truth)

        # Check how many claims are in contexts
        context_text = "\n\n".join(c.content for c in contexts)
        supported_claims = 0
        for claim in claims:
            if await self._is_claim_supported(claim, context_text):
                supported_claims += 1

        recall = supported_claims / len(claims)
        return EvaluationResult(
            metric_name="context_recall",
            score=recall,
        )
```

#### 3. Faithfulness

**Measures**: Factual consistency of response with context.

$$\text{Faithfulness} = \frac{\text{Response claims supported by context}}{\text{Total response claims}}$$

**High faithfulness**: No hallucinations, all claims grounded.

```python
class FaithfulnessEvaluator(BaseEvaluator):
    """Measures response grounding in context (no hallucinations)."""

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
    ) -> EvaluationResult:
        # Extract claims from response
        claims = await self._extract_claims(response)

        # Verify each claim against context
        context_text = "\n\n".join(c.content for c in contexts)
        supported_claims = 0
        hallucinations = []

        for claim in claims:
            verdict = await self._verify_claim(claim, context_text)
            if verdict == "supported":
                supported_claims += 1
            elif verdict == "contradicted":
                hallucinations.append(claim)

        faithfulness = supported_claims / len(claims)
        return EvaluationResult(
            metric_name="faithfulness",
            score=faithfulness,
            details={"hallucinations": hallucinations},
        )
```

**Claim Verification**:

```
Claim: "Transformers use self-attention"
Context: "The Transformer architecture relies on self-attention mechanisms..."

Verdict: SUPPORTED | CONTRADICTED | NEUTRAL
```

#### 4. Answer Relevancy

**Measures**: How well response addresses the query.

Uses reverse question generation:
1. Generate questions the response would answer
2. Measure similarity to original query

$$\text{Answer Relevancy} = \max_{q' \in \text{generated}} \text{sim}(q, q')$$

```python
class AnswerRelevancyEvaluator(BaseEvaluator):
    """Measures how well response addresses the query."""

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
    ) -> EvaluationResult:
        # Generate questions this response answers
        prompt = f"""What question(s) does this response answer?
        Response: "{response}"
        List questions:"""

        generated_questions = await self._generate_questions(prompt)

        # Find best matching question
        max_similarity = 0.0
        for gen_q in generated_questions:
            similarity = await self._calculate_similarity(query, gen_q)
            max_similarity = max(max_similarity, similarity)

        return EvaluationResult(
            metric_name="answer_relevancy",
            score=max_similarity,
        )
```

### RAGAS Score

Combined metric:

$$\text{RAGAS Score} = \frac{1}{4}(\text{Precision} + \text{Recall} + \text{Faithfulness} + \text{Relevancy})$$

Or with custom weights:

```python
def aggregate_scores(results, weights=None):
    if weights is None:
        weights = {
            "context_precision": 0.2,
            "context_recall": 0.2,
            "faithfulness": 0.3,     # Higher weight - hallucinations are bad
            "answer_relevancy": 0.3,
        }

    return sum(results[k].score * w for k, w in weights.items())
```

---

## Self-RAG Reflection Tokens

**Self-RAG** uses reflection tokens for fine-grained quality control.

> **Reference**: Asai, A., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection."

### Reflection Tokens

| Token | Question | Values |
|-------|----------|--------|
| **ISREL** | Is the retrieved context relevant? | FULLY / PARTIALLY / NOT |
| **ISSUP** | Is the response supported by context? | FULLY / PARTIALLY / NOT |
| **ISUSE** | Is the response useful to the user? | FULLY / PARTIALLY / NOT |

### Token Meanings

**ISREL (Relevance)**:
```
FULLY: Context directly addresses the query
PARTIALLY: Some relevant information, incomplete
NOT: Context is unrelated
```

**ISSUP (Support)**:
```
FULLY: Every claim is supported by context
PARTIALLY: Some claims lack support
NOT: Significant unsupported claims (hallucinations)
```

**ISUSE (Usefulness)**:
```
FULLY: Response completely answers the query
PARTIALLY: Response is incomplete or unclear
NOT: Response doesn't help answer the query
```

### Implementation

```python
class SelfRAGEvaluator:
    """Complete Self-RAG evaluation."""

    def __init__(self, generator, regenerate_threshold=0.5):
        self._isrel = IsRelEvaluator(generator)
        self._issup = IsSupEvaluator(generator)
        self._isuse = IsUseEvaluator(generator)
        self.regenerate_threshold = regenerate_threshold

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[Chunk],
    ) -> SelfRAGOutput:
        # Run all evaluations
        isrel_result = await self._isrel.evaluate(query, response, contexts)
        issup_result = await self._issup.evaluate(query, response, contexts)
        isuse_result = await self._isuse.evaluate(query, response, contexts)

        # Calculate overall score
        overall = (
            isrel_result.score +
            issup_result.score +
            isuse_result.score
        ) / 3

        return SelfRAGOutput(
            isrel=isrel_result.details["value"],
            issup=issup_result.details["value"],
            isuse=isuse_result.details["value"],
            overall_score=overall,
            should_regenerate=overall < self.regenerate_threshold,
        )
```

### Regeneration Decision

```
If any token is NOT:
    → Trigger regeneration

If ISREL is NOT:
    → Re-retrieve with query expansion

If ISSUP is NOT:
    → Regenerate with stricter grounding prompt

If ISUSE is NOT:
    → Regenerate with clearer instructions
```

---

## Component-Level Evaluation

### Retrieval Metrics

| Metric | Description | When to Use |
|--------|-------------|-------------|
| Recall@K | Relevant docs in top K | Standard retrieval |
| MRR | Mean reciprocal rank | Single correct answer |
| NDCG | Normalized DCG | Graded relevance |
| Context Precision | RAGAS precision | RAG evaluation |

### Generation Metrics

| Metric | Description | When to Use |
|--------|-------------|-------------|
| Faithfulness | Grounding in context | Hallucination detection |
| Answer Relevancy | Query addressing | Response quality |
| Semantic Similarity | Embedding comparison | Ground truth available |
| Citation Accuracy | Correct source refs | Attribution quality |

### End-to-End Metrics

| Metric | Description | When to Use |
|--------|-------------|-------------|
| RAGAS Score | Combined metric | Overall quality |
| Self-RAG | Reflection tokens | Agentic pipelines |
| Human Eval | Expert ratings | Final validation |

---

## Configuration

### RAGAS Evaluation

```python
from agentic_rag.evaluation import RAGASEvaluator
from agentic_rag.generation import create_generator

# Create evaluator
generator = create_generator("claude", model="claude-sonnet-4-5-20250929")
evaluator = RAGASEvaluator(generator)

# Evaluate
results = await evaluator.evaluate(
    query="What is the Transformer architecture?",
    response="The Transformer uses self-attention...",
    contexts=[chunk1, chunk2, chunk3],
    ground_truth="The Transformer is a neural network..."  # Optional
)

# Access individual metrics
print(f"Context Precision: {results['context_precision'].score}")
print(f"Faithfulness: {results['faithfulness'].score}")
print(f"Answer Relevancy: {results['answer_relevancy'].score}")

# Get aggregate score
ragas_score = evaluator.aggregate_scores(results)
print(f"RAGAS Score: {ragas_score}")
```

### Self-RAG Evaluation

```python
from agentic_rag.evaluation import SelfRAGEvaluator

evaluator = SelfRAGEvaluator(
    generator=generator,
    regenerate_threshold=0.5,  # Suggest regeneration if below
)

output = await evaluator.evaluate(
    query="Explain attention mechanism",
    response="Attention allows models to focus...",
    contexts=retrieved_contexts,
)

print(f"ISREL: {output.isrel} (score: {output.isrel_score})")
print(f"ISSUP: {output.issup} (score: {output.issup_score})")
print(f"ISUSE: {output.isuse} (score: {output.isuse_score})")
print(f"Overall: {output.overall_score}")
print(f"Should regenerate: {output.should_regenerate}")

# Get human-readable feedback
feedback = evaluator.create_feedback(output)
print(f"Feedback: {feedback}")
```

### Pipeline Integration

```python
from agentic_rag.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .with_evaluation(
        enable_ragas=True,
        enable_self_rag=True,
        regenerate_on_low_score=True,
        regenerate_threshold=0.5,
    )
    .build()
)

# Query returns evaluation with response
result = await pipeline.query("What is ML?", collection="docs")

print(f"Response: {result.response}")
print(f"Evaluation: {result.evaluation}")
```

---

## Best Practices

### 1. Use Appropriate Metrics

| Scenario | Recommended Metrics |
|----------|---------------------|
| General RAG | Context Precision, Faithfulness |
| Factual QA | Faithfulness, Context Recall |
| Conversational | Answer Relevancy, ISUSE |
| Research | All RAGAS + ground truth |

### 2. Set Sensible Thresholds

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Context Precision | > 0.8 | 0.6-0.8 | < 0.6 |
| Faithfulness | > 0.9 | 0.7-0.9 | < 0.7 |
| Answer Relevancy | > 0.8 | 0.6-0.8 | < 0.6 |
| RAGAS Score | > 0.8 | 0.6-0.8 | < 0.6 |

### 3. Iterative Improvement

```
Low Context Precision → Improve retrieval (reranking, filtering)
Low Context Recall → Increase top_k, use HyDE
Low Faithfulness → Stricter prompts, better context
Low Relevancy → Better prompt engineering, query understanding
```

### 4. Monitor Over Time

Track metrics across:
- Query types (factual, analytical, exploratory)
- Document types
- User segments
- Model versions

---

## References

1. Es, S., et al. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)

2. Asai, A., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

3. RAGAS Documentation. "List of available metrics." [docs.ragas.io](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

4. Qdrant. (2024). "Best Practices in RAG Evaluation." [qdrant.tech/blog](https://qdrant.tech/blog/rag-evaluation-guide/)

5. Confident AI. (2024). "RAG Evaluation Metrics." [confident-ai.com/blog](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)

