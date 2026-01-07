# Context Compression

> **Reducing Token Costs While Preserving Information**
>
> This document covers extractive compression, LongLLMLingua, and other context compression techniques for RAG systems.

---

## Table of Contents

1. [Overview](#overview)
2. [The Problem: Context Overload](#the-problem-context-overload)
3. [Extractive Compression](#extractive-compression)
4. [LongLLMLingua](#longllmlingua)
5. [Comparison and Selection](#comparison-and-selection)
6. [Configuration](#configuration)

---

## Overview

Context compression reduces the number of tokens sent to the LLM while retaining the most relevant information.

### Key Benefits

| Benefit | Impact |
|---------|--------|
| **Cost Reduction** | 50-80% fewer tokens = 50-80% cost savings |
| **Latency Improvement** | Fewer tokens = faster inference |
| **Context Fitting** | Fit more information in limited context windows |
| **Quality Improvement** | Less noise = better LLM focus |

### Compression in the Pipeline

```
Retrieved Chunks (10 chunks, ~5000 tokens)
         │
         ▼
  ┌──────────────────┐
  │    Compression   │
  │   (Extractive/   │
  │   LongLLMLingua) │
  └────────┬─────────┘
           │
           ▼
Compressed Context (~1500 tokens)
         │
         ▼
    ┌─────────┐
    │   LLM   │
    └─────────┘
```

---

## The Problem: Context Overload

### Lost in the Middle

Research shows LLMs struggle with information in the middle of long contexts:

```
Position in context:
[Beginning] ─────────── [Middle] ─────────── [End]
    ▲                      │                   ▲
    │                      ▼                   │
High attention        Low attention       High attention
```

> **Reference**: Liu, N., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts."

### Token Economics

| Model | Context Window | Cost per 1M tokens |
|-------|---------------|-------------------|
| GPT-4o | 128K | $2.50 |
| Claude 3.5 Sonnet | 200K | $3.00 |
| Gemini 1.5 Flash | 1M | $0.075 |
| Claude 4.5 Sonnet | 200K | $3.00 |

**Example savings**:
- 10 queries/day × 5000 tokens × $0.03/1K = $1.50/day
- With 70% compression: $0.45/day → **$1.05 saved daily**

---

## Extractive Compression

Extractive compression **selects** the most relevant sentences without modifying them.

### Approach

1. Split chunks into sentences
2. Score each sentence for relevance to the query
3. Select top sentences until target token count reached
4. Reassemble in original order

### Scoring Methods

#### 1. Reranker-Based Scoring

Use a reranking model to score query-sentence pairs:

$$\text{score}(s_i) = \text{Reranker}(q, s_i)$$

#### 2. Embedding Similarity

Compare query embedding to sentence embeddings:

$$\text{score}(s_i) = \cos(E_q, E_{s_i}) = \frac{E_q \cdot E_{s_i}}{\|E_q\| \|E_{s_i}\|}$$

#### 3. TF-IDF Overlap

Score based on term overlap with query:

$$\text{score}(s_i) = \sum_{t \in q \cap s_i} \text{TF-IDF}(t, s_i)$$

### Implementation

```python
class ExtractiveCompressor(BaseCompressor):
    def __init__(
        self,
        reranker: Any,
        target_tokens: int | None = None,
        compression_ratio: float = 0.5,
        min_sentences: int = 3,
    ):
        super().__init__(target_tokens, compression_ratio)
        self._reranker = reranker
        self._min_sentences = min_sentences

    async def compress(
        self,
        query: str,
        chunks: list[Chunk],
        target_tokens: int | None = None,
    ) -> CompressionResult:
        # 1. Split chunks into sentences
        # 2. Score sentences with reranker
        # 3. Select top sentences within token budget
        # 4. Restore original order
```

### Advantages & Limitations

| Advantages | Limitations |
|------------|-------------|
| Preserves exact wording | May lose context |
| Fast (no LLM needed) | Binary selection (keep/discard) |
| Interpretable | No abstractive capability |
| Cheap | May fragment coherent passages |

---

## LongLLMLingua

**LongLLMLingua** uses perplexity-based scoring to identify important tokens.

> **Reference**: Jiang, H., et al. (2024). "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression." ACL 2024. [arXiv:2310.06839](https://arxiv.org/abs/2310.06839)

### Core Insight

**Low perplexity tokens are more predictable → less informative**
**High perplexity tokens are less predictable → more informative**

### Mathematical Foundation

#### Perplexity

For a sequence of tokens $T = [t_1, t_2, ..., t_n]$:

$$\text{PPL}(T) = \exp\left(-\frac{1}{n} \sum_{i=1}^{n} \log P(t_i | t_{<i})\right)$$

#### Token-Level Importance

For each token $t_i$, compute importance based on:

$$\text{importance}(t_i) = -\log P(t_i | t_{<i})$$

Higher value = more surprising = more important.

#### Question-Aware Scoring

LongLLMLingua conditions on the question for better scoring:

$$\text{importance}(t_i | q) = -\log P(t_i | t_{<i}, q)$$

### The LongLLMLingua Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    LongLLMLingua Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Coarse-grained Compression                              │
│     ┌─────────────────────────────────────────────────┐     │
│     │ Score entire documents/chunks for relevance     │     │
│     │ Remove low-relevance documents                  │     │
│     └─────────────────────────────────────────────────┘     │
│                          │                                   │
│                          ▼                                   │
│  2. Document Reordering                                     │
│     ┌─────────────────────────────────────────────────┐     │
│     │ Place most relevant documents at start & end    │     │
│     │ (Addresses "lost in the middle" problem)        │     │
│     └─────────────────────────────────────────────────┘     │
│                          │                                   │
│                          ▼                                   │
│  3. Fine-grained Compression                                │
│     ┌─────────────────────────────────────────────────┐     │
│     │ Score individual tokens using perplexity        │     │
│     │ Remove low-importance tokens                    │     │
│     └─────────────────────────────────────────────────┘     │
│                          │                                   │
│                          ▼                                   │
│  4. Subsequence Recovery                                    │
│     ┌─────────────────────────────────────────────────┐     │
│     │ Ensure grammatical coherence                    │     │
│     │ Recover key subsequences if needed              │     │
│     └─────────────────────────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Dynamic Compression Ratio

Different documents may need different compression levels:

$$\tau_i = \frac{\text{relevance}(d_i, q)}{\sum_j \text{relevance}(d_j, q)} \cdot \tau_\text{budget}$$

Where:
- $\tau_i$ = token budget for document $i$
- $\tau_\text{budget}$ = total token budget

More relevant documents get more tokens.

### Implementation

```python
class LongLLMLinguaCompressor(BaseCompressor):
    def __init__(
        self,
        generator: Any,
        target_tokens: int | None = None,
        compression_ratio: float = 0.5,
        min_sentences: int = 3,
        use_query_conditioning: bool = True,
    ):
        super().__init__(target_tokens, compression_ratio)
        self._generator = generator
        self._min_sentences = min_sentences
        self._use_query_conditioning = use_query_conditioning
```

### Performance Results

From the LongLLMLingua paper:

| Benchmark | Baseline | +LongLLMLingua | Tokens Used |
|-----------|----------|----------------|-------------|
| NaturalQuestions | 56.3% | 68.2% (+21.4%) | 1/4 |
| LooGLE | - | - | 1/17 (94% reduction) |
| Multi-hop QA | 41.2% | 52.8% (+28.2%) | 1/6 |

### Cost Analysis

| Scenario | Without Compression | With LongLLMLingua |
|----------|--------------------|--------------------|
| Tokens per query | 8,000 | 2,000 |
| Cost per 1000 queries | $240 | $60 |
| **Savings** | - | **$180 (75%)** |

---

## Comparison and Selection

### Method Comparison

| Aspect | Extractive | LongLLMLingua |
|--------|------------|---------------|
| Granularity | Sentence-level | Token-level |
| Speed | Fast (~50ms) | Slow (~500ms) |
| Quality | Good | Better |
| Cost | Minimal | LLM inference cost |
| Complexity | Low | High |

### Selection Guide

```
Is inference cost a concern?
├─ Yes
│  ├─ Need highest quality?
│  │  ├─ Yes → LongLLMLingua
│  │  └─ No → Extractive
│  └─ Is compression latency acceptable?
│     ├─ Yes → LongLLMLingua
│     └─ No → Extractive
└─ No → Skip compression
```

### Compression Ratio Guidelines

| Use Case | Recommended Ratio |
|----------|-------------------|
| Cost optimization | 0.3-0.5 (50-70% reduction) |
| Context window fitting | Dynamic (fit to window) |
| Quality focus | 0.5-0.7 (30-50% reduction) |
| Aggressive savings | 0.1-0.3 (70-90% reduction) |

---

## Configuration

### Extractive Compression

```python
from agentic_rag.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .with_retrieval("hybrid", top_k=20)
    .with_compression(
        method="extractive",
        compression_ratio=0.5,
        min_sentences=3
    )
    .build()
)
```

### LongLLMLingua Compression

```python
pipeline = (
    PipelineBuilder()
    .with_retrieval("hybrid", top_k=20)
    .with_compression(
        method="longllmlingua",
        compression_ratio=0.3,
        use_question_conditioning=True
    )
    .build()
)
```

### Target Token Mode

```python
pipeline = (
    PipelineBuilder()
    .with_compression(
        target_tokens=2000,  # Absolute limit instead of ratio
    )
    .build()
)
```

---

## References

1. Jiang, H., et al. (2024). "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression." ACL 2024. [arXiv:2310.06839](https://arxiv.org/abs/2310.06839)

2. Microsoft Research. (2024). "LLMLingua: Innovating LLM efficiency with prompt compression." [microsoft.com/en-us/research/blog/llmlingua](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/)

3. LlamaIndex. "LongLLMLingua: Bye-bye to Middle Loss and Save on Your RAG Costs." [llamaindex.ai/blog/longllmlingua](https://www.llamaindex.ai/blog/longllmlingua-bye-bye-to-middle-loss-and-save-on-your-rag-costs-via-prompt-compression-54b559b9ddf7)

4. Liu, N., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
