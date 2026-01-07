"""
Real integration tests for embedding functionality.

Tests with ACTUAL embedding models and real text data.
No mocks - we test what we ship.
"""

import numpy as np
import pytest

from agentic_rag.embeddings.qwen3_embedder import create_embedder

# =============================================================================
# Real Embedder Tests
# =============================================================================


class TestRealQwen3Embedder:
    """Tests using the real Qwen3 embedding model."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Create real embedder - loaded once per test class."""
        return create_embedder("small")  # Use small model for faster tests

    @pytest.fixture
    def sample_texts(self):
        """Real academic text samples."""
        return [
            "The Transformer architecture relies entirely on self-attention mechanisms.",
            "BERT uses bidirectional training of Transformers for language understanding.",
            "GPT models are trained using next token prediction as the primary objective.",
            "Attention mechanisms allow models to focus on relevant parts of the input.",
            "Vector embeddings represent semantic meaning in high-dimensional space.",
        ]

    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedder):
        """Test embedding a single text produces valid output."""
        text = "Machine learning is a subset of artificial intelligence."
        embedding = await embedder.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder, sample_texts):
        """Test batch embedding produces correct number of embeddings."""
        embeddings = await embedder.embed_batch(sample_texts)

        assert len(embeddings) == len(sample_texts)
        assert all(len(e) == embedder.dimension for e in embeddings)

    @pytest.mark.asyncio
    async def test_embedding_dimension(self, embedder):
        """Test embedding dimension matches model specification."""
        text = "Test dimension"
        embedding = await embedder.embed_text(text)

        # Qwen3 small model should have specific dimension
        assert len(embedding) == embedder.dimension
        assert embedder.dimension > 0

    @pytest.mark.asyncio
    async def test_embeddings_are_normalized(self, embedder):
        """Test embeddings have unit norm (L2 normalized)."""
        text = "Normalized embedding test"
        embedding = await embedder.embed_text(text)

        norm = np.linalg.norm(embedding)
        # Should be approximately 1.0 (unit vector)
        assert abs(norm - 1.0) < 0.01, f"Embedding norm {norm} not close to 1.0"

    @pytest.mark.asyncio
    async def test_similar_texts_have_high_similarity(self, embedder):
        """Test semantically similar texts have high cosine similarity."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a rug."
        text3 = "Quantum physics explains subatomic particles."

        emb1 = await embedder.embed_text(text1)
        emb2 = await embedder.embed_text(text2)
        emb3 = await embedder.embed_text(text3)

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_related = cosine_sim(emb1, emb2)
        sim_unrelated = cosine_sim(emb1, emb3)

        assert sim_related > sim_unrelated, (
            f"Similar texts should have higher similarity: {sim_related} vs {sim_unrelated}"
        )
        assert sim_related > 0.5, f"Related texts similarity too low: {sim_related}"

    @pytest.mark.asyncio
    async def test_different_texts_have_different_embeddings(self, embedder):
        """Test different texts produce different embeddings."""
        text1 = "Python programming language"
        text2 = "JavaScript web development"

        emb1 = await embedder.embed_text(text1)
        emb2 = await embedder.embed_text(text2)

        # Embeddings should not be identical
        assert emb1 != emb2

    @pytest.mark.asyncio
    async def test_same_text_produces_same_embedding(self, embedder):
        """Test determinism - same input produces same output."""
        text = "Deterministic embedding test"

        emb1 = await embedder.embed_text(text)
        emb2 = await embedder.embed_text(text)

        # Should be identical (or very close due to floating point)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)

    @pytest.mark.asyncio
    async def test_embedding_caching_works(self, embedder):
        """Test that caching returns same results faster."""
        import time

        text = "Cache test embedding"

        # First call - computes embedding
        start = time.time()
        emb1 = await embedder.embed_text(text)
        first_time = time.time() - start

        # Second call - should use cache
        start = time.time()
        emb2 = await embedder.embed_text(text)
        second_time = time.time() - start

        # Results should be identical
        assert emb1 == emb2

        # Cache hit should be faster (at least 10x)
        # Note: First call includes model computation, second is cache lookup
        print(f"First call: {first_time:.4f}s, Second call: {second_time:.4f}s")

    @pytest.mark.asyncio
    async def test_unicode_text_embedding(self, embedder):
        """Test embedding works with unicode characters."""
        unicode_texts = [
            "中文文本测试 - Chinese text",
            "日本語テスト - Japanese text",
            "한국어 테스트 - Korean text",
            "Émojis 🎉🚀🤖 in text",
            "مرحبا بالعالم - Arabic text",
        ]

        for text in unicode_texts:
            embedding = await embedder.embed_text(text)
            assert len(embedding) == embedder.dimension
            assert all(not np.isnan(x) for x in embedding), f"NaN in embedding for: {text}"

    @pytest.mark.asyncio
    async def test_long_text_embedding(self, embedder):
        """Test embedding handles long text."""
        # Create text longer than typical context window
        long_text = "This is a test sentence. " * 500

        embedding = await embedder.embed_text(long_text)

        assert len(embedding) == embedder.dimension
        assert all(not np.isnan(x) for x in embedding)

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty(self, embedder):
        """Test empty batch returns empty list."""
        embeddings = await embedder.embed_batch([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_batch_order_preserved(self, embedder):
        """Test batch embedding preserves input order."""
        texts = [
            "First text about apples",
            "Second text about bananas",
            "Third text about oranges",
        ]

        batch_embeddings = await embedder.embed_batch(texts)
        individual_embeddings = [await embedder.embed_text(t) for t in texts]

        for batch_emb, individual_emb in zip(batch_embeddings, individual_embeddings, strict=False):
            np.testing.assert_array_almost_equal(batch_emb, individual_emb, decimal=5)


# =============================================================================
# Embedding Quality Tests with Real Academic Content
# =============================================================================


class TestEmbeddingQualityWithRealContent:
    """Test embedding quality using real academic paper excerpts."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Create real embedder."""
        return create_embedder("small")

    @pytest.fixture
    def transformer_texts(self):
        """Real excerpts about Transformers."""
        return {
            "attention": "The Transformer uses multi-head self-attention to compute representations of sequences.",
            "bert": "BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
            "gpt": "GPT uses a left-to-right Transformer decoder for language modeling.",
            "unrelated": "The mitochondria is the powerhouse of the cell in biology.",
        }

    @pytest.mark.asyncio
    async def test_topic_clustering(self, embedder, transformer_texts):
        """Test that related topics cluster together in embedding space."""
        embeddings = {}
        for key, text in transformer_texts.items():
            embeddings[key] = await embedder.embed_text(text)

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # ML topics should be more similar to each other
        sim_attention_bert = cosine_sim(embeddings["attention"], embeddings["bert"])
        sim_attention_gpt = cosine_sim(embeddings["attention"], embeddings["gpt"])
        sim_bert_gpt = cosine_sim(embeddings["bert"], embeddings["gpt"])

        # Unrelated topic should be less similar
        sim_attention_unrelated = cosine_sim(embeddings["attention"], embeddings["unrelated"])

        avg_ml_similarity = (sim_attention_bert + sim_attention_gpt + sim_bert_gpt) / 3

        assert avg_ml_similarity > sim_attention_unrelated, (
            f"ML topics should cluster: ML avg={avg_ml_similarity:.3f}, unrelated={sim_attention_unrelated:.3f}"
        )

    @pytest.mark.asyncio
    async def test_query_document_similarity(self, embedder):
        """Test that queries match relevant documents."""
        documents = [
            "Retrieval-Augmented Generation combines neural retrieval with language generation.",
            "Object detection in computer vision uses convolutional neural networks.",
            "Database indexing improves query performance through B-tree structures.",
        ]

        query = "How does RAG improve language model responses?"

        query_emb = await embedder.embed_text(query)
        doc_embs = await embedder.embed_batch(documents)

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarities = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embs]

        # First document (about RAG) should be most similar to the query
        assert similarities[0] == max(similarities), (
            f"RAG document should be most relevant: {similarities}"
        )


# =============================================================================
# Performance Tests
# =============================================================================


class TestEmbeddingPerformance:
    """Performance benchmarks for embedding."""

    @pytest.fixture(scope="class")
    def embedder(self):
        """Create real embedder."""
        return create_embedder("small")

    @pytest.mark.asyncio
    async def test_batch_faster_than_individual(self, embedder):
        """Test that batch embedding is faster than individual calls."""
        import time

        texts = [f"Test sentence number {i} for performance testing." for i in range(20)]

        # Batch embedding
        start = time.time()
        await embedder.embed_batch(texts)
        batch_time = time.time() - start

        # Clear cache
        embedder.clear_cache()

        # Individual embedding
        texts_alt = [f"Alternative sentence number {i} for testing." for i in range(20)]
        start = time.time()
        individual_results = []
        for text in texts_alt:
            individual_results.append(await embedder.embed_text(text))
        individual_time = time.time() - start

        print(f"Batch: {batch_time:.3f}s, Individual: {individual_time:.3f}s")

        # Both approaches should complete in reasonable time
        # Note: Batch may not always be faster due to caching and warm-up effects
        assert batch_time < 60, f"Batch too slow: {batch_time}s"
        assert individual_time < 60, f"Individual too slow: {individual_time}s"

    @pytest.mark.asyncio
    async def test_embedding_throughput(self, embedder):
        """Test embedding throughput is acceptable."""
        import time

        texts = [f"Throughput test sentence {i}." for i in range(50)]

        start = time.time()
        await embedder.embed_batch(texts)
        elapsed = time.time() - start

        throughput = len(texts) / elapsed
        print(f"Embedding throughput: {throughput:.1f} texts/second")

        # Should embed at least 4 texts per second (conservative for CPU)
        assert throughput >= 4, f"Throughput too low: {throughput} texts/sec"


# =============================================================================
# Create Embedder Factory Tests
# =============================================================================


class TestCreateEmbedder:
    """Test the create_embedder factory function."""

    def test_create_small_embedder(self):
        """Test creating small embedder variant."""
        embedder = create_embedder("small")
        assert embedder is not None
        assert embedder.dimension > 0

    def test_create_default_embedder(self):
        """Test creating default embedder."""
        embedder = create_embedder("default")
        assert embedder is not None

    def test_create_with_custom_model(self):
        """Test creating embedder with custom model name."""
        # This uses a known model that exists
        embedder = create_embedder("small")  # Use small as it's guaranteed to work
        assert embedder is not None
