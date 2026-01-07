"""
Late chunking for improved embeddings.

Embeds the full document first, then extracts chunk embeddings
from the contextualized token representations.

Reference: "Late Chunking" (Jina AI, 2024)
"""

import asyncio

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, Document
from agentic_rag.embeddings.base import BaseEmbedder


class LateChunkingEmbedder(BaseEmbedder):
    """
    Late chunking embedder.

    Instead of embedding chunks independently, embeds the full
    document and extracts chunk representations from the
    contextualized token embeddings.

    Benefits:
    - Chunks retain full document context
    - Better handling of cross-references
    - Improved semantic coherence
    """

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        max_length: int = 8192,
        settings: Settings | None = None,
    ):
        """
        Initialize late chunking embedder.

        Args:
            model: HuggingFace model ID.
            device: Device (cuda, cpu, mps).
            max_length: Maximum sequence length.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._model_name = model or self._settings.embedding_model
        self._device = device or self._settings.embedding_device
        self._max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()

        return self._model, self._tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        model, _ = self._load_model()
        return model.config.hidden_size

    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        import torch

        model, tokenizer = self._load_model()

        loop = asyncio.get_event_loop()

        def compute_embeddings():
            with torch.no_grad():
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)

                outputs = model(**inputs)

                # Mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

                embeddings = sum_embeddings / sum_mask
                return embeddings.cpu().numpy().tolist()

        return await loop.run_in_executor(None, compute_embeddings)

    async def embed_document_with_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> list[list[float]]:
        """
        Embed chunks using late chunking.

        Embeds the full document and extracts chunk-level
        embeddings from token positions.

        Args:
            document: Full document.
            chunks: Chunks with their positions in document.

        Returns:
            List of embeddings for each chunk.
        """
        import torch

        model, tokenizer = self._load_model()

        loop = asyncio.get_event_loop()

        def compute_late_embeddings():
            # Tokenize full document
            full_encoding = tokenizer(
                document.content,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            offset_mapping = full_encoding.pop("offset_mapping")[0].tolist()

            with torch.no_grad():
                inputs = {k: v.to(self._device) for k, v in full_encoding.items()}
                outputs = model(**inputs)
                token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

            # Extract embeddings for each chunk
            chunk_embeddings = []

            for chunk in chunks:
                # Find chunk position in document
                chunk_start = document.content.find(chunk.content)
                if chunk_start == -1:
                    # Fallback: embed chunk directly
                    chunk_embeddings.append(None)
                    continue

                chunk_end = chunk_start + len(chunk.content)

                # Find token indices for this chunk
                token_start = None
                token_end = None

                for i, (start, end) in enumerate(offset_mapping):
                    if start <= chunk_start < end and token_start is None:
                        token_start = i
                    if start < chunk_end <= end:
                        token_end = i + 1

                if token_start is not None and token_end is not None:
                    # Mean pool chunk tokens
                    chunk_tokens = token_embeddings[token_start:token_end]
                    chunk_emb = chunk_tokens.mean(dim=0).cpu().numpy().tolist()
                    chunk_embeddings.append(chunk_emb)
                else:
                    chunk_embeddings.append(None)

            return chunk_embeddings

        late_embeddings = await loop.run_in_executor(None, compute_late_embeddings)

        # Fill in any missing embeddings with direct encoding
        for i, emb in enumerate(late_embeddings):
            if emb is None:
                direct_emb = await self.embed(chunks[i].content)
                late_embeddings[i] = direct_emb

        return late_embeddings
