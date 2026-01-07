"""
Base pipeline protocol and interfaces.

Defines the common interface for all RAG pipeline variants.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag.core.models import Chunk, Document


class PipelineResult(BaseModel):
    """Result from a pipeline query."""

    response: str = Field(description="Generated response")
    sources: list[Chunk] = Field(default_factory=list, description="Source chunks")
    confidence: float = Field(default=0.0, description="Confidence score")
    provider: str = Field(default="", description="LLM provider used")
    model: str = Field(default="", description="Model used")
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    latency_ms: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Result from document ingestion."""

    documents: int = Field(description="Number of documents ingested")
    chunks: int = Field(description="Number of chunks created")
    collection: str = Field(description="Target collection name")
    metadata: dict[str, Any] = Field(default_factory=dict)


class BasePipeline(ABC):
    """
    Abstract base class for RAG pipelines.

    All pipeline variants (standard, agentic, corrective) implement this interface.
    """

    @abstractmethod
    async def query(
        self,
        question: str,
        collection: str,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        Query the pipeline.

        Args:
            question: User question to answer.
            collection: Vector DB collection to search.
            **kwargs: Additional parameters.

        Returns:
            PipelineResult with response and metadata.
        """
        ...

    @abstractmethod
    async def ingest(
        self,
        documents: list[Document],
        collection: str,
        **kwargs: Any,
    ) -> IngestResult:
        """
        Ingest documents into the pipeline.

        Args:
            documents: Documents to ingest.
            collection: Target collection.
            **kwargs: Additional parameters.

        Returns:
            IngestResult with ingestion statistics.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close pipeline resources."""
        ...

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
