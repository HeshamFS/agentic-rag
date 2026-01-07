"""
Base vector database protocol.

Defines the interface for vector database implementations.
"""

from abc import ABC, abstractmethod
from typing import Any

from agentic_rag.core.models import Chunk


class BaseVectorDB(ABC):
    """
    Abstract base class for vector databases.

    All vector DB implementations should inherit from this.
    """

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        **kwargs: Any,
    ) -> bool:
        """
        Create a new collection.

        Args:
            name: Collection name.
            dimension: Vector dimension.
            **kwargs: Additional parameters.

        Returns:
            True if created successfully.
        """
        ...

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Args:
            name: Collection name.

        Returns:
            True if deleted successfully.
        """
        ...

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            name: Collection name.

        Returns:
            True if exists.
        """
        ...

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        chunks: list[Chunk],
        **kwargs: Any,
    ) -> int:
        """
        Insert or update chunks.

        Args:
            collection: Collection name.
            chunks: Chunks to upsert.
            **kwargs: Additional parameters.

        Returns:
            Number of chunks upserted.
        """
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for similar vectors.

        Args:
            collection: Collection name.
            query_vector: Query embedding.
            top_k: Number of results.
            filter: Metadata filters.
            **kwargs: Additional parameters.

        Returns:
            List of (chunk, score) tuples.
        """
        ...

    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> int:
        """
        Delete chunks by ID.

        Args:
            collection: Collection name.
            ids: Chunk IDs to delete.

        Returns:
            Number of chunks deleted.
        """
        ...

    @abstractmethod
    async def get(
        self,
        collection: str,
        ids: list[str],
    ) -> list[Chunk]:
        """
        Get chunks by ID.

        Args:
            collection: Collection name.
            ids: Chunk IDs.

        Returns:
            List of chunks.
        """
        ...

    @abstractmethod
    async def count(self, collection: str) -> int:
        """
        Count chunks in collection.

        Args:
            collection: Collection name.

        Returns:
            Number of chunks.
        """
        ...

    async def close(self) -> None:
        """Close database connection."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
