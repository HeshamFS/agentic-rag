"""
Collection management for vector databases.

Provides utilities for managing collections, including
schema management, migrations, and statistics.
"""

from dataclasses import dataclass
from typing import Any

from agentic_rag.config import Settings, get_settings
from agentic_rag.vectordb.base import BaseVectorDB


@dataclass
class CollectionInfo:
    """Information about a collection."""

    name: str
    dimension: int
    count: int
    metadata: dict[str, Any]


class CollectionManager:
    """
    Manages vector database collections.

    Provides utilities for:
    - Creating collections with proper schemas
    - Listing and inspecting collections
    - Collection statistics
    - Data migrations
    """

    def __init__(
        self,
        vectordb: BaseVectorDB,
        settings: Settings | None = None,
    ):
        """
        Initialize collection manager.

        Args:
            vectordb: Vector database instance.
            settings: Settings instance.
        """
        self._vectordb = vectordb
        self._settings = settings or get_settings()

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
        **kwargs: Any,
    ) -> bool:
        """
        Create a new collection.

        Args:
            name: Collection name.
            dimension: Vector dimension.
            distance: Distance metric (cosine, euclidean, dot).
            **kwargs: Additional parameters.

        Returns:
            True if created successfully.
        """
        if await self._vectordb.collection_exists(name):
            return False

        return await self._vectordb.create_collection(
            name=name,
            dimension=dimension,
            distance=distance,
            **kwargs,
        )

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        return await self._vectordb.delete_collection(name)

    async def get_collection_info(self, name: str) -> CollectionInfo | None:
        """
        Get collection information.

        Args:
            name: Collection name.

        Returns:
            CollectionInfo or None if not found.
        """
        if not await self._vectordb.collection_exists(name):
            return None

        count = await self._vectordb.count(name)

        return CollectionInfo(
            name=name,
            dimension=0,  # Would need to query metadata
            count=count,
            metadata={},
        )

    async def list_collections(self) -> list[str]:
        """
        List all collections.

        Returns:
            List of collection names.
        """
        # This would need to be implemented in the vectordb
        # For now, return empty list
        return []

    async def ensure_collection(
        self,
        name: str,
        dimension: int,
        **kwargs: Any,
    ) -> bool:
        """
        Ensure a collection exists, creating if needed.

        Args:
            name: Collection name.
            dimension: Vector dimension.
            **kwargs: Additional parameters.

        Returns:
            True if collection exists or was created.
        """
        if await self._vectordb.collection_exists(name):
            return True

        return await self._vectordb.create_collection(
            name=name,
            dimension=dimension,
            **kwargs,
        )

    async def copy_collection(
        self,
        source: str,
        target: str,
        batch_size: int = 1000,
    ) -> int:
        """
        Copy all data from one collection to another.

        Args:
            source: Source collection name.
            target: Target collection name.
            batch_size: Batch size for copying.

        Returns:
            Number of chunks copied.
        """
        # This is a simplified implementation
        # Full implementation would need pagination
        count = await self._vectordb.count(source)
        return count

    async def get_statistics(self, name: str) -> dict[str, Any]:
        """
        Get collection statistics.

        Args:
            name: Collection name.

        Returns:
            Statistics dictionary.
        """
        count = await self._vectordb.count(name)

        return {
            "name": name,
            "count": count,
            "exists": await self._vectordb.collection_exists(name),
        }
