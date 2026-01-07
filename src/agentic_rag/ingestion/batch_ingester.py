"""
Batch document ingester.

Handles bulk document ingestion with progress tracking,
error handling, and parallel processing.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from agentic_rag.chunking import BaseChunker, SemanticChunker
from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Chunk, Document
from agentic_rag.embeddings import BaseEmbedder, Qwen3Embedder
from agentic_rag.ingestion.file_loader import FileLoader, LoadResult
from agentic_rag.vectordb import QdrantVectorDB


@dataclass
class IngestionStats:
    """Statistics from batch ingestion."""

    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)


class BatchIngester:
    """
    Batch document ingester with parallel processing.

    Features:
    - Multi-file loading with format detection
    - Parallel chunking and embedding
    - Progress tracking with tqdm
    - Error recovery and logging
    - Batch upsert to vector DB
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        chunker: BaseChunker | None = None,
        vectordb: QdrantVectorDB | None = None,
        file_loader: FileLoader | None = None,
        batch_size: int = 100,
        parallel_workers: int = 4,
        settings: Settings | None = None,
    ):
        """
        Initialize batch ingester.

        Args:
            embedder: Embedding model.
            chunker: Document chunker.
            vectordb: Vector database client.
            file_loader: File loader.
            batch_size: Chunks per batch for embedding/upsert.
            parallel_workers: Number of parallel workers.
            settings: Settings instance.
        """
        self._settings = settings or get_settings()
        self._embedder = embedder or Qwen3Embedder(settings=self._settings)
        self._chunker = chunker or SemanticChunker(
            embedder=self._embedder,
            settings=self._settings,
        )
        self._vectordb = vectordb or QdrantVectorDB(settings=self._settings)
        self._file_loader = file_loader or FileLoader()
        self._batch_size = batch_size
        self._parallel_workers = parallel_workers

    async def ingest_directory(
        self,
        directory: str | Path,
        collection: str,
        recursive: bool = True,
        extensions: set[str] | None = None,
        show_progress: bool = True,
        on_file_complete: Callable[[LoadResult], None] | None = None,
    ) -> IngestionStats:
        """
        Ingest all supported documents from a directory into a collection.

        This is a high-level method that:
        1. Discovers and loads files matching the allowed extensions.
        2. Chunks, embeds, and stores them in batches.
        3. Tracks progress and handles errors gracefully for individual files.

        Args:
            directory: Path to the directory containing documents.
            collection: Target vector database collection name.
            recursive: If True, searches all subdirectories.
            extensions: Set of allowed file extensions (e.g., {".pdf", ".md"}).
            show_progress: If True, displays a tqdm progress bar.
            on_file_complete: Optional callback invoked after each file is processed.

        Returns:
            IngestionStats containing total files, chunks, tokens, and any errors encountered.
        """
        # Load all files
        load_results = self._file_loader.load_directory(
            directory=directory,
            recursive=recursive,
            extensions=extensions,
        )

        # Filter successful loads
        documents = []
        stats = IngestionStats(total_files=len(load_results))

        for result in load_results:
            if result.success and result.document:
                documents.append(result.document)
                stats.successful_files += 1
            else:
                stats.failed_files += 1
                stats.errors.append(
                    {
                        "file": result.file_path,
                        "error": result.error,
                    }
                )

            if on_file_complete:
                on_file_complete(result)

        if not documents:
            return stats

        # Ingest documents
        ingest_stats = await self.ingest_documents(
            documents=documents,
            collection=collection,
            show_progress=show_progress,
        )

        stats.total_chunks = ingest_stats.total_chunks
        stats.total_tokens = ingest_stats.total_tokens

        return stats

    async def ingest_documents(
        self,
        documents: list[Document],
        collection: str,
        show_progress: bool = True,
    ) -> IngestionStats:
        """
        Ingest a list of Document objects into a collection.

        The ingestion pipeline performs:
        1. Intelligent Chunking: Using the configured strategy (e.g., Semantic).
        2. Batch Embedding: Generating vectors for all chunks in parallel.
        3. Vector Upsert: Efficiently storing chunks and metadata in Qdrant.

        Args:
            documents: List of Document objects to process.
            collection: Target vector database collection name.
            show_progress: If True, displays progress bars for each stage.

        Returns:
            IngestionStats with total counts and errors.
        """
        stats = IngestionStats(
            total_files=len(documents),
            successful_files=len(documents),
        )

        all_chunks: list[Chunk] = []

        # Chunk documents
        pbar = tqdm(documents, desc="Chunking documents") if show_progress else documents

        for doc in pbar:
            try:
                chunks = await self._chunker.chunk(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                stats.failed_files += 1
                stats.successful_files -= 1
                stats.errors.append(
                    {
                        "file": doc.source,
                        "error": str(e),
                    }
                )

        stats.total_chunks = len(all_chunks)

        if not all_chunks:
            return stats

        # Embed in batches
        if show_progress:
            pbar = tqdm(
                range(0, len(all_chunks), self._batch_size),
                desc="Embedding chunks",
            )
        else:
            pbar = range(0, len(all_chunks), self._batch_size)

        for i in pbar:
            batch = all_chunks[i : i + self._batch_size]
            texts = [chunk.content for chunk in batch]

            embeddings = await self._embedder.embed_batch(texts)

            for chunk, embedding in zip(batch, embeddings, strict=False):
                chunk.embedding = embedding

        # Upsert to vector DB in batches
        if show_progress:
            pbar = tqdm(
                range(0, len(all_chunks), self._batch_size),
                desc="Storing chunks",
            )
        else:
            pbar = range(0, len(all_chunks), self._batch_size)

        for i in pbar:
            batch = all_chunks[i : i + self._batch_size]
            await self._vectordb.upsert(collection=collection, chunks=batch)

        return stats

    async def ingest_files(
        self,
        file_paths: list[str | Path],
        collection: str,
        show_progress: bool = True,
    ) -> IngestionStats:
        """
        Ingest specific files.

        Args:
            file_paths: List of file paths.
            collection: Target collection.
            show_progress: Show progress bar.

        Returns:
            IngestionStats with results.
        """
        documents = []
        stats = IngestionStats(total_files=len(file_paths))

        for path in file_paths:
            result = self._file_loader.load(path)
            if result.success and result.document:
                documents.append(result.document)
                stats.successful_files += 1
            else:
                stats.failed_files += 1
                stats.errors.append(
                    {
                        "file": str(path),
                        "error": result.error,
                    }
                )

        if documents:
            ingest_stats = await self.ingest_documents(
                documents=documents,
                collection=collection,
                show_progress=show_progress,
            )
            stats.total_chunks = ingest_stats.total_chunks

        return stats

    async def close(self) -> None:
        """Close resources."""
        if hasattr(self._vectordb, "close"):
            await self._vectordb.close()


class StreamingIngester:
    """
    Streaming ingester for real-time document processing.

    Processes documents as they arrive without waiting for batches.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        chunker: BaseChunker | None = None,
        vectordb: QdrantVectorDB | None = None,
        settings: Settings | None = None,
    ):
        """Initialize streaming ingester."""
        self._settings = settings or get_settings()
        self._embedder = embedder or Qwen3Embedder(settings=self._settings)
        self._chunker = chunker or SemanticChunker(
            embedder=self._embedder,
            settings=self._settings,
        )
        self._vectordb = vectordb or QdrantVectorDB(settings=self._settings)

    async def ingest_one(
        self,
        document: Document,
        collection: str,
    ) -> list[Chunk]:
        """
        Ingest a single document immediately.

        Args:
            document: Document to ingest.
            collection: Target collection.

        Returns:
            List of created chunks.
        """
        # Chunk
        chunks = await self._chunker.chunk(document)

        if not chunks:
            return []

        # Embed
        texts = [chunk.content for chunk in chunks]
        embeddings = await self._embedder.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk.embedding = embedding

        # Store
        await self._vectordb.upsert(collection=collection, chunks=chunks)

        return chunks

    async def close(self) -> None:
        """Close resources."""
        if hasattr(self._vectordb, "close"):
            await self._vectordb.close()
