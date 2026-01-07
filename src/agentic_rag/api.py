"""
FastAPI REST API for RAG Optimizer.

Provides HTTP endpoints for:
- File upload (PDF, DOCX, TXT, MD)
- Document ingestion and chunking
- Semantic search
- RAG querying with streaming
- Multi-turn chat conversations

Usage:
    uvicorn agentic_rag.api:app --reload --port 8000
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from agentic_rag.config import Settings, get_settings
from agentic_rag.core.models import Document
from agentic_rag.embeddings import Qwen3Embedder
from agentic_rag.ingestion.file_loader import FileLoader
from agentic_rag.pipeline import PipelineBuilder

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agentic_rag.api")

# =============================================================================
# Request/Response Models
# =============================================================================


class UploadResponse(BaseModel):
    """File upload response."""

    success: bool
    file_id: str
    filename: str
    chunks_created: int
    collection: str
    processing_time_sec: float
    # GraphRAG stats
    entities_extracted: int = 0
    relationships_extracted: int = 0
    # Caching
    cached: bool = False


class IngestRequest(BaseModel):
    """Document ingestion request (JSON)."""

    documents: list[dict[str, Any]]
    collection: str = "default"
    chunk_strategy: str = "semantic"
    chunk_size: int = 512


class IngestResponse(BaseModel):
    """Document ingestion response."""

    success: bool
    documents_processed: int
    chunks_created: int
    collection: str


class SearchRequest(BaseModel):
    """Search request (retrieval only, no generation)."""

    query: str
    collection: str = "default"
    top_k: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = True


class SearchResult(BaseModel):
    """Individual search result."""

    chunk_id: str
    content: str
    score: float
    document_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response."""

    results: list[SearchResult]
    query: str
    collection: str
    total_results: int
    search_time_ms: float


class QueryRequest(BaseModel):
    """Query request with LLM generation."""

    question: str
    collection: str = "default"
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = "standard"  # standard, agentic, corrective
    # Generation config (overrides pipeline defaults)
    provider: str | None = None  # claude, openai, gemini, local
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int = 2048
    # Retrieval config
    use_hyde: bool | None = None
    use_multi_query: bool | None = None
    use_reranking: bool | None = None


class PipelineConfigRequest(BaseModel):
    """Pipeline configuration update request."""

    # Generation
    provider: str | None = None  # claude, openai, gemini, local
    model: str | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None  # GPT-5: none/low/medium/high/xhigh
    # Retrieval
    use_hyde: bool | None = None
    use_multi_query: bool | None = None
    use_reranking: bool | None = None
    retrieval_strategy: str | None = None  # dense, sparse, hybrid
    # Agentic
    enable_self_rag: bool | None = None
    enable_planning: bool | None = None


class PipelineConfigResponse(BaseModel):
    """Current pipeline configuration."""

    provider: str
    model: str
    temperature: float
    reasoning_effort: str  # GPT-5: none/low/medium/high/xhigh
    use_hyde: bool
    use_multi_query: bool
    use_reranking: bool
    retrieval_strategy: str
    enable_self_rag: bool
    enable_planning: bool


class PipelineStep(BaseModel):
    """A step in the RAG pipeline."""

    name: str
    duration_ms: float
    details: dict[str, Any] = Field(default_factory=dict)


class IntermediateAnswer(BaseModel):
    """Intermediate answer for a query variation."""

    query: str
    answer: str
    index: int


class QueryResponse(BaseModel):
    """Query response with full pipeline visibility."""

    response: str
    sources: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Pipeline transparency
    pipeline_steps: list[PipelineStep] = Field(default_factory=list)
    query_variations: list[str] = Field(default_factory=list)  # Multi-query variations
    intermediate_answers: list[IntermediateAnswer] = Field(
        default_factory=list
    )  # Answers for each variation
    thinking: str = ""  # LLM thinking/reasoning (if available)


class ChatMessage(BaseModel):
    """Chat message."""

    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Multi-turn chat request."""

    message: str
    collection: str = "default"
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)
    provider: str | None = None
    model: str | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response."""

    response: str
    sources: list[dict[str, Any]]
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionInfo(BaseModel):
    """Collection information."""

    name: str
    chunk_count: int
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"
    qdrant_connected: bool = False
    embedder_loaded: bool = False


# =============================================================================
# Global State
# =============================================================================

_pipeline = None
_settings: Settings | None = None
_file_loader: FileLoader | None = None
_chat_sessions: dict[str, list[ChatMessage]] = {}


# =============================================================================
# Application Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _pipeline, _settings, _file_loader

    logger.info("=" * 60)
    logger.info("STARTUP: Beginning API initialization")
    total_start = time.time()

    step_start = time.time()
    _settings = get_settings()
    logger.info(f"STARTUP: Settings loaded in {time.time() - step_start:.2f}s")

    step_start = time.time()
    _file_loader = FileLoader()
    logger.info(f"STARTUP: FileLoader created in {time.time() - step_start:.2f}s")

    # Initialize pipeline components
    try:
        step_start = time.time()
        logger.info(
            f"STARTUP: Creating embedder (model={_settings.embedding_model}, device={_settings.embedding_device})"
        )
        embedder = Qwen3Embedder(
            model_name=_settings.embedding_model,
            device=_settings.embedding_device,
        )
        logger.info(f"STARTUP: Embedder created in {time.time() - step_start:.2f}s")

        step_start = time.time()
        logger.info("STARTUP: Building pipeline...")
        builder = PipelineBuilder()
        logger.info(f"  - PipelineBuilder created: {time.time() - step_start:.3f}s")

        step_start = time.time()
        builder = builder.with_embedder(embedder)
        logger.info(f"  - with_embedder: {time.time() - step_start:.3f}s")

        step_start = time.time()
        builder = builder.with_vectordb(
            "qdrant",
            url=str(_settings.qdrant_url),
            api_key=_settings.qdrant_api_key.get_secret_value()
            if _settings.qdrant_api_key
            else None,
        )
        logger.info(f"  - with_vectordb: {time.time() - step_start:.3f}s")

        step_start = time.time()
        builder = builder.with_generator(_settings.llm_provider, model=_settings.llm_model)
        logger.info(f"  - with_generator: {time.time() - step_start:.3f}s")

        step_start = time.time()
        # SemanticChunker: +70% accuracy over fixed-size (RAG.md research)
        # Trade-off: requires embedding every sentence for intelligent breakpoints
        builder = builder.with_chunking("semantic", chunk_size=_settings.default_chunk_size)
        logger.info(
            f"  - with_chunking (semantic - +70% accuracy): {time.time() - step_start:.3f}s"
        )

        step_start = time.time()
        builder = builder.with_retrieval(
            "hybrid",
            use_hyde=False,
            use_multi_query=True,
            num_queries=4,
        )
        logger.info(f"  - with_retrieval (multi-query): {time.time() - step_start:.3f}s")

        # Contextual Retrieval: -67% failed retrievals (Anthropic research, RAG.md)
        # Trade-off: requires LLM call per chunk for context headers
        step_start = time.time()
        builder = builder.with_contextual_chunking(enabled=True)
        logger.info(
            f"  - with_contextual_chunking (-67% failed retrievals): {time.time() - step_start:.3f}s"
        )

        # GraphRAG disabled for now (adds latency during ingestion)
        step_start = time.time()
        builder = builder.with_graphrag(enabled=False)
        logger.info(f"  - with_graphrag (disabled): {time.time() - step_start:.3f}s")

        step_start = time.time()
        _pipeline = builder.build()
        logger.info(f"  - build(): {time.time() - step_start:.3f}s")

        # Log active components summary
        logger.info("=" * 60)
        logger.info("ACTIVE COMPONENTS SUMMARY:")
        logger.info(
            f"  ✓ Embedder: {_pipeline.embedder.model_name} (dim={_pipeline.embedder.dimension})"
        )
        logger.info(f"  ✓ Vector DB: {_pipeline.vectordb.db_type}")
        logger.info(f"  ✓ Generator: {_pipeline.generator.provider}")
        chunker_name = type(_pipeline.chunker).__name__ if _pipeline.chunker else "Fallback"
        chunker_type = "ContextualChunker" if _pipeline.use_contextual_chunking else chunker_name
        logger.info(f"  ✓ Chunker: {chunker_type}")
        logger.info(
            f"  ✓ Reranker: {'ColBERT (jinaai/jina-colbert-v2)' if _pipeline.reranker else 'None'}"
        )
        logger.info(
            f"  ✓ Multi-Query: {'Enabled (' + str(_pipeline.num_queries) + ' variations)' if _pipeline.use_multi_query else 'Disabled'}"
        )
        logger.info(f"  ✓ HyDE: {'Enabled' if _pipeline.config.use_hyde else 'Disabled'}")
        logger.info(
            f"  ✓ Hybrid Retrieval: {'Enabled' if _pipeline.config.retrieval_strategy == 'hybrid' else 'Disabled'}"
        )
        logger.info(f"  ✓ RRF Fusion: {'Enabled' if _pipeline.config.use_rrf else 'Disabled'}")
        logger.info(
            f"  ✓ Contextual Chunking: {'Enabled (-67% failed retrievals)' if _pipeline.use_contextual_chunking else 'Disabled'}"
        )
        logger.info(f"  ✓ GraphRAG: {'Enabled' if _pipeline.use_graphrag else 'Disabled'}")
        logger.info("=" * 60)

        logger.info(f"STARTUP: Total initialization time: {time.time() - total_start:.2f}s")
        logger.info("STARTUP: API ready to serve requests!")
        logger.info("=" * 60)

        yield
    finally:
        _pipeline = None
        logger.info("SHUTDOWN: API shutting down")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="RAG Optimizer API",
    description="""
## Production-grade Agentic RAG Framework

### Features
- **File Upload**: Upload PDF, DOCX, TXT, MD files
- **Semantic Search**: Find relevant content without generation
- **RAG Query**: Question answering with source attribution
- **Streaming**: Real-time response streaming
- **Multi-turn Chat**: Conversation with history

### Quick Start
1. Upload documents: `POST /upload`
2. Search: `POST /search`
3. Query with AI: `POST /query`
4. Chat: `POST /chat`
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the API and its dependencies.
    """
    qdrant_ok = False
    embedder_ok = False

    if _pipeline:
        try:
            # Check components
            qdrant_ok = _pipeline.vectordb is not None
            embedder_ok = _pipeline.embedder is not None
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if _pipeline else "starting",
        qdrant_connected=qdrant_ok,
        embedder_loaded=embedder_ok,
    )


@app.get("/", tags=["System"])
async def root():
    """API root - returns basic info."""
    return {
        "name": "RAG Optimizer API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# File Upload Endpoint (with streaming progress)
# =============================================================================


@app.post("/upload/stream", tags=["Documents"])
async def upload_file_stream(
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    chunk_size: int = Form(default=512),
):
    """
    Upload with streaming progress updates via SSE.

    Events:
    - progress: {step: "parsing|chunking|embedding|storing", message: str, percent: int}
    - done: {success: true, ...UploadResponse fields}
    - error: {error: str}
    """
    if not _pipeline or not _file_loader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}",
        )

    async def generate() -> AsyncGenerator[str, None]:
        import hashlib

        start_time = time.time()

        try:
            # Step 1: Reading file
            yield f"event: progress\ndata: {json.dumps({'step': 'reading', 'message': 'Reading file...', 'percent': 5})}\n\n"
            content = await file.read()
            file_hash = hashlib.md5(content).hexdigest()
            cache_key = f"{filename}_{file_hash[:8]}"

            # Step 2: Check cache (skip if collection doesn't exist or index not configured)
            yield f"event: progress\ndata: {json.dumps({'step': 'checking', 'message': 'Preparing...', 'percent': 10})}\n\n"
            # Note: Cache check requires Qdrant payload index - skip for now to avoid errors
            # The non-streaming endpoint handles caching; this streaming endpoint prioritizes UX

            # Step 3: Parse file
            yield f"event: progress\ndata: {json.dumps({'step': 'parsing', 'message': f'Parsing {ext} file...', 'percent': 20})}\n\n"
            with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            result = _file_loader.load(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

            if not result.success or not result.document:
                yield f"event: error\ndata: {json.dumps({'error': f'Failed to parse: {result.error}'})}\n\n"
                return

            # Step 4: Chunking
            yield f"event: progress\ndata: {json.dumps({'step': 'chunking', 'message': 'Splitting into chunks...', 'percent': 40})}\n\n"
            file_id = str(uuid.uuid4())[:8]
            result.document.metadata["file_id"] = file_id
            result.document.metadata["filename"] = filename
            result.document.metadata["uploaded_at"] = datetime.now().isoformat()
            result.document.metadata["source_file"] = cache_key

            # Step 5: Embedding (slowest step) - send periodic heartbeats
            yield f"event: progress\ndata: {json.dumps({'step': 'embedding', 'message': 'Generating embeddings (this may take 30-60 seconds)...', 'percent': 50})}\n\n"

            # Run ingest with periodic progress updates
            import asyncio

            # Create the ingest task
            ingest_task = asyncio.create_task(
                _pipeline.ingest(
                    documents=[result.document],
                    collection=collection,
                )
            )

            # Send heartbeats while waiting
            progress_pct = 50
            while not ingest_task.done():
                await asyncio.sleep(2)  # Check every 2 seconds
                if not ingest_task.done():
                    progress_pct = min(progress_pct + 5, 85)  # Slowly increment to 85%
                    yield f"event: progress\ndata: {json.dumps({'step': 'embedding', 'message': 'Still processing... (chunking & embedding)', 'percent': progress_pct})}\n\n"

            # Get result (may raise exception)
            ingest_result = await ingest_task

            # Step 6: Storing complete
            yield f"event: progress\ndata: {json.dumps({'step': 'storing', 'message': 'Finalizing...', 'percent': 95})}\n\n"

            # Done
            processing_time = time.time() - start_time
            yield f"event: done\ndata: {json.dumps({'success': True, 'file_id': file_id, 'filename': filename, 'chunks_created': ingest_result.get('chunks', 0), 'collection': collection, 'processing_time_sec': round(processing_time, 2), 'entities_extracted': ingest_result.get('entities', 0), 'relationships_extracted': ingest_result.get('relationships', 0), 'cached': False})}\n\n"

        except Exception as e:
            logger.error(f"Upload stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_file(
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    chunk_size: int = Form(default=512),
) -> UploadResponse:
    """
    Upload and process a document file.

    The ingestion pipeline performs the following steps:
    1. Parse: Extract text from PDF, DOCX, TXT, MD, or HTML.
    2. Chunk: Split text using semantic boundaries (+70% retrieval accuracy).
    3. Contextualize: Generate context headers for each chunk (-67% failed retrievals).
    4. Embed: Generate high-dimensional vectors using Qwen2-1.5B.
    5. Store: Index chunks and vectors in Qdrant for fast retrieval.

    Args:
        file: The document file to upload.
        collection: Target Qdrant collection name.
        chunk_size: Target size for semantic chunks.

    Returns:
        UploadResponse with processing statistics and file metadata.
    """
    if not _pipeline or not _file_loader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    # Validate file extension
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Supported: PDF, DOCX, TXT, MD, HTML",
        )

    logger.info(f"UPLOAD: Starting upload for {filename}")
    start_time = time.time()

    try:
        # Read file content for hashing
        content = await file.read()

        # Check if document already exists in collection (cache check)
        import hashlib

        file_hash = hashlib.md5(content).hexdigest()
        cache_key = f"{filename}_{file_hash[:8]}"
        logger.info(f"UPLOAD: Cache key = {cache_key}")

        # Check if already embedded (only if collection exists)
        try:
            collection_exists = await _pipeline.vectordb.collection_exists(collection)
            if collection_exists:
                logger.info(f"UPLOAD: Checking cache in collection '{collection}'...")
                existing = await _pipeline.vectordb.search_by_payload(
                    collection=collection,
                    key="source_file",
                    value=cache_key,
                    top_k=1,
                )
                if existing:
                    logger.info(f"UPLOAD: ✓ Cache HIT - Document '{filename}' already embedded")
                    # Get chunk count
                    all_chunks = await _pipeline.vectordb.search_by_payload(
                        collection=collection,
                        key="source_file",
                        value=cache_key,
                        top_k=1000,
                    )
                    return UploadResponse(
                        success=True,
                        file_id=cache_key,
                        filename=filename,
                        chunks_created=len(all_chunks),
                        collection=collection,
                        processing_time_sec=round(time.time() - start_time, 2),
                        entities_extracted=0,
                        relationships_extracted=0,
                        cached=True,
                    )
                else:
                    logger.info(f"UPLOAD: Cache MISS - Document not found with key '{cache_key}'")
            else:
                logger.info(
                    f"UPLOAD: Collection '{collection}' doesn't exist yet - skipping cache check"
                )
        except Exception as e:
            logger.warning(f"UPLOAD: Cache check failed: {e}")

        # Save uploaded file temporarily
        step_start = time.time()
        with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(
            f"UPLOAD: File saved to temp ({len(content)} bytes) in {time.time() - step_start:.2f}s"
        )

        # Load document
        step_start = time.time()
        result = _file_loader.load(tmp_path)
        logger.info(f"UPLOAD: File parsed in {time.time() - step_start:.2f}s")

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        if not result.success or not result.document:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse file: {result.error}",
            )

        doc_len = len(result.document.content)
        logger.info(f"UPLOAD: Document content length: {doc_len} chars")

        # Generate file ID
        file_id = str(uuid.uuid4())[:8]

        # Update document metadata
        result.document.metadata["file_id"] = file_id
        result.document.metadata["filename"] = filename
        result.document.metadata["uploaded_at"] = datetime.now().isoformat()
        result.document.metadata["source_file"] = cache_key  # For caching lookup

        # Ingest document
        logger.info(f"UPLOAD: Starting ingestion to collection '{collection}'...")
        step_start = time.time()
        ingest_result = await _pipeline.ingest(
            documents=[result.document],
            collection=collection,
        )
        logger.info(f"UPLOAD: Ingestion completed in {time.time() - step_start:.2f}s")

        processing_time = time.time() - start_time
        logger.info(
            f"UPLOAD: Total upload time: {processing_time:.2f}s, chunks: {ingest_result.get('chunks', 0)}"
        )

        return UploadResponse(
            success=True,
            file_id=file_id,
            filename=filename,
            chunks_created=ingest_result.get("chunks", 0),
            collection=collection,
            processing_time_sec=round(processing_time, 2),
            entities_extracted=ingest_result.get("entities", 0),
            relationships_extracted=ingest_result.get("relationships", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"UPLOAD: Error - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        )


# =============================================================================
# Document Ingestion (JSON)
# =============================================================================


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """
    Ingest documents from JSON.

    For direct API integration when you have text content.
    Use /upload for file uploads.

    Args:
        request: List of documents with content and metadata

    Returns:
        Ingestion result
    """
    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        # Convert dicts to Document objects
        documents = [
            Document(
                id=doc.get("id", f"doc_{i}"),
                content=doc["content"],
                metadata=doc.get("metadata", {}),
            )
            for i, doc in enumerate(request.documents)
        ]

        # Ingest documents
        result = await _pipeline.ingest(
            documents=documents,
            collection=request.collection,
        )

        return IngestResponse(
            success=True,
            documents_processed=len(documents),
            chunks_created=result.get("chunks", 0),
            collection=request.collection,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# Search Endpoint (Retrieval Only)
# =============================================================================


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest) -> SearchResponse:
    """
    Semantic search without LLM generation.

    Returns relevant chunks ranked by similarity score.
    Use this to find relevant content before deciding to generate.

    Args:
        request: Search query and parameters

    Returns:
        Ranked list of relevant chunks
    """
    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    logger.info(
        f"SEARCH: Query='{request.query[:50]}...', collection={request.collection}, top_k={request.top_k}"
    )
    start_time = time.time()

    try:
        # Embed query
        step_start = time.time()
        query_vector = await _pipeline.embedder.embed_text(request.query)
        logger.info(f"SEARCH: Query embedded in {time.time() - step_start:.2f}s")

        # Search with extra candidates for reranking
        retrieve_k = request.top_k * 3 if _pipeline.reranker else request.top_k
        step_start = time.time()
        results = await _pipeline.vectordb.search(
            collection=request.collection,
            query_vector=query_vector,
            top_k=retrieve_k,
        )
        logger.info(
            f"SEARCH: Qdrant search in {time.time() - step_start:.2f}s, found {len(results)} results"
        )

        # Rerank if available
        chunks = [chunk for chunk, _ in results]
        [score for _, score in results]

        if _pipeline.reranker and len(chunks) > request.top_k:
            try:
                step_start = time.time()
                logger.info(f"SEARCH: Reranking {len(chunks)} chunks with ColBERT...")
                rerank_result = await _pipeline.reranker.rerank(
                    query=request.query,
                    chunks=chunks,
                    top_k=request.top_k,
                )
                # Create new results with reranked scores
                results = [
                    (chunk, score)
                    for chunk, score in zip(rerank_result.chunks, rerank_result.scores, strict=False)
                ]
                logger.info(
                    f"SEARCH: Reranked to {len(results)} results in {time.time() - step_start:.2f}s"
                )
            except Exception as e:
                logger.warning(f"SEARCH: Reranking failed: {e}, using original results")
                results = results[: request.top_k]
        else:
            results = results[: request.top_k]

        search_time = (time.time() - start_time) * 1000
        logger.info(f"SEARCH: Total time: {search_time:.0f}ms")

        return SearchResponse(
            results=[
                SearchResult(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    score=score,
                    document_id=chunk.document_id,
                    metadata=chunk.metadata if request.include_metadata else {},
                )
                for chunk, score in results
            ],
            query=request.query,
            collection=request.collection,
            total_results=len(results),
            search_time_ms=round(search_time, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# Query Endpoint (RAG with Generation)
# =============================================================================


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    Execute a RAG query with LLM response generation.

    This endpoint provides full pipeline visibility, returning not just the answer
    but also the retrieved sources, the reasoning steps, and performance metrics.

    It supports multiple modes:
    - standard: Linear retrieve-then-generate flow.
    - agentic: Multi-agent orchestration with planning and self-correction.
    - corrective: CRAG-based retrieval with query refinement.

    Args:
        request: QueryRequest containing the question and optional overrides.

    Returns:
        QueryResponse with generated answer, sources, and pipeline transparency.
    """
    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        # Apply request overrides to pipeline config for this query
        if request.use_hyde is not None:
            _pipeline.config.use_hyde = request.use_hyde
        if request.use_multi_query is not None:
            _pipeline.use_multi_query = request.use_multi_query

        result = await _pipeline.query(
            question=request.question,
            collection=request.collection,
            top_k=request.top_k,
            use_reranking=request.use_reranking,  # Pass to pipeline
        )

        # Extract pipeline details from result metadata
        result_metadata = result.metadata or {}
        pipeline_steps = result_metadata.get("pipeline_steps", [])
        query_variations = result_metadata.get("query_variations", [request.question])
        thinking = result_metadata.get("thinking", "")
        intermediate_answers = result_metadata.get("intermediate_answers", [])

        # Deduplicate sources - avoid showing nearly identical content
        seen_content = set()
        unique_sources = []
        for chunk in result.sources:
            # Use first 200 chars as fingerprint for deduplication
            fingerprint = chunk.content[:200].strip().lower()
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                unique_sources.append(
                    {
                        "content": chunk.content[:500],
                        "document_id": chunk.document_id,
                        "metadata": chunk.metadata,
                    }
                )

        return QueryResponse(
            response=result.response,
            sources=unique_sources,
            metadata={
                "provider": result.provider,
                "model": result.model or (_settings.llm_model if _settings else "unknown"),
                "confidence": result.confidence,
                "mode": request.mode,
            },
            # Pipeline transparency
            pipeline_steps=[
                PipelineStep(
                    name=step["name"],
                    duration_ms=step["duration_ms"],
                    details=step.get("details", {}),
                )
                for step in pipeline_steps
            ],
            query_variations=query_variations,
            intermediate_answers=[
                IntermediateAnswer(
                    query=ia["query"],
                    answer=ia["answer"],
                    index=ia["index"],
                )
                for ia in intermediate_answers
            ],
            thinking=thinking,  # Gemini 3 thinking tokens
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# Streaming Query Endpoint
# =============================================================================


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Query with streaming response and full pipeline visibility.

    Returns Server-Sent Events (SSE) stream for real-time token streaming.

    Event types:
    - `step`: Pipeline step started/completed
    - `queries`: Multi-query variations (if enabled)
    - `sources`: Source documents with citations
    - `chunk`: Generated text chunk
    - `thinking`: LLM reasoning (if available)
    - `done`: Stream complete with metadata
    - `error`: Error occurred

    Args:
        request: Question and configuration

    Returns:
        SSE stream
    """
    import time

    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    async def generate() -> AsyncGenerator[str, None]:
        pipeline_steps = []
        query_variations = [request.question]
        thinking = ""

        try:
            # Apply request overrides to pipeline config for this query
            if request.use_hyde is not None:
                _pipeline.config.use_hyde = request.use_hyde
            if request.use_multi_query is not None:
                _pipeline.use_multi_query = request.use_multi_query

            # Step 1: Query Expansion (if multi-query enabled)
            time.time()
            yield f"event: step\ndata: {json.dumps({'name': 'query_expansion', 'status': 'started'})}\n\n"

            # Use full pipeline query which handles multi-query internally
            result = await _pipeline.query(
                question=request.question,
                collection=request.collection,
                top_k=request.top_k,
                use_reranking=request.use_reranking,  # Pass to pipeline
            )

            # Extract metadata from pipeline result
            result_metadata = result.metadata or {}
            query_variations = result_metadata.get("query_variations", [request.question])
            thinking = result_metadata.get("thinking", "")
            pipeline_steps = result_metadata.get("pipeline_steps", [])

            # Send query variations if multi-query was used
            if len(query_variations) > 1:
                yield f"event: queries\ndata: {json.dumps({'original': request.question, 'variations': query_variations})}\n\n"

            # Send pipeline steps
            for step in pipeline_steps:
                yield f"event: step\ndata: {json.dumps({'name': step['name'], 'status': 'completed', 'duration_ms': step['duration_ms'], 'details': step.get('details', {})})}\n\n"

            # Send sources with citation numbers and relevance scores
            seen_content = set()
            sources = []
            citation_num = 1
            for chunk in result.sources:
                fingerprint = chunk.content[:200].strip().lower()
                if fingerprint not in seen_content:
                    seen_content.add(fingerprint)
                    # Extract filename and score from metadata
                    metadata = chunk.metadata or {}
                    filename = (
                        metadata.get("filename")
                        or metadata.get("source_file", "").split("/")[-1].split("\\")[-1]
                        or chunk.document_id
                    )
                    # Get reranking score if available, otherwise use retrieval score
                    score = metadata.get("rerank_score", metadata.get("score", 0))
                    sources.append(
                        {
                            "citation": citation_num,
                            "content": chunk.content[:500],
                            "document_id": chunk.document_id,
                            "filename": filename,
                            "score": round(score, 3) if score else 0,  # Include relevance score
                            "metadata": metadata,
                        }
                    )
                    citation_num += 1

            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

            # Send thinking if available (Gemini 3)
            if thinking:
                yield f"event: thinking\ndata: {json.dumps({'content': thinking})}\n\n"

            # Stream the response text
            response_text = result.response
            words = response_text.split()
            for i in range(0, len(words), 3):
                chunk_text = " ".join(words[i : i + 3]) + " "
                yield f"event: chunk\ndata: {json.dumps({'text': chunk_text})}\n\n"
                await asyncio.sleep(0.015)

            # Send done with full metadata
            yield f"event: done\ndata: {json.dumps({'status': 'complete', 'provider': result.provider, 'model': result.model, 'confidence': result.confidence})}\n\n"

        except Exception as e:
            import traceback

            logger.error(f"Stream error: {e}\n{traceback.format_exc()}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# =============================================================================
# Chat Endpoint (Multi-turn)
# =============================================================================


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Multi-turn chat with conversation history.

    Maintains context across multiple turns for coherent conversations.

    Args:
        request: Message, history, and configuration

    Returns:
        Assistant response with sources
    """
    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())

        # Get history from request or session
        if request.history:
            history = request.history
        elif session_id in _chat_sessions:
            history = _chat_sessions[session_id]
        else:
            history = []

        # Add user message to history
        history.append(ChatMessage(role="user", content=request.message))

        # Retrieve relevant context
        query_vector = await _pipeline.embedder.embed_text(request.message)
        results = await _pipeline.vectordb.search(
            collection=request.collection,
            query_vector=query_vector,
            top_k=request.top_k,
        )

        # Build context from retrieved chunks
        context = "\n\n".join([chunk.content for chunk, _ in results])

        # Build conversation prompt
        conversation = "\n".join(
            [
                f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
                for msg in history[-10:]  # Last 10 messages for context
            ]
        )

        prompt = f"""You are a helpful research assistant. Use the following context to answer questions.

Context from documents:
{context}

Conversation history:
{conversation}

Provide a helpful, accurate response based on the context. If the context doesn't contain relevant information, say so."""

        # Generate response
        response = await _pipeline.generator.generate_text(
            prompt=prompt,
            temperature=0.3,
            max_tokens=2048,
        )

        # Add assistant response to history
        history.append(ChatMessage(role="assistant", content=response))

        # Store session
        _chat_sessions[session_id] = history

        return ChatResponse(
            response=response,
            sources=[
                {
                    "content": chunk.content[:500],
                    "document_id": chunk.document_id,
                    "metadata": chunk.metadata,
                }
                for chunk, _ in results
            ],
            session_id=session_id,
            metadata={
                "provider": _settings.llm_provider if _settings else "unknown",
                "model": _settings.llm_model if _settings else "unknown",
                "history_length": len(history),
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.delete("/chat/{session_id}", tags=["Chat"])
async def clear_chat_session(session_id: str):
    """
    Clear a chat session's history.

    Args:
        session_id: Session to clear

    Returns:
        Confirmation
    """
    if session_id in _chat_sessions:
        del _chat_sessions[session_id]
        return {"success": True, "session_id": session_id, "message": "Session cleared"}

    return {"success": False, "session_id": session_id, "message": "Session not found"}


# =============================================================================
# Pipeline Configuration
# =============================================================================

# Cache for dynamically created generators
_generator_cache: dict[str, Any] = {}


def _get_generator(
    provider: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
):
    """Get or create a generator for the given provider/model.

    Args:
        provider: LLM provider (claude, openai, gemini, local)
        model: Model name
        reasoning_effort: For GPT-5 models (none/low/medium/high/xhigh)
    """
    from agentic_rag.generation import (
        ClaudeGenerator,
        GeminiGenerator,
        LocalGenerator,
        OpenAIGenerator,
    )

    # Include reasoning_effort in cache key for OpenAI
    cache_key = f"{provider}:{model or 'default'}:{reasoning_effort or 'default'}"
    if cache_key in _generator_cache:
        return _generator_cache[cache_key]

    # Create new generator
    if provider == "claude":
        gen = ClaudeGenerator(model=model or "claude-sonnet-4-5-20250929")
    elif provider == "openai":
        # GPT-5 uses reasoning_effort instead of temperature
        gen = OpenAIGenerator(
            model=model or "gpt-5-mini",
            reasoning_effort=reasoning_effort or "medium",
        )
    elif provider == "gemini":
        gen = GeminiGenerator(model=model or "gemini-2.5-flash")
    elif provider == "local":
        gen = LocalGenerator(model=model or "qwen2.5:7b")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    _generator_cache[cache_key] = gen
    logger.info(f"Created new generator: {provider}/{model}")
    return gen


@app.get("/config", response_model=PipelineConfigResponse, tags=["Configuration"])
async def get_config() -> PipelineConfigResponse:
    """
    Get current pipeline configuration.

    Returns:
        Current configuration settings
    """
    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    # Get model name - generators use model_name property
    model = getattr(_pipeline.generator, "model_name", None) or getattr(
        _pipeline.generator, "model", "unknown"
    )

    # Get reasoning effort for GPT-5 models
    reasoning_effort = getattr(_pipeline.generator, "reasoning_effort", "medium")

    return PipelineConfigResponse(
        provider=_pipeline.generator.provider,
        model=model,
        temperature=getattr(_pipeline.generator, "temperature", 0.3),
        reasoning_effort=reasoning_effort,
        use_hyde=_pipeline.config.use_hyde,
        use_multi_query=_pipeline.use_multi_query,
        use_reranking=_pipeline.reranker is not None,
        retrieval_strategy=_pipeline.config.retrieval_strategy,
        enable_self_rag=getattr(_pipeline, "enable_self_rag", False),
        enable_planning=getattr(_pipeline, "enable_planning", False),
    )


@app.put("/config", response_model=PipelineConfigResponse, tags=["Configuration"])
async def update_config(request: PipelineConfigRequest) -> PipelineConfigResponse:
    """
    Update pipeline configuration.

    Changes take effect immediately for subsequent queries.

    Args:
        request: Configuration updates

    Returns:
        Updated configuration settings
    """
    global _pipeline

    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        # Update generator if provider/model/reasoning_effort changed
        if request.provider or request.model or request.reasoning_effort:
            provider = request.provider or _pipeline.generator.provider
            model = request.model or getattr(_pipeline.generator, "model_name", None)
            reasoning_effort = request.reasoning_effort or getattr(
                _pipeline.generator, "reasoning_effort", "medium"
            )
            new_generator = _get_generator(provider, model, reasoning_effort)
            # The built pipeline uses .generator (not ._generator)
            _pipeline.generator = new_generator
            logger.info(
                f"CONFIG: ✓ Updated generator to {provider}/{model} (reasoning_effort={reasoning_effort})"
            )

        # Update retrieval settings
        if request.use_hyde is not None:
            _pipeline.config.use_hyde = request.use_hyde
            logger.info(f"CONFIG: ✓ use_hyde = {request.use_hyde}")

        if request.use_multi_query is not None:
            # The built pipeline uses .use_multi_query (not ._use_multi_query)
            _pipeline.use_multi_query = request.use_multi_query
            logger.info(f"CONFIG: ✓ use_multi_query = {request.use_multi_query}")

        if request.retrieval_strategy is not None:
            _pipeline.config.retrieval_strategy = request.retrieval_strategy
            logger.info(f"CONFIG: ✓ retrieval_strategy = {request.retrieval_strategy}")

        # Update agentic settings (check if attributes exist)
        if request.enable_self_rag is not None:
            if hasattr(_pipeline, "enable_self_rag"):
                _pipeline.enable_self_rag = request.enable_self_rag
            elif hasattr(_pipeline, "config"):
                _pipeline.config.enable_self_rag = request.enable_self_rag
            logger.info(f"CONFIG: ✓ enable_self_rag = {request.enable_self_rag}")

        if request.enable_planning is not None:
            if hasattr(_pipeline, "enable_planning"):
                _pipeline.enable_planning = request.enable_planning
            elif hasattr(_pipeline, "config"):
                _pipeline.config.enable_planning = request.enable_planning
            logger.info(f"CONFIG: ✓ enable_planning = {request.enable_planning}")

        # Log current state after update
        current_model = getattr(_pipeline.generator, "model_name", None) or getattr(
            _pipeline.generator, "model", "unknown"
        )
        logger.info(f"CONFIG: Current generator = {_pipeline.generator.provider}/{current_model}")

        return await get_config()

    except Exception as e:
        logger.error(f"CONFIG: Failed to update: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# Collection Management
# =============================================================================


@app.get("/collections", tags=["Collections"])
async def list_collections() -> dict[str, Any]:
    """
    List all collections.

    Returns:
        List of collection names
    """
    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        collections = await _pipeline.vectordb.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/collections/{name}", response_model=CollectionInfo, tags=["Collections"])
async def get_collection_info(name: str) -> CollectionInfo:
    """
    Get information about a specific collection.

    Args:
        name: Collection name

    Returns:
        Collection statistics
    """
    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        # Check if collection exists
        exists = await _pipeline.vectordb.collection_exists(name)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{name}' not found",
            )

        # Get collection info - handle both dict and object responses
        info = await _pipeline.vectordb.get_collection_info(name)

        # Handle different return types
        if isinstance(info, dict):
            chunk_count = info.get("vectors_count", info.get("points_count", 0))
            created_at = info.get("created_at")
            metadata = info
        else:
            # CollectionInfo object from collection_manager
            chunk_count = getattr(info, "count", 0)
            created_at = None
            metadata = {"dimension": getattr(info, "dimension", 0)}

        return CollectionInfo(
            name=name,
            chunk_count=chunk_count,
            created_at=created_at,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/collections/{name}/documents", tags=["Collections"])
async def list_documents(name: str) -> dict[str, Any]:
    """
    List unique documents in a collection.

    Returns filenames of all uploaded documents.
    Uses optimized scrolling with timeout to handle large collections.
    """
    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        # Check if collection exists
        exists = await _pipeline.vectordb.collection_exists(name)
        if not exists:
            return {"documents": [], "collection": name, "total_chunks": 0}

        # Get collection info for total chunk count (fast operation)
        info = await _pipeline.vectordb.get_collection_info(name)
        total_chunks = info.get("points_count", info.get("vectors_count", 0))

        # Use scrolling with smaller batches and timeout to extract unique filenames
        # This is faster than get_all() for large collections
        client = await _pipeline.vectordb._get_client()

        seen_files: dict[str, dict] = {}
        offset = None
        max_iterations = 50  # Limit iterations to prevent infinite loops
        batch_size = 100  # Smaller batches for faster response

        for _ in range(max_iterations):
            try:
                # Use asyncio.wait_for to add timeout per batch
                results, next_offset = await asyncio.wait_for(
                    client.scroll(
                        collection_name=name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,  # Don't fetch vectors - much faster!
                    ),
                    timeout=5.0,  # 5 second timeout per batch
                )

                if not results:
                    break

                # Extract unique filenames from this batch
                for point in results:
                    payload = point.payload or {}

                    # Try filename first, then extract from source_file (cache key format: "file.pdf_abc123")
                    filename = payload.get("filename")
                    source_file = payload.get("source_file", "")

                    # Fallback: extract filename from source_file if no filename
                    if not filename and source_file:
                        # source_file format is "filename_hash" - extract filename part
                        # e.g., "document.pdf_abc12345" -> "document.pdf"
                        parts = source_file.rsplit("_", 1)
                        if len(parts) == 2 and len(parts[1]) == 8:  # hash is 8 chars
                            filename = parts[0]
                        else:
                            filename = source_file  # Use as-is if format doesn't match

                    # Also try "source" field as another fallback
                    if not filename:
                        source = payload.get("source", "")
                        if source:
                            # Extract just the filename from path
                            filename = source.split("/")[-1].split("\\")[-1]

                    if filename and filename not in seen_files:
                        seen_files[filename] = {
                            "filename": filename,
                            "file_id": payload.get("file_id", source_file),
                            "source_file": source_file,
                            "chunk_count": 0,
                        }
                    if filename:
                        seen_files[filename]["chunk_count"] += 1

                offset = next_offset
                if offset is None:
                    break

            except TimeoutError:
                logger.warning(
                    f"Timeout while scrolling collection {name}, returning partial results"
                )
                break

        return {
            "documents": list(seen_files.values()),
            "collection": name,
            "total_chunks": total_chunks,
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"documents": [], "collection": name, "total_chunks": 0, "error": str(e)}


@app.delete("/collections/{name}", tags=["Collections"])
async def delete_collection(name: str) -> dict[str, Any]:
    """
    Delete a collection.

    Args:
        name: Collection to delete

    Returns:
        Confirmation
    """
    if not _pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )

    try:
        await _pipeline.vectordb.delete_collection(name)
        return {"success": True, "deleted": name}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# UI Endpoint
# =============================================================================


@app.get("/ui", response_class=HTMLResponse, tags=["System"])
async def serve_ui():
    """
    Serve the test UI.

    Open http://localhost:8000/ui in your browser.
    """
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent.parent / "ui" / "index.html",  # src/../ui
        Path(__file__).parent.parent.parent.parent / "ui" / "index.html",  # agentic_rag/ui
        Path.cwd() / "ui" / "index.html",  # current working dir
    ]

    for ui_path in possible_paths:
        if ui_path.exists():
            return ui_path.read_text(encoding="utf-8")

    return HTMLResponse(
        content=f"<h1>UI not found</h1><p>Tried: {[str(p) for p in possible_paths]}</p>",
        status_code=404,
    )
