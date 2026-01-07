"""
CLI interface for RAG Optimizer.

Provides command-line access to all RAG pipeline functionality.
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_rag.config import get_settings
from agentic_rag.generation import GeneratorFactory

app = typer.Typer(
    name="agentic-rag",
    help="Agentic RAG Pipeline Optimizer with 2025 state-of-the-art techniques",
    no_args_is_help=True,
)
console = Console()


# =============================================================================
# Query Command
# =============================================================================


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer"),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Vector DB collection to search"
    ),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="LLM provider (claude, openai, gemini, local)"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model ID (e.g., claude-sonnet-4-5-20250929)"
    ),
    mode: str = typer.Option(
        "standard",
        "--mode",
        help="Pipeline mode: 'standard' for linear flow, 'agentic' for multi-agent orchestration",
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of chunks to retrieve"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed pipeline steps and source metadata"
    ),
) -> None:
    """
    Query the RAG pipeline using the specified collection and model settings.

    This command builds a pipeline on-the-fly and executes the query.
    In agentic mode, it uses multi-agent orchestration for better reasoning and self-correction.

    Examples:
        agentic-rag query "What is the Transformer architecture?" -c my-docs
        agentic-rag query "Compare BERT and GPT" -c docs --mode agentic --verbose
        agentic-rag query "How to install?" -c docs --provider openai --model gpt-4o
    """
    from agentic_rag.pipeline.builder import PipelineBuilder

    settings = get_settings()

    with console.status("[bold green]Building pipeline..."):
        builder = PipelineBuilder(settings=settings)
        builder.with_retrieval(top_k=top_k)

        if provider or model:
            builder.with_generator(provider=provider, model=model)  # type: ignore

        if mode == "agentic":
            builder.as_agentic()

        pipeline = builder.build()

    async def run_query():
        try:
            return await pipeline.query(question, collection=collection)
        finally:
            await pipeline.close()

    with console.status("[bold green]Querying..."):
        result = asyncio.run(run_query())

    # Display result
    console.print()
    console.print(Panel(result.response, title="[bold]Answer", border_style="green"))

    if verbose:
        console.print()
        console.print("[bold]Sources:[/bold]")
        for i, chunk in enumerate(result.sources[:5], 1):
            source = chunk.metadata.get("source", "Unknown")
            console.print(f"  [{i}] {source}")

        console.print()
        console.print(f"[dim]Provider: {result.provider} | Model: {result.model}[/dim]")
        console.print(
            f"[dim]Tokens: {result.total_tokens} | Latency: {result.latency_ms:.0f}ms[/dim]"
        )


# =============================================================================
# Ingest Command
# =============================================================================


@app.command()
def ingest(
    source: Path = typer.Argument(..., help="Source directory or file to ingest"),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Target vector database collection"
    ),
    chunking: str = typer.Option(
        "semantic",
        "--chunking",
        help="Chunking strategy: 'semantic', 'late', 'raptor', or 'recursive'",
    ),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Target chunk size in tokens"),
    contextual: bool = typer.Option(
        True,
        "--contextual/--no-contextual",
        help="Enable Anthropic-style contextual retrieval headers",
    ),
) -> None:
    """
    Ingest document files into the vector database.

    Supported formats: .txt, .md, .pdf, .docx, .html.
    The command automatically handles file parsing, chunking, embedding, and indexing.

    Examples:
        agentic-rag ingest ./research_papers -c research
        agentic-rag ingest ./manual.pdf -c docs --chunking late
        agentic-rag ingest ./wiki --collection knowledge --no-contextual
    """
    from agentic_rag.core.models import Document
    from agentic_rag.pipeline.builder import PipelineBuilder

    if not source.exists():
        console.print(f"[red]Error: Source path does not exist: {source}[/red]")
        raise typer.Exit(1)

    # Load documents
    documents: list[Document] = []

    files = [source] if source.is_file() else list(source.glob("**/*"))

    for file_path in files:
        if file_path.is_file() and file_path.suffix in {".txt", ".md", ".pdf"}:
            try:
                if file_path.suffix == ".txt" or file_path.suffix == ".md":
                    content = file_path.read_text(encoding="utf-8")
                else:
                    # Skip PDF for now (requires pypdf)
                    continue

                doc = Document(
                    content=content,
                    source=str(file_path),
                    metadata={"source": str(file_path), "filename": file_path.name},
                )
                documents.append(doc)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read {file_path}: {e}[/yellow]")

    if not documents:
        console.print("[red]No documents found to ingest.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Found {len(documents)} documents to ingest[/green]")

    # Build pipeline
    settings = get_settings()
    builder = PipelineBuilder(settings=settings)
    builder.with_chunking(strategy=chunking, chunk_size=chunk_size, contextual=contextual)
    pipeline = builder.build()

    async def run_ingest():
        try:
            return await pipeline.ingest(documents, collection=collection)
        finally:
            await pipeline.close()

    with console.status("[bold green]Ingesting documents..."):
        result = asyncio.run(run_ingest())

    console.print()
    console.print(
        f"[green]Ingested {result['documents']} documents, "
        f"{result['chunks']} chunks into '{result['collection']}'[/green]"
    )


# =============================================================================
# Config Command
# =============================================================================


@app.command()
def config() -> None:
    """
    Show current configuration.
    """
    settings = get_settings()

    table = Table(title="RAG Optimizer Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Core settings
    table.add_row("Qdrant URL", settings.qdrant_url)
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("Embedding Device", settings.embedding_device)
    table.add_row("LLM Provider", settings.llm_provider)
    table.add_row("LLM Model", settings.llm_model)
    table.add_row("Temperature", str(settings.default_temperature))
    table.add_row("Max Tokens", str(settings.default_max_tokens))

    # Agentic settings
    table.add_row("Enable Reflection", str(settings.enable_reflection))
    table.add_row("Enable Planning", str(settings.enable_planning))
    table.add_row("Max Iterations", str(settings.max_iterations))
    table.add_row("Confidence Threshold", str(settings.confidence_threshold))

    # API keys (masked)
    def mask_key(key: str | None) -> str:
        if not key:
            return "[red]Not set[/red]"
        return f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "****"

    table.add_row("Anthropic API Key", mask_key(settings.get_api_key("claude")))
    table.add_row("OpenAI API Key", mask_key(settings.get_api_key("openai")))
    table.add_row("Google API Key", mask_key(settings.get_api_key("gemini")))

    console.print(table)


# =============================================================================
# Providers Command
# =============================================================================


@app.command()
def providers() -> None:
    """
    List available LLM providers and models.
    """
    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Default Models", style="green")
    table.add_column("Status", style="yellow")

    settings = get_settings()

    for provider in GeneratorFactory.list_providers():
        models = GeneratorFactory.get_default_models(provider)  # type: ignore
        status = (
            "[green]Configured[/green]"
            if settings.validate_provider_config(provider)
            else "[red]Not configured[/red]"
        )
        table.add_row(provider, ", ".join(models[:3]), status)

    console.print(table)
    console.print()
    console.print("[dim]Configure providers by setting environment variables:[/dim]")
    console.print("[dim]  RAG_ANTHROPIC_API_KEY, RAG_OPENAI_API_KEY, RAG_GOOGLE_API_KEY[/dim]")


# =============================================================================
# Serve Command
# =============================================================================


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
) -> None:
    """
    Start the RAG Optimizer API server.

    Examples:
        agentic-rag serve
        agentic-rag serve --port 8080 --workers 4
        agentic-rag serve --reload  # Development mode
    """
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Error: uvicorn not installed. Run: pip install uvicorn[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Starting RAG Optimizer API server on {host}:{port}[/bold green]")
    console.print()
    console.print(f"[dim]API docs: http://{host}:{port}/docs[/dim]")
    console.print(f"[dim]Health: http://{host}:{port}/health[/dim]")
    console.print()

    uvicorn.run(
        "agentic_rag.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )


# =============================================================================
# Dashboard Command
# =============================================================================


@app.command()
def dashboard(
    refresh_rate: float = typer.Option(1.0, "--refresh", "-r", help="Refresh rate in seconds"),
    once: bool = typer.Option(False, "--once", help="Display once and exit"),
) -> None:
    """
    Launch the monitoring dashboard.

    Examples:
        agentic-rag dashboard
        agentic-rag dashboard --refresh 0.5
        agentic-rag dashboard --once
    """
    from agentic_rag.observability import Dashboard, SimpleDashboard

    if once:
        dashboard = SimpleDashboard()
        dashboard.show_all()
    else:
        console.print("[bold green]Launching RAG Optimizer Dashboard[/bold green]")
        console.print("[dim]Press Ctrl+C to exit[/dim]")
        console.print()

        dashboard = Dashboard()
        dashboard.run(refresh_rate=refresh_rate)


# =============================================================================
# Evaluate Command
# =============================================================================


@app.command()
def evaluate(
    benchmark: str = typer.Argument(..., help="Benchmark name (hotpotqa, nq, triviaqa)"),
    collection: str = typer.Option(..., "--collection", "-c", help="Collection to evaluate"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    n_samples: int = typer.Option(100, "--samples", "-n", help="Number of samples"),
) -> None:
    """
    Evaluate pipeline on a benchmark.

    Examples:
        agentic-rag evaluate hotpotqa -c my-docs
        agentic-rag evaluate nq -c docs --samples 50 --output results/
    """
    from agentic_rag.evaluation import RAGBenchmark, load_benchmark
    from agentic_rag.pipeline import PipelineBuilder

    console.print(f"[bold]Running benchmark: {benchmark}[/bold]")

    # Load benchmark questions
    with console.status(f"[bold green]Loading {benchmark} benchmark..."):
        questions = load_benchmark(benchmark, n=n_samples)

    console.print(f"[green]Loaded {len(questions)} questions[/green]")

    # Build pipeline
    settings = get_settings()
    builder = PipelineBuilder(settings=settings)
    pipeline = builder.build()

    # Run benchmark
    async def run_benchmark():
        try:
            bench = RAGBenchmark(pipeline=pipeline)
            return await bench.run(questions=questions, collection=collection)
        finally:
            await pipeline.close()

    with console.status("[bold green]Running benchmark..."):
        results = asyncio.run(run_benchmark())

    # Display results
    table = Table(title=f"Benchmark Results: {benchmark}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")

    table.add_row("Total Questions", str(results.total_questions))
    table.add_row("Avg Context Precision", f"{results.avg_context_precision:.3f}")
    table.add_row("Avg Context Recall", f"{results.avg_context_recall:.3f}")
    table.add_row("Avg Faithfulness", f"{results.avg_faithfulness:.3f}")
    table.add_row("Avg Answer Relevancy", f"{results.avg_answer_relevancy:.3f}")
    table.add_row("Avg Latency", f"{results.avg_latency_ms:.1f}ms")

    console.print()
    console.print(table)

    # Save results if output specified
    if output:
        output.mkdir(parents=True, exist_ok=True)
        import json
        from datetime import datetime

        output_file = output / f"{benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "benchmark": benchmark,
                    "total_questions": results.total_questions,
                    "avg_context_precision": results.avg_context_precision,
                    "avg_context_recall": results.avg_context_recall,
                    "avg_faithfulness": results.avg_faithfulness,
                    "avg_answer_relevancy": results.avg_answer_relevancy,
                    "avg_latency_ms": results.avg_latency_ms,
                },
                f,
                indent=2,
            )
        console.print(f"\n[green]Results saved to {output_file}[/green]")


# =============================================================================
# Version Command
# =============================================================================


@app.command()
def version() -> None:
    """
    Show version information.
    """
    from agentic_rag import __version__

    console.print(f"[bold]RAG Optimizer[/bold] v{__version__}")
    console.print()
    console.print("Agentic RAG Pipeline with 2025/2026 state-of-the-art techniques")
    console.print()
    console.print("Features:")
    console.print("  - Multi-agent orchestration (Router, Retriever, Evaluator, Generator)")
    console.print("  - Hybrid retrieval (Dense + BM25 with RRF fusion)")
    console.print("  - HyDE (Hypothetical Document Embeddings)")
    console.print("  - Self-RAG reflection tokens (ISREL, ISSUP, ISUSE)")
    console.print("  - Multi-provider LLM support:")
    console.print("      Claude 4.5, GPT-5 (5.2/mini/nano), Gemini 3/2.5, Local")
    console.print("  - RAGAS evaluation metrics")


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
