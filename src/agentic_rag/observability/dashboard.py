"""
Rich CLI dashboard for RAG pipeline monitoring.

Provides a terminal-based dashboard for viewing
pipeline metrics and traces in real-time.
"""

from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentic_rag.observability.metrics import RAGMetrics, get_metrics
from agentic_rag.observability.tracer import RAGTracer, get_tracer


class Dashboard:
    """
    Rich terminal dashboard for RAG monitoring.

    Displays real-time metrics, recent traces, and
    pipeline health status.
    """

    def __init__(
        self,
        metrics: RAGMetrics | None = None,
        tracer: RAGTracer | None = None,
    ):
        """
        Initialize dashboard.

        Args:
            metrics: Metrics collector.
            tracer: Tracer instance.
        """
        self._metrics = metrics or get_metrics()
        self._tracer = tracer or get_tracer()
        self._console = Console()

    def _create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="traces", ratio=1),
        )

        return layout

    def _render_header(self) -> Panel:
        """Render header panel."""
        text = Text()
        text.append("RAG Optimizer Dashboard", style="bold blue")
        text.append(" | ", style="dim")
        text.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style="dim")

        return Panel(text, style="blue")

    def _render_metrics(self) -> Panel:
        """Render metrics panel."""
        table = Table(title="Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        all_metrics = self._metrics.get_all_metrics()

        # Add counters
        for name, value in all_metrics.get("counters", {}).items():
            table.add_row(name, str(value))

        # Add summary stats
        for name, summary in all_metrics.get("summaries", {}).items():
            if summary["count"] > 0:
                table.add_row(
                    name,
                    f"avg={summary['mean']:.2f} (n={summary['count']})",
                )

        return Panel(table, title="[bold]Metrics[/bold]")

    def _render_traces(self) -> Panel:
        """Render traces panel."""
        table = Table(title="Recent Traces", show_header=True)
        table.add_column("Span", style="cyan")
        table.add_column("Duration", style="yellow")
        table.add_column("Status", style="green")

        for span in self._tracer.get_recent_spans(10):
            status_style = "green" if span.status == "ok" else "red"
            table.add_row(
                span.name,
                f"{span.duration_ms:.1f}ms",
                Text(span.status, style=status_style),
            )

        return Panel(table, title="[bold]Recent Traces[/bold]")

    def _render_footer(self) -> Panel:
        """Render footer panel."""
        text = Text()
        text.append("Press ", style="dim")
        text.append("Ctrl+C", style="bold")
        text.append(" to exit", style="dim")

        return Panel(text, style="dim")

    def render(self) -> Layout:
        """Render the full dashboard."""
        layout = self._create_layout()

        layout["header"].update(self._render_header())
        layout["metrics"].update(self._render_metrics())
        layout["traces"].update(self._render_traces())
        layout["footer"].update(self._render_footer())

        return layout

    def show(self) -> None:
        """Display the dashboard once."""
        self._console.print(self.render())

    def run(self, refresh_rate: float = 1.0) -> None:
        """
        Run the dashboard with live updates.

        Args:
            refresh_rate: Refresh rate in seconds.
        """
        try:
            with Live(
                self.render(),
                console=self._console,
                refresh_per_second=1 / refresh_rate,
            ) as live:
                while True:
                    live.update(self.render())
        except KeyboardInterrupt:
            pass


class SimpleDashboard:
    """
    Simple one-shot dashboard display.

    Prints a summary of current metrics and traces.
    """

    def __init__(self):
        """Initialize simple dashboard."""
        self._console = Console()
        self._metrics = get_metrics()

    def show_metrics(self) -> None:
        """Display current metrics."""
        table = Table(title="RAG Optimizer Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Avg", style="yellow")
        table.add_column("Min", style="dim")
        table.add_column("Max", style="dim")

        all_metrics = self._metrics.get_all_metrics()

        for name, summary in all_metrics.get("summaries", {}).items():
            if summary["count"] > 0:
                table.add_row(
                    name,
                    str(summary["count"]),
                    f"{summary['mean']:.2f}",
                    f"{summary['min']:.2f}" if summary["min"] else "-",
                    f"{summary['max']:.2f}" if summary["max"] else "-",
                )

        self._console.print(table)

    def show_counters(self) -> None:
        """Display counters."""
        table = Table(title="Counters")
        table.add_column("Counter", style="cyan")
        table.add_column("Value", style="green")

        all_metrics = self._metrics.get_all_metrics()

        for name, value in all_metrics.get("counters", {}).items():
            table.add_row(name, str(value))

        self._console.print(table)

    def show_all(self) -> None:
        """Display all metrics."""
        self.show_counters()
        self._console.print()
        self.show_metrics()


def show_dashboard() -> None:
    """Show the simple dashboard."""
    dashboard = SimpleDashboard()
    dashboard.show_all()


def run_dashboard(refresh_rate: float = 1.0) -> None:
    """Run the live dashboard."""
    dashboard = Dashboard()
    dashboard.run(refresh_rate)
