import asyncio
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.multimodal_agent import MultimodalRAGAgent
from src.generation.LLMGenerator import LLMGenerator
from src.observability.logging import get_logger
from src.retrieval.vector_store import get_retriever

logger = get_logger(__name__)
console = Console()

app = typer.Typer()


@app.command()
def benchmark(num_queries: int = typer.Option(5, help="Number of queries to benchmark"),
    collection: str = typer.Option("documents", help="Vector collection name"),
    skip_llm: bool = typer.Option(False, help="Run retrieval-only benchmark without LLM"),):
    asyncio.run(_benchmark_async(num_queries, collection, skip_llm))


async def _benchmark_async(num_queries: int, collection: str, skip_llm: bool):
    
    console.print("\n[bold blue]RAG Agent Performance Benchmark[/bold blue]")
    console.print(f"Target Collection : [cyan]{collection}[/cyan]")
    console.print(f"Query Count       : [cyan]{num_queries}[/cyan]")
    console.print(f"LLM Synthesis     : [cyan]{'Disabled (Retrieval Only)' if skip_llm else 'Enabled'}[/cyan]\n")

    queries = [
        "What are the core capabilities of this system?",
        "Explain how multimodal document ingestion works.",
        "How is context retrieved and reranked?",
        "What evaluation metrics are monitored during inference?",
        "Summarize the architecture workflow.",
    ]

    try:
        console.print("[yellow]Initializing retriever and agent components...[/yellow]")
        retriever = await get_retriever(collection_name=collection)
        llm_client = None if skip_llm else LLMGenerator()

        agent = MultimodalRAGAgent(retriever=retriever,llm_client=llm_client,use_reflection=not skip_llm,)

        console.print("[yellow]Executing warm-up query...[/yellow]")
        await agent.run(query=queries[0])
        console.print("[green]Warm-up finished.[/green]\n")

        console.print(f"[yellow]Running {num_queries} benchmark cycles...[/yellow]")
        latencies = []
        doc_counts = []

        for i in range(num_queries):
            query = queries[i % len(queries)]
            start_time = time.perf_counter()
            result = await agent.run(query=query)
            elapsed = time.perf_counter() - start_time

            docs_retrieved = len(result.get("documents", []))
            latencies.append(elapsed)
            doc_counts.append(docs_retrieved)

            console.print(
                f"  Query {i + 1}/{num_queries}: [bold cyan]{elapsed:.3f}s[/bold cyan] "
                f"({docs_retrieved} docs retrieved) -> '{query[:35]}...'"
            )

        _render_results(latencies, doc_counts)

    except Exception as e:
        console.print(f"\n[bold red]Benchmark run failed: {e}[/bold red]\n")
        logger.error(f"Benchmark error: {e}", exc_info=True)
        sys.exit(1)


def _render_results(latencies: list, doc_counts: list):
    console.print("\n[bold green]Benchmark Performance Results[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    total_time = sum(latencies)
    throughput = len(latencies) / total_time if total_time > 0 else 0.0

    table.add_row("Total Executions", str(len(latencies)))
    table.add_row("Mean Latency", f"{mean(latencies):.3f}s")
    table.add_row("Median Latency", f"{median(latencies):.3f}s")
    table.add_row("Min Latency", f"{min(latencies):.3f}s")
    table.add_row("Max Latency", f"{max(latencies):.3f}s")

    if len(latencies) > 1:
        table.add_row("Std Dev Latency", f"{stdev(latencies):.3f}s")

    table.add_row("Avg Docs Retrieved", f"{mean(doc_counts):.1f}")
    table.add_row("Throughput", f"{throughput:.2f} queries/s")

    console.print(table)
    console.print()


if __name__ == "__main__":
    app()