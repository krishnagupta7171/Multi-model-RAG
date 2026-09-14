#!/usr/bin/env python3
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.multimodal_agent import MultimodalRAGAgent
from src.evaluation.rag_eval import (RAGMetricsTracker,compute_answer_relevancy,compute_context_precision,compute_context_recall,)
from src.observability.logging import get_logger, setup_logging
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_store import get_retriever

setup_logging()
logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Run automated quality and performance benchmarks on the Multi-Modal RAG system.")


@app.command()
def evaluate(
    test_file: str = typer.Argument(
        "configs/evaluation/sample_queries.json",
        help="Path to evaluation test cases JSON file",
    ),
    collection: str = typer.Option("documents", help="Vector store collection to query"),
    use_reranker: bool = typer.Option(True, help="Enable combined reranker during retrieval"),
):
    # Entry point for the evaluation pipeline. It runs the evaluation asynchronously and handles exceptions.    

    asyncio.run(_evaluate_async(test_file, collection, use_reranker))


async def _evaluate_async(test_file: str, collection: str, use_reranker: bool):
    
    console.print("\n[bold blue]=== Multi-Modal RAG Quality Evaluation Pipeline ===[/bold blue]")
    console.print(f"Test queries dataset: [cyan]{test_file}[/cyan]")
    console.print(f"Target collection:    [cyan]{collection}[/cyan]")
    console.print(f"Reranking enabled:    [cyan]{use_reranker}[/cyan]\n")

    try:
        test_path = Path(test_file)
        if not test_path.exists():
            _generate_sample_test_file(test_path)
            console.print(f"[yellow]Generated initial test template at: {test_file}[/yellow]\n")

        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        queries_list = test_data.get("queries", [])
        if not queries_list:
            console.print("[bold red]Evaluation aborted: No test queries found in file.[/bold red]")
            return

        # Initialize Retriever and Multimodal Agent
        console.print("[yellow]Initializing Vector Store Retriever and Agent components...[/yellow]")
        retriever = await get_retriever(collection_name=collection)
        reranker = get_reranker("combined") if use_reranker else None

        agent = MultimodalRAGAgent(retriever=retriever,reranker=reranker,use_reflection=True,min_reflection_score=6.0,max_retries=2,)

        metrics_tracker = RAGMetricsTracker()
        run_results: List[Dict[str, Any]] = []

        console.print(f"[yellow]Running evaluation across {len(queries_list)} queries...[/yellow]\n")

        for idx, item in enumerate(queries_list, start=1):
            query_str = item["query"]
            expected_kws = item.get("expected_keywords", [])
            ground_truth_ids = item.get("relevant_docs", [])

            console.print(f"[bold cyan][{idx}/{len(queries_list)}][/bold cyan] Evaluating: '{query_str}'")

            # Run agent query loop
            output = await agent.run(query=query_str)

            generated_answer = output.get("answer", "")
            docs_retrieved = output.get("documents", [])
            retrieved_ids = [d.get("id", "") for d in docs_retrieved]

            # Compute metric scores
            relevancy = compute_answer_relevancy(query=query_str,answer=generated_answer,expected_keywords=expected_kws,)

            precision = (
                compute_context_precision(retrieved_ids, ground_truth_ids)
                if ground_truth_ids
                else None)
            recall = (
                compute_context_recall(retrieved_ids, ground_truth_ids)
                if ground_truth_ids
                else None
            )

            reflection_dict = output.get("reflection", {})
            reflection_score = reflection_dict.get("score") if isinstance(reflection_dict, dict) else None

            metrics_tracker.record_run(num_docs=len(docs_retrieved),reflection_score=reflection_score,relevancy=relevancy,precision=precision,recall=recall,)

            run_results.append(
                {
                    "query": query_str,
                    "answer_snippet": (
                        (generated_answer[:80] + "...")
                        if len(generated_answer) > 80
                        else generated_answer
                    ),
                    "docs_count": len(docs_retrieved),
                    "relevancy": relevancy,
                    "precision": precision,
                    "recall": recall,
                    "reflection": reflection_score,
                }
            )

        _display_evaluation_report(run_results, metrics_tracker)

    except Exception as e:
        console.print(f"\n[bold red]Pipeline evaluation failed: {e}[/bold red]\n")
        logger.error(f"Evaluation script error: {e}", exc_info=True)
        sys.exit(1)


def _display_evaluation_report(results: List[Dict[str, Any]], tracker: RAGMetricsTracker) -> None:
    # format eval data into table and summary
    console.print("\n[bold green]=== Query-Level Performance Results ===[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Query", style="cyan", width=36)
    table.add_column("Docs", justify="center", width=6)
    table.add_column("Relevancy", justify="right", width=12)
    table.add_column("Precision", justify="right", width=12)
    table.add_column("Recall", justify="right", width=12)
    table.add_column("Reflection", justify="right", width=12)

    for row in results:
        table.add_row(
            row["query"][:33] + "..." if len(row["query"]) > 36 else row["query"],
            str(row["docs_count"]),
            f"{row['relevancy']:.2f}" if row["relevancy"] is not None else "-",
            f"{row['precision']:.2f}" if row["precision"] is not None else "-",
            f"{row['recall']:.2f}" if row["recall"] is not None else "-",
            f"{row['reflection']:.1f}" if row["reflection"] is not None else "-",
        )

    console.print(table)

    summary = tracker.get_summary()
    console.print("\n[bold]Aggregated Metric Summary:[/bold]")
    console.print(f"• Total Queries Executed:   [bold]{summary['total_queries']}[/bold]")
    console.print(f"• Mean Docs / Query:        {summary['avg_docs_per_query']:.2f}")
    console.print(f"• Mean Reflection Score:    {summary['avg_reflection_score']:.2f} / 10.0")
    console.print(f"• Mean Relevancy:           {summary['avg_relevancy']:.2%}")
    if summary["avg_precision"] > 0.0:
        console.print(f"• Mean Context Precision:   {summary['avg_precision']:.2%}")
    if summary["avg_recall"] > 0.0:
        console.print(f"• Mean Context Recall:      {summary['avg_recall']:.2%}")
    console.print("")


def _generate_sample_test_file(path: Path) -> None:
    
    sample_data = {
        "queries": [
            {
                "query": "What are vector embeddings and how do they capture semantic meaning?",
                "expected_keywords": ["vector", "embeddings", "semantic", "dimension"],
                "relevant_docs": [],
            },
            {
                "query": "How does reciprocal rank fusion combine dense and sparse search?",
                "expected_keywords": ["reciprocal", "rank", "fusion", "sparse", "dense"],
                "relevant_docs": [],
            },
            {
                "query": "Explain how agent reflection improves response accuracy",
                "expected_keywords": ["reflection", "accuracy", "critique", "score"],
                "relevant_docs": [],
            },
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2)


if __name__ == "__main__":
    app()







