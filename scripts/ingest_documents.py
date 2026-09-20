import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.deps import get_retriever
from src.ingestion.chunker import DocumentChunker, get_chunker
from src.ingestion.loader import DocumentLoader
from src.observability.logging import get_logger

logger = get_logger(__name__)
console = Console()

app = typer.Typer()


@app.command()
def ingest(directory: str = typer.Argument(..., help="Directory containing documents"),
    collection: str = typer.Option("documents", help="Collection name"),
    recursive: bool = typer.Option(True, help="Recursively scan subdirectories"),):
    asyncio.run(_ingest_async(directory, collection, recursive))


async def _ingest_async(directory: str, collection: str, recursive: bool):
    dir_path = Path(directory)
    if not dir_path.exists():
        console.print(f"[red]Directory '{directory}' does not exist![/red]")
        return

    console.print("\n[bold blue]Starting document ingestion[/bold blue]")
    console.print(f"Directory: {directory}")
    console.print(f"Collection: {collection}")
    console.print(f"Recursive: {recursive}\n")

    try:
        console.print("[yellow]Loading documents...[/yellow]")
        loader = DocumentLoader()

        pattern = "**/*" if recursive else "*"
        supported_exts = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg"}
        file_paths = [
            p for p in dir_path.glob(pattern) if p.is_file() and p.suffix.lower() in supported_exts
        ]

        if not file_paths:
            console.print("[red]No supported documents found![/red]")
            return

        documents = []
        for file_path in file_paths:
            try:
                docs = loader.load(str(file_path))
                if isinstance(docs, list):
                    documents.extend(docs)
                elif docs:
                    documents.append(docs)
            except Exception as e:
                logger.warning(f"Skipping {file_path.name}: {e}")

        if not documents:
            console.print("[red]No valid content extracted from files![/red]")
            return

        console.print(f"[green]Loaded {len(documents)} document objects[/green]")

        console.print("[yellow]Chunking documents...[/yellow]")
        chunker = get_chunker()
        chunks = chunker.chunk_documents(documents)
        console.print(f"[green]Created {len(chunks)} chunks[/green]")

        console.print("[yellow]Initializing retriever and vector store...[/yellow]")
        retriever = await get_retriever(collection_name=collection)

        console.print("[yellow]Ingesting into vector store...[/yellow]")

        with Progress() as progress:
            task = progress.add_task("[cyan]Ingesting...", total=len(chunks))

            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]

                texts = [getattr(c, "text", str(c)) for c in batch]
                metadatas = [getattr(c, "metadata", {}) for c in batch]
                ids = [
                    getattr(c, "chunk_id", None)
                    or getattr(c, "id", None)
                    or f"doc_{i + idx}"
                    for idx, c in enumerate(batch)
                ]

                if hasattr(retriever, "add_texts"):
                    await retriever.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                elif hasattr(retriever, "vector_store") and hasattr(retriever.vector_store, "add_texts"):
                    retriever.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                elif hasattr(retriever, "add_documents"):
                    await retriever.add_documents(batch)

                progress.update(task, advance=len(batch))

        console.print(
            f"\n[bold green]✓ Successfully ingested {len(documents)} documents "
            f"({len(chunks)} chunks) into collection '{collection}'[/bold green]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]✗ Ingestion failed: {e}[/bold red]\n")
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app()