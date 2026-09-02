import pytest

from src.ingestion.loader import (
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    CSVLoader,
    JSONLoader,
)


@pytest.mark.asyncio
async def test_text_loader(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("Hello RAG", encoding="utf-8")

    documents = await TextLoader().load(str(file))

    assert len(documents) == 1
    assert documents[0].content == "Hello RAG"
    assert documents[0].metadata["type"] == "text"


@pytest.mark.asyncio
async def test_markdown_loader(tmp_path):
    file = tmp_path / "test.md"
    file.write_text("# Hello RAG", encoding="utf-8")

    documents = await MarkdownLoader().load(str(file))

    assert len(documents) == 1
    assert documents[0].metadata["type"] == "markdown"


@pytest.mark.asyncio
async def test_html_loader(tmp_path):
    file = tmp_path / "test.html"
    file.write_text(
        "<html><body><h1>Hello RAG</h1></body></html>",
        encoding="utf-8",
    )

    documents = await HTMLLoader().load(str(file))

    assert len(documents) == 1
    assert "Hello RAG" in documents[0].content


@pytest.mark.asyncio
async def test_csv_loader(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text(
        "name,age\nKrishna,22\nRahul,23",
        encoding="utf-8",
    )

    documents = await CSVLoader().load(str(file))

    assert len(documents) == 2
    assert documents[0].metadata["type"] == "csv"


@pytest.mark.asyncio
async def test_json_loader(tmp_path):
    file = tmp_path / "test.json"
    file.write_text(
        '{"name": "Krishna", "type": "RAG"}',
        encoding="utf-8",
    )

    documents = await JSONLoader().load(str(file))

    assert len(documents) == 1
    assert documents[0].metadata["type"] == "json"
    assert "Krishna" in documents[0].content