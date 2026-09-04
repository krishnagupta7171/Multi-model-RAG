import pytest

from src.database.vector_store import ChromaVectorStore


@pytest.mark.asyncio
async def test_create_and_search_collection(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma")
    )

    collection_name = "test_collection"

    await store.create_collection(
        name=collection_name,
        dimension=3,
    )

    await store.add_documents(
        collection_name=collection_name,
        ids=["doc_1", "doc_2"],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        documents=[
            "This is document one.",
            "This is document two.",
        ],
        metadata=[
            {"source": "test1.txt"},
            {"source": "test2.txt"},
        ],
    )

    results = await store.search(
        collection_name=collection_name,
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0] == "doc_1"
    assert results[0][2]["text"] == "This is document one."


@pytest.mark.asyncio
async def test_delete_documents(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma")
    )

    collection_name = "delete_test"

    await store.create_collection(
        name=collection_name,
        dimension=3,
    )

    await store.add_documents(
        collection_name=collection_name,
        ids=["doc_1"],
        embeddings=[[1.0, 0.0, 0.0]],
        documents=["Test document"],
        metadata=[{"source": "test.txt"}],
    )

    await store.delete_documents(
        collection_name=collection_name,
        ids=["doc_1"],
    )

    results = await store.search(
        collection_name=collection_name,
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 0