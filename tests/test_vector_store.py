from unittest.mock import AsyncMock, MagicMock
import pytest

from src.retrieval.vector_store import VectorStoreRetriever


@pytest.fixture
def mock_embedding_service():
    service = MagicMock()
    service.embed_batch_async = AsyncMock(
        return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    service.embed_text_async = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return service


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.create_collection = AsyncMock()
    store.add_documents = AsyncMock()
    store.search = AsyncMock(
        return_value=[
            ("doc_1", 0.95, {"text": "Alpha document", "metadata": {"source": "a.txt"}}),
            ("doc_2", 0.80, {"text": "Beta document", "metadata": {"source": "b.txt"}}),
        ]
    )
    store.delete_documents = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_retriever_add_texts(mock_vector_store, mock_embedding_service):
    retriever = VectorStoreRetriever(
        collection_name="test_docs",
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
    )

    texts = ["Alpha document", "Beta document"]
    metadata = [{"source": "a.txt"}, {"source": "b.txt"}]

    ids = await retriever.add_texts(texts=texts, metadata=metadata)

    assert len(ids) == 2
    mock_embedding_service.embed_batch_async.assert_awaited_once_with(texts)
    mock_vector_store.add_documents.assert_awaited_once()


@pytest.mark.asyncio
async def test_retriever_empty_metadata_sanitization(mock_vector_store, mock_embedding_service):
    retriever = VectorStoreRetriever(
        collection_name="test_docs",
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
    )

    texts = ["Sample text"]
    await retriever.add_texts(texts=texts, metadata=[{}])

    call_args = mock_vector_store.add_documents.call_args[1]
    assert call_args["metadata"] == [{"source": "unknown"}]


@pytest.mark.asyncio
async def test_retriever_similarity_search(mock_vector_store, mock_embedding_service):
    retriever = VectorStoreRetriever(
        collection_name="test_docs",
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
    )

    results = await retriever.similarity_search("find alpha", top_k=2)

    assert len(results) == 2
    assert results[0][0] == "doc_1"
    assert results[0][1] == 0.95
    assert results[0][2] == "Alpha document"


@pytest.mark.asyncio
async def test_retriever_get_relevant_documents(mock_vector_store, mock_embedding_service):
    retriever = VectorStoreRetriever(
        collection_name="test_docs",
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
    )

    docs = await retriever.get_relevant_documents("find alpha", top_k=2)

    assert docs == ["Alpha document", "Beta document"]
