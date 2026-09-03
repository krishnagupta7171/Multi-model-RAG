from src.ingestion.chunker import RecursiveCharacterSplitter
from src.ingestion.chunker import get_chunker
from src.ingestion.chunker import SemanticChunker
from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import Document



def test_recursive_splitter():
    splitter = RecursiveCharacterSplitter(
        chunk_size=50,
        chunk_overlap=10,
    )

    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence."
    )

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)




def test_semantic_chunker():
    chunker = SemanticChunker(
        chunk_size=50,
        chunk_overlap=10,
    )

    text = (
        "This is the first sentence. "
        "This is the second sentence. "
        "This is the third sentence."
    )

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)




def test_document_chunker():
    document = Document(
        content="This is a test document. " * 50,
        metadata={"source": "test.txt"},
        doc_id="test_doc",
    )

    chunker = DocumentChunker(chunk_size=100,chunk_overlap=20,)

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["total_chunks"] == len(chunks)
    assert chunks[0].chunk_id == "test_doc_0"




def test_get_chunker():
    chunker = get_chunker()

    assert isinstance(chunker, DocumentChunker)
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 50