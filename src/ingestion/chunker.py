import re

from typing import Any, Dict

from typing import Optional

from ..utils.config import get_settings
from .loader import Document

class TextChunk:
    #Text chunk with metadata

    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_id: str,
    ):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id

    def __repr__(self) -> str:
        return (
            f"TextChunk("
            f"id={self.chunk_id}, "
            f"length={len(self.text)}"
            f")"
        )

class RecursiveCharacterSplitter:
    #Recursive character-based text splitter.

    def __init__(self,chunk_size: int = 512,chunk_overlap: int = 50,separators: list[str] | None = None,):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]

    def split_text(self, text: str) -> list[str]:
        #Split text into chunks recursively.

        if len(text) <= self.chunk_size:
            return [text]

        for separator in self.separators:
            if separator == "":
                return self._split_by_character(text)

            if separator in text:
                splits = text.split(separator)

                chunks = []
                current_chunk = []
                current_length = 0

                for split in splits:
                    split_length = len(split) + len(separator)

                    if current_length + split_length > self.chunk_size:
                        if current_chunk:
                            chunk_text = separator.join(current_chunk)
                            chunks.append(chunk_text)

                            if self.chunk_overlap > 0:
                                overlap_text = chunk_text[
                                    -self.chunk_overlap:
                                ]

                                current_chunk = [
                                    overlap_text,
                                    split,
                                ]

                                current_length = (
                                    len(overlap_text) + split_length
                                )
                            else:
                                current_chunk = [split]
                                current_length = split_length
                        else:
                            sub_chunks = self.split_text(split)
                            chunks.extend(sub_chunks)

                            current_chunk = []
                            current_length = 0
                    else:
                        current_chunk.append(split)
                        current_length += split_length

                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                return [
                    chunk for chunk in chunks if chunk.strip()
                ]

        return self._split_by_character(text)

    def _split_by_character(self, text: str) -> list[str]:
        #Split text by character count.

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            if self.chunk_overlap > 0:
                start = end - self.chunk_overlap
            else:
                start = end

        return chunks






class SemanticChunker:
    #Semantic chunking based on sentence boundaries.

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        #Split text into semantic chunks.

        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                    if self.chunk_overlap > 0:
                        overlap_sentences = []
                        overlap_length = 0

                        for s in reversed(current_chunk):
                            if (
                                overlap_length + len(s)
                                <= self.chunk_overlap
                            ):
                                overlap_sentences.insert(0, s)
                                overlap_length += len(s)
                            else:
                                break

                        current_chunk = (
                            overlap_sentences + [sentence]
                        )
                        current_length = (
                            overlap_length + sentence_length
                        )
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        #Split text into sentences.

        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_endings.split(text)

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]



class DocumentChunker:
    #Main document chunking service.

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: str = "recursive",
    ):
        settings = get_settings()

        self.chunk_size = (
            chunk_size if chunk_size is not None else settings.chunk_size
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.chunk_overlap
        )
        self.strategy = strategy

        if strategy == "recursive":
            self.splitter = RecursiveCharacterSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif strategy == "semantic":
            self.splitter = SemanticChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}"
            )

    def chunk_document(self,document: Document,doc_id_prefix: str = "",) -> list[TextChunk]:
        #Chunk a single document.

        text_chunks = self.splitter.split_text(
            document.content
        )

        chunks = []

        for i, text in enumerate(text_chunks):
            chunk_id = (
                f"{doc_id_prefix}{document.doc_id}_{i}"
            )

            metadata = {
                **document.metadata,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            }

            chunks.append(
                TextChunk(
                    text=text,
                    metadata=metadata,
                    chunk_id=chunk_id,
                )
            )

        return chunks

    def chunk_documents(
        self,
        documents: list[Document],
    ) -> list[TextChunk]:
        #Chunk multiple documents.

        all_chunks = []

        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)

        return all_chunks


#global chunker instance

_chunker: Optional[DocumentChunker] = None


def get_chunker() -> DocumentChunker:
    """Get global document chunker instance."""

    global _chunker

    if _chunker is None:
        _chunker = DocumentChunker()

    return _chunker