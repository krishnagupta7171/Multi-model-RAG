import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.deps import (get_request_id,get_retriever_dependency,verify_api_key,)
from src.api.payload import BatchIngestRequest, IngestRequest, IngestResponse
from src.ingestion.chunker import get_chunker
from src.ingestion.loader import Document, get_loader
from src.observability.logging import get_logger
from src.observability.trace import trace_event
from src.retrieval.vector_store import VectorStoreRetriever
from src.utils.exceptions import IngestionError

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(request: IngestRequest,retriever: VectorStoreRetriever = Depends(get_retriever_dependency),
    request_id: str = Depends(get_request_id),api_key: str = Depends(verify_api_key),) -> IngestResponse:
    trace_event("ingest_started", {"request_id": request_id, "mode": "single"})
    logger.info(f"Ingest request [{request_id}]")

    try:
        documents: List[Document] = []

        if request.file_path:
            loader = get_loader(request.file_path)
            documents = await loader.load(request.file_path)
        elif request.text:
            documents = [
                Document(
                    content=request.text,
                    metadata=request.metadata,
                )
            ]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either file_path or text must be provided",
            )

        chunker = get_chunker()
        chunks = chunker.chunk_documents(documents)

        texts = [chunk.text for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        document_ids = await retriever.add_texts(
            texts=texts,
            metadata=metadata,
            ids=ids,
        )

        trace_event("ingest_completed", {"request_id": request_id, "chunks": len(chunks)})

        return IngestResponse(
            success=True,
            num_documents=len(documents),
            num_chunks=len(chunks),
            document_ids=document_ids or ids,
            message=f"Successfully ingested {len(documents)} documents",
        )

    except HTTPException:
        raise
    except IngestionError as e:
        logger.error(f"Ingestion error [{request_id}]: {e}")
        trace_event("ingest_error", {"request_id": request_id, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected ingestion error [{request_id}]: {e}", exc_info=True)
        trace_event("ingest_error", {"request_id": request_id, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.post("/batch", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_directory(request: BatchIngestRequest,retriever: VectorStoreRetriever = Depends(get_retriever_dependency),
    request_id: str = Depends(get_request_id),api_key: str = Depends(verify_api_key),) -> IngestResponse:
    trace_event("ingest_started", {"request_id": request_id, "mode": "batch", "dir": request.directory})
    dir_path = Path(request.directory)

    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directory does not exist: {request.directory}",
        )

    pattern = "**/*" if request.recursive else "*"
    all_files = [p for p in dir_path.glob(pattern) if p.is_file()]

    loaded_documents: List[Document] = []
    for file_path in all_files:
        try:
            loader = get_loader(str(file_path))
            docs = await loader.load(str(file_path))
            loaded_documents.extend(docs)
        except IngestionError:
            continue

    if not loaded_documents:
        return IngestResponse(
            success=True,
            num_documents=0,
            num_chunks=0,
            document_ids=[],
            message="No supported documents found in directory",
        )

    chunker = get_chunker()
    chunks = chunker.chunk_documents(loaded_documents)

    texts = [chunk.text for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]
    ids = [chunk.chunk_id for chunk in chunks]

    document_ids = await retriever.add_texts(texts=texts,metadata=metadata,ids=ids,)

    trace_event("ingest_completed", {"request_id": request_id, "chunks": len(chunks)})

    return IngestResponse(
        success=True,
        num_documents=len(loaded_documents),
        num_chunks=len(chunks),
        document_ids=document_ids or ids,
        message=f"Successfully ingested {len(loaded_documents)} documents from directory",
    )


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...),retriever: VectorStoreRetriever = Depends(get_retriever_dependency),
    request_id: str = Depends(get_request_id),api_key: str = Depends(verify_api_key),) -> IngestResponse:
    trace_event("ingest_started", {"request_id": request_id, "mode": "upload", "filename": file.filename})

    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loader = get_loader(tmp_path)
        documents = await loader.load(tmp_path)

        chunker = get_chunker()
        chunks = chunker.chunk_documents(documents)

        texts = [chunk.text for chunk in chunks]
        metadata = [{**chunk.metadata, "filename": file.filename} for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        document_ids = await retriever.add_texts(
            texts=texts,
            metadata=metadata,
            ids=ids,
        )

        trace_event("ingest_completed", {"request_id": request_id, "chunks": len(chunks)})

        return IngestResponse(
            success=True,
            num_documents=len(documents),
            num_chunks=len(chunks),
            document_ids=document_ids or ids,
            message=f"Successfully uploaded and ingested {file.filename}",
        )

    except IngestionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)