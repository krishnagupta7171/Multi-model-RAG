from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query prompt")
    top_k: Optional[int] = Field(default=None, ge=1, description="Number of context documents to retrieve")
    use_reflection: bool = Field(default=True, description="Enable agent self-reflection loop")
    stream: bool = Field(default=False, description="Stream LLM generation chunks")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the primary bottlenecks in multi-modal retrieval?",
                "top_k": 3,
                "use_reflection": True,
                "stream": False,
            }
        }
    }


class IngestRequest(BaseModel):
   #  Request schema for ingesting a single document or simple text
    file_path: Optional[str] = Field(default=None, description="Path to a local file")
    text: Optional[str] = Field(default=None, description="Raw text content to ingest")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata key-values")
    collection_name: str = Field(default="documents", description="Target vector store collection name")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Quarterly earnings report highlights...",
                "metadata": {"source": "earnings_call", "quarter": "Q3"},
                "collection_name": "financial_docs",
            }
        }
    }


class BatchIngestRequest(BaseModel):
    # Request schema for batch directory ingestion

    directory: str = Field(..., min_length=1, description="Path to directory containing documents")
    recursive: bool = Field(default=True, description="Recursively search directory")
    collection_name: str = Field(default="documents", description="Target vector store collection name")

    model_config = {
        "json_schema_extra": {
            "example": {
                "directory": "./data/raw_reports",
                "recursive": True,
                "collection_name": "documents",
            }
        }
    }


class DocumentResponse(BaseModel):
    # Payload representation of a retrieved document chunk.

    id: str = Field(..., description="Unique document chunk identifier")
    score: float = Field(..., description="Similarity or reranking score")
    content: str = Field(..., description="Text content of the retrieved chunk")


class QueryResponse(BaseModel):
    # Response schema returned from the query pipeline.

    query: str = Field(..., description="Original user prompt")
    answer: str = Field(..., description="Generated answer from the agent")
    documents: List[DocumentResponse] = Field(default_factory=list, description="Retrieved evidence chunks")
    reflection: Optional[Dict[str, Any]] = Field(default=None, description="Agent evaluation & reflection data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution performance and tracing metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is a field of artificial intelligence...",
                "documents": [
                    {"id": "doc_1", "score": 0.94, "content": "Machine learning focuses on..."}
                ],
                "reflection": {"score": 9.0, "is_sufficient": True},
                "metadata": {"processing_time_sec": 1.15},
            }
        }
    }


class IngestResponse(BaseModel):
    #Response schema summarizing ingestion batch results.

    success: bool = Field(..., description="True if ingestion succeeded without fatal errors")
    num_documents: int = Field(..., ge=0, description="Total documents processed")
    num_chunks: int = Field(..., ge=0, description="Total vector chunks generated")
    document_ids: List[str] = Field(default_factory=list, description="List of generated chunk IDs")
    message: str = Field(..., description="Result message")


class HealthResponse(BaseModel):
    # Service health and component availability status.

    status: str = Field(..., description="Overall status (healthy, degraded, unhealthy)")
    version: str = Field(default="0.1.0", description="API version")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Health status of external dependencies")


class ErrorResponse(BaseModel):
    #Standardized API error envelope.

    error_type: str = Field(..., description="Class or category of error")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Contextual debugging details")