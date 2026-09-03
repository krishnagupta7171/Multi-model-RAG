

from typing import Any, Optional


class RAGException(Exception):

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(RAGException):
    #Raised when there's a configuration issue
    pass


class IngestionError(RAGException):
    #Raised when document ingestion fails
    pass


class RetrievalError(RAGException):
    #Raised when retrieval operations fail
    pass


class GenerationError(RAGException):
    #Raised when LLM generation fails
    pass