from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DocumentMetadata(Base):

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    doc_id = Column(String(255), unique=True, index=True, nullable=False)
    source = Column(String(500), nullable=False)
    type = Column(String(50), nullable=False)  # pdf, image, table, txt
    content_hash = Column(String(64), index=True, nullable=True)
    # Map DB column 'metadata' to python field 'doc_metadata' to avoid shadowing Base.metadata
    doc_metadata = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class QueryLog(Base):

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String(255), index=True, nullable=False)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    num_documents = Column(Integer, default=0)
    reflection_score = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)  # in seconds
    extra_metadata = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class IngestionLog(Base):

    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String(255), index=True, nullable=False)
    source = Column(String(500), nullable=False)
    num_documents = Column(Integer, default=0)
    num_chunks = Column(Integer, default=0)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    extra_metadata = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)