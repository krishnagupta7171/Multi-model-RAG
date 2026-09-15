from datetime import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.schemas import Base, DocumentMetadata, IngestionLog, QueryLog


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    Base.metadata.drop_all(engine)


def test_create_document_metadata(db_session):
    doc = DocumentMetadata(
        doc_id="doc_101",
        source="data/reports/annual_2025.pdf",
        type="pdf",
        content_hash="abc123hash",
        doc_metadata={"pages": 12, "author": "Finance Team"},
    )
    db_session.add(doc)
    db_session.commit()

    retrieved = db_session.query(DocumentMetadata).filter_by(doc_id="doc_101").first()
    assert retrieved is not None
    assert retrieved.source == "data/reports/annual_2025.pdf"
    assert retrieved.doc_metadata["pages"] == 12
    assert isinstance(retrieved.created_at, datetime)


def test_create_query_log(db_session):
    log = QueryLog(
        request_id="req_xyz_1",
        query="What was our Q3 EBITDA?",
        answer="The Q3 EBITDA was $4.2M.",
        num_documents=4,
        reflection_score=8.5,
        processing_time=1.42,
        extra_metadata={"model": "gemini"},
    )
    db_session.add(log)
    db_session.commit()

    retrieved = db_session.query(QueryLog).filter_by(request_id="req_xyz_1").first()
    assert retrieved is not None
    assert retrieved.reflection_score == 8.5
    assert retrieved.extra_metadata["model"] == "gemini"


def test_create_ingestion_log(db_session):
    log = IngestionLog(
        request_id="batch_001",
        source="s3://bucket/data",
        num_documents=10,
        num_chunks=45,
        success=True,
    )
    db_session.add(log)
    db_session.commit()

    retrieved = db_session.query(IngestionLog).filter_by(request_id="batch_001").first()
    assert retrieved is not None
    assert retrieved.success is True
    assert retrieved.num_chunks == 45