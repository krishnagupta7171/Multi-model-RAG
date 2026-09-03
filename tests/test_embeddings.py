from unittest.mock import MagicMock, patch

import numpy as np

from src.retrieval.embeddings import EmbeddingService


@patch("src.retrieval.embeddings.SentenceTransformer")
def test_embed_text(mock_model):
    model = MagicMock()
    model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_model.return_value = model

    service = EmbeddingService()

    result = service.embed_text("hello world")

    assert result == [0.1, 0.2, 0.3]
    model.encode.assert_called_once()


@patch("src.retrieval.embeddings.SentenceTransformer")
def test_embed_batch(mock_model):
    model = MagicMock()
    model.encode.return_value = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ])
    mock_model.return_value = model

    service = EmbeddingService()

    result = service.embed_batch(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]


@patch("src.retrieval.embeddings.SentenceTransformer")
def test_similarity(mock_model):
    mock_model.return_value = MagicMock()

    service = EmbeddingService()

    result = service.similarity(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )

    assert result == 1.0