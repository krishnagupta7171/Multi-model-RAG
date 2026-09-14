import base64
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from src.ingestion.multi_modal import (
    ImageProcessor,
    MultiModalProcessor,
    TableExtractor,
    get_multi_modal_processor,
)
from src.utils.exceptions import IngestionError


@pytest.fixture
def temp_sample_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "sample.png"
    
    img = Image.new("RGB", (100, 50), color="blue")
    img.save(img_path)
    return img_path


def test_image_metadata_extraction(temp_sample_image: Path):
    processor = ImageProcessor()
    meta = processor.get_image_metadata(str(temp_sample_image))

    assert meta["width"] == 100
    assert meta["height"] == 50
    assert meta["format"] == "PNG"
    assert meta["mode"] == "RGB"


def test_encode_image_base64(temp_sample_image: Path):
    processor = ImageProcessor()
    b64_str = processor.encode_image_base64(str(temp_sample_image))

    assert isinstance(b64_str, str)
    decoded_bytes = base64.b64decode(b64_str)
    assert len(decoded_bytes) > 0


def test_encode_image_file_not_found():
    processor = ImageProcessor()
    with pytest.raises(IngestionError, match="Image file not found"):
        processor.encode_image_base64("non_existent_image.png")


def test_ocr_extract_text_mocked(temp_sample_image: Path):
    processor = ImageProcessor()
    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.return_value = "Extracted OCR Header\nFinancial Metrics"

    with patch.object(processor, "_get_pytesseract", return_value=mock_pytesseract):
        text = processor.extract_text(str(temp_sample_image))
        assert "Extracted OCR Header" in text
        assert "Financial Metrics" in text


def test_table_to_text_conversion():
    extractor = TableExtractor()
    html_table = "<tr><th>Metric</th><th>Value</th></tr><tr><td>Revenue</td><td>$5M</td></tr>"
    converted = extractor.table_to_text(html_table)

    assert "Metric" in converted
    assert "Value" in converted
    assert "Revenue" in converted
    assert "$5M" in converted


def test_process_image_document_end_to_end(temp_sample_image: Path):
    processor = MultiModalProcessor()
    mock_ocr = MagicMock()
    mock_ocr.image_to_string.return_value = "Quarterly Architecture Diagram"

    with patch.object(processor.image_processor, "_get_pytesseract", return_value=mock_ocr):
        result = processor.process_image_document(str(temp_sample_image))
        assert result["text"] == "Quarterly Architecture Diagram"
        assert result["metadata"]["width"] == 100
        assert result["metadata"]["type"] == "image"


def test_get_multi_modal_processor_singleton():
    p1 = get_multi_modal_processor()
    p2 = get_multi_modal_processor()
    assert p1 is p2