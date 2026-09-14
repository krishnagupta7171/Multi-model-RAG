import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ..observability.logging import get_logger
from ..utils.exceptions import IngestionError

logger = get_logger(__name__)


class ImageProcessor:
    # Extract text via OCR and extract visual metadata from images.
    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd
        self._pytesseract = None

    def _get_pytesseract(self):
    
        if self._pytesseract is None:
            try:
                import pytesseract
                if self.tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                self._pytesseract = pytesseract
            except ImportError as e:
                logger.error("pytesseract is not installed in the current environment")
                raise IngestionError(
                    "pytesseract is required for image OCR but is not installed",
                    original_error=e,
                ) from e
        return self._pytesseract

    def extract_text(self, image_path: str) -> str:
        # this will extract text from img using OCR
        path = Path(image_path)
        if not path.exists():
            raise IngestionError(f"Image file not found at: {image_path}")

        try:
            logger.debug(f"Extracting text from image: {image_path}")
            ocr = self._get_pytesseract()
            with Image.open(path) as img:
                text = ocr.image_to_string(img)
            logger.debug(f"Extracted {len(text)} characters from {image_path}")
            return text.strip()
        except IngestionError:
            raise
        except Exception as e:
            logger.error(f"OCR failed for image {image_path}: {e}")
            raise IngestionError(
                f"Failed to extract text from image {image_path}",
                original_error=e,
            ) from e

    def extract_text_from_pil(self, image: Image.Image) -> str:

        try:
            ocr = self._get_pytesseract()
            text = ocr.image_to_string(image)
            return text.strip()
        except IngestionError:
            raise
        except Exception as e:
            logger.error(f"OCR failed for PIL image: {e}")
            raise IngestionError(
                "Failed to extract text from PIL image",
                original_error=e,
            ) from e

    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        # extract the dimension,format from the img.
        path = Path(image_path)
        if not path.exists():
            return {}

        try:
            with Image.open(path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format or "UNKNOWN",
                    "mode": img.mode,
                }
        except Exception as e:
            logger.error(f"Failed to read image metadata for {image_path}: {e}")
            return {}

    def encode_image_base64(self, image_path: str) -> str:
        #encode img to a base64 utf-8 tring
        path = Path(image_path)
        if not path.exists():
            raise IngestionError(f"Image file not found: {image_path}")

        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise IngestionError(
                f"Failed to encode image {image_path}",
                original_error=e,
            ) from e


class TableExtractor:


    def extract_tables_from_pdf(self,pdf_path: str,strategy: str = "auto",) -> List[Dict[str, Any]]:
        
        path = Path(pdf_path)
        if not path.exists():
            raise IngestionError(f"PDF file not found at: {pdf_path}")

        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as e:
            logger.error("unstructured library is not installed")
            raise IngestionError(
                "unstructured[pdf] is required for PDF table extraction",
                original_error=e,
            ) from e

        try:
            logger.info(f"Extracting tables from PDF: {pdf_path}")
            elements = partition_pdf(filename=str(path),strategy=strategy,infer_table_structure=True,)

            tables: List[Dict[str, Any]] = []
            for element in elements:
                category = getattr(getattr(element, "metadata", None), "category", None)
                if category == "Table":
                    tables.append(
                        {
                            "text": str(element),
                            "page": getattr(element.metadata, "page_number", None),
                            "html": getattr(element.metadata, "text_as_html", None),
                        }
                    )

            logger.info(f"Extracted {len(tables)} tables from PDF: {pdf_path}")
            return tables

        except Exception as e:
            logger.error(f"Table extraction failed for {pdf_path}: {e}")
            raise IngestionError(
                f"Failed to extract tables from {pdf_path}",
                original_error=e,
            ) from e

    def table_to_text(self, table_html: str) -> str:
        if not table_html or not table_html.strip():
            return ""

        text = table_html.replace("<tr>", "\n").replace("</tr>", "")
        text = text.replace("<td>", " | ").replace("</td>", "")
        text = text.replace("<th>", " | ").replace("</th>", "")
        return text.strip()


class MultiModalProcessor:
    
    def __init__(self,image_processor: Optional[ImageProcessor] = None,table_extractor: Optional[TableExtractor] = None,):
        self.image_processor = image_processor or ImageProcessor()
        self.table_extractor = table_extractor or TableExtractor()

    def process_pdf_with_images(self,pdf_path: str,extract_images: bool = True,extract_tables: bool = True,) -> Dict[str, Any]:
        
        path = Path(pdf_path)
        if not path.exists():
            raise IngestionError(f"PDF file not found: {pdf_path}")

        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as e:
            raise IngestionError(
                "unstructured[pdf] is required for multi-modal PDF parsing",
                original_error=e,
            ) from e

        try:
            logger.info(f"Processing multi-modal PDF: {pdf_path}")
            elements = partition_pdf(
                filename=str(path),
                strategy="hi_res" if extract_images else "fast",
                extract_images_in_pdf=extract_images,
                infer_table_structure=extract_tables,
            )

            result: Dict[str, Any] = {
                "text": "",
                "images": [],
                "tables": [],
                "metadata": {"source": str(path)},
            }

            text_parts = []
            for element in elements:
                cat = getattr(getattr(element, "metadata", None), "category", "text")

                if cat == "Table" and extract_tables:
                    result["tables"].append(
                        {
                            "text": str(element),
                            "page": getattr(element.metadata, "page_number", None),
                        }
                    )
                else:
                    text_parts.append(str(element))

            result["text"] = "\n\n".join(text_parts)
            logger.info(
                f"Processed PDF: {len(result['text'])} chars, {len(result['tables'])} tables"
            )
            return result

        except Exception as e:
            logger.error(f"Multi-modal PDF processing failed: {e}")
            raise IngestionError(
                f"Failed to process multi-modal PDF {pdf_path}",
                original_error=e,
            ) from e

    def process_image_document(self, image_path: str) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            raise IngestionError(f"Image document not found: {image_path}")

        try:
            logger.info(f"Processing image document: {image_path}")
            text = self.image_processor.extract_text(image_path)
            metadata = self.image_processor.get_image_metadata(image_path)
            metadata["source"] = str(path)
            metadata["type"] = "image"

            return {
                "text": text,
                "metadata": metadata,
            }
        except Exception as e:
            logger.error(f"Image document processing failed: {e}")
            raise IngestionError(
                f"Failed to process image document {image_path}",
                original_error=e,
            ) from e


_multi_modal_processor: Optional[MultiModalProcessor] = None


def get_multi_modal_processor() -> MultiModalProcessor:

    global _multi_modal_processor
    if _multi_modal_processor is None:
        _multi_modal_processor = MultiModalProcessor()
    return _multi_modal_processor