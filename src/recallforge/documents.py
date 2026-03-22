"""Local-first document extraction helpers for RecallForge.

Supports practical office-document ingestion for:
- PDF (via pypdf when installed, plus a lightweight fallback parser)
- DOCX (direct OOXML parsing)
- PPTX (direct OOXML parsing)
"""

from __future__ import annotations

import importlib.util
import io
import re
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from xml.etree import ElementTree as ET
from zipfile import ZipFile


DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}

_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_A_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


@dataclass
class DocumentSection:
    logical_path: str
    title: str
    text: str
    section_type: str
    index: int
    content_type: str = "text"
    image_path: Optional[str] = None


@dataclass
class DocumentArtifacts:
    sections: List[DocumentSection]
    document_type: str
    extractor: str


def is_document_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in DOCUMENT_EXTENSIONS


def extract_document_artifacts(document_path: str | Path, logical_path: str) -> DocumentArtifacts:
    path = Path(document_path).expanduser().resolve()
    suffix = path.suffix.lower()

    if suffix == ".docx":
        return _extract_docx(path, logical_path)
    if suffix == ".pptx":
        return _extract_pptx(path, logical_path)
    if suffix == ".pdf":
        return _extract_pdf(path, logical_path)

    raise ValueError(f"Unsupported document type: {path.suffix}")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _group_sections(
    texts: Iterable[str],
    logical_path: str,
    section_type: str,
    title_prefix: str,
    max_chars: int = 1800,
) -> List[DocumentSection]:
    sections: List[DocumentSection] = []
    buffer: List[str] = []
    buffer_chars = 0
    logical_index = 0

    def flush() -> None:
        nonlocal buffer, buffer_chars, logical_index
        if not buffer:
            return
        logical_index += 1
        text = _clean_text("\n\n".join(buffer))
        if not text:
            buffer = []
            buffer_chars = 0
            return
        sections.append(
            DocumentSection(
                logical_path=f"{logical_path}::{section_type}:{logical_index:04d}",
                title=f"{title_prefix} {logical_index}",
                text=text,
                section_type=section_type,
                index=logical_index,
                content_type="text",
            )
        )
        buffer = []
        buffer_chars = 0

    for raw in texts:
        cleaned = _clean_text(raw)
        if not cleaned:
            continue
        if buffer and buffer_chars + len(cleaned) > max_chars:
            flush()
        buffer.append(cleaned)
        buffer_chars += len(cleaned)

    flush()
    return sections


def _extract_docx(path: Path, logical_path: str) -> DocumentArtifacts:
    import logging

    logger = logging.getLogger("recallforge.documents")

    paragraphs: List[str] = []
    with ZipFile(path) as archive:
        try:
            xml_bytes = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError(f"DOCX missing word/document.xml: {path}") from exc

    root = ET.fromstring(xml_bytes)
    for paragraph in root.findall(".//w:body/w:p", _W_NS):
        runs = [node.text for node in paragraph.findall(".//w:t", _W_NS) if node.text]
        text = _clean_text(" ".join(runs))
        if text:
            paragraphs.append(text)

    sections = _group_sections(
        paragraphs,
        logical_path=logical_path,
        section_type="section",
        title_prefix=f"{path.stem} section",
        max_chars=1600,
    )
    if not sections:
        logger.warning("No extractable text found in DOCX: %s", path)
        return DocumentArtifacts(sections=[], document_type="docx", extractor="ooxml")
    return DocumentArtifacts(sections=sections, document_type="docx", extractor="ooxml")


def _extract_pptx(path: Path, logical_path: str) -> DocumentArtifacts:
    import logging

    logger = logging.getLogger("recallforge.documents")

    slides: List[str] = []
    with ZipFile(path) as archive:
        slide_names = sorted(
            (
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ),
            key=_natural_slide_key,
        )
        if not slide_names:
            logger.warning("PPTX contains no slides: %s", path)
            return DocumentArtifacts(sections=[], document_type="pptx", extractor="ooxml")

        for slide_name in slide_names:
            xml_bytes = archive.read(slide_name)
            root = ET.fromstring(xml_bytes)
            texts = [node.text for node in root.findall(".//a:t", _A_NS) if node.text]
            slide_text = _clean_text(" ".join(texts))
            if slide_text:
                slides.append(slide_text)

    sections = [
        DocumentSection(
            logical_path=f"{logical_path}::slide:{index:04d}",
            title=f"{path.stem} slide {index}",
            text=text,
            section_type="slide",
            index=index,
            content_type="text",
        )
        for index, text in enumerate(slides, start=1)
        if text
    ]
    if not sections:
        logger.warning("No extractable text found in PPTX: %s", path)
        return DocumentArtifacts(sections=[], document_type="pptx", extractor="ooxml")
    return DocumentArtifacts(sections=sections, document_type="pptx", extractor="ooxml")


def _natural_slide_key(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def _extract_pdf(path: Path, logical_path: str) -> DocumentArtifacts:
    if importlib.util.find_spec("pypdf") is not None:
        return _extract_pdf_with_pypdf(path, logical_path)
    return _extract_pdf_fallback(path, logical_path)


def _render_pdf_page_as_image(
    pdf_path: Path, page_number: int, temp_dir: Optional[Path] = None
) -> Optional[str]:
    """Render a PDF page as an image. Returns the image path or None if failed.

    Fallback chain:
    1. Try pypdf to extract embedded images from the page
    2. Try pymupdf (fitz) to render the page
    3. Return None if all methods fail
    """
    import logging

    logger = logging.getLogger("recallforge.documents")

    # Try pymupdf first (best quality page rendering)
    if importlib.util.find_spec("fitz") is not None:
        try:
            import fitz  # type: ignore

            # Create temp dir if not provided
            if temp_dir is None:
                temp_dir = Path(tempfile.mkdtemp(prefix="recallforge_pdf_"))

            doc = fitz.open(str(pdf_path))
            page = doc.load_page(page_number - 1)  # 0-indexed
            pix = page.get_pixmap(dpi=150)
            image_path = temp_dir / f"page_{page_number:04d}.png"
            pix.save(str(image_path))
            doc.close()
            return str(image_path)
        except Exception as e:
            logger.debug("pymupdf page rendering failed for %s page %d: %s", pdf_path, page_number, e)

    # Try pypdf to extract embedded images
    if importlib.util.find_spec("pypdf") is not None:
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(pdf_path))
            if page_number <= len(reader.pages):
                page = reader.pages[page_number - 1]
                if hasattr(page, "images") and page.images:
                    # Create temp dir if not provided
                    if temp_dir is None:
                        temp_dir = Path(tempfile.mkdtemp(prefix="recallforge_pdf_"))

                    # Extract the first image from the page
                    for img_index, image in enumerate(page.images):
                        try:
                            image_data = image.data
                            image_path = temp_dir / f"page_{page_number:04d}_img_{img_index:02d}.png"
                            with open(image_path, "wb") as f:
                                f.write(image_data)
                            return str(image_path)
                        except Exception as img_e:
                            logger.debug("Failed to extract image %d from page %d: %s", img_index, page_number, img_e)
                            continue
        except Exception as e:
            logger.debug("pypdf image extraction failed for %s page %d: %s", pdf_path, page_number, e)

    return None


def _extract_pdf_with_pypdf(path: Path, logical_path: str) -> DocumentArtifacts:
    import logging
    from pypdf import PdfReader  # type: ignore

    logger = logging.getLogger("recallforge.documents")

    reader = PdfReader(str(path))
    sections: List[DocumentSection] = []

    # Create temp directory for page images if needed
    temp_dir: Optional[Path] = None

    for index, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            sections.append(
                DocumentSection(
                    logical_path=f"{logical_path}::page:{index:04d}",
                    title=f"{path.stem} page {index}",
                    text=text,
                    section_type="page",
                    index=index,
                    content_type="text",
                )
            )
            continue

        # No text extracted - try to render page as image
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="recallforge_pdf_"))
        image_path = _render_pdf_page_as_image(path, index, temp_dir)
        if image_path:
            sections.append(
                DocumentSection(
                    logical_path=f"{logical_path}::page:{index:04d}",
                    title=f"{path.stem} page {index}",
                    text="",  # No text, image will be embedded
                    section_type="page",
                    index=index,
                    content_type="image",
                    image_path=image_path,
                )
            )

    if not sections:
        logger.warning("No extractable text or images found in PDF: %s", path)
        return DocumentArtifacts(sections=[], document_type="pdf", extractor="pypdf")
    return DocumentArtifacts(sections=sections, document_type="pdf", extractor="pypdf")


def _extract_pdf_fallback(path: Path, logical_path: str) -> DocumentArtifacts:
    import logging

    logger = logging.getLogger("recallforge.documents")

    raw = path.read_bytes()
    texts: List[str] = []
    for stream_bytes in _iter_pdf_streams(raw):
        strings = _extract_pdf_literal_strings(stream_bytes)
        if strings:
            texts.append(_clean_text(" ".join(strings)))

    merged = _clean_text("\n\n".join(texts))
    if merged:
        return DocumentArtifacts(
            sections=[
                DocumentSection(
                    logical_path=f"{logical_path}::page:0001",
                    title=f"{path.stem} page 1",
                    text=merged,
                    section_type="page",
                    index=1,
                    content_type="text",
                )
            ],
            document_type="pdf",
            extractor="builtin-pdf-fallback",
        )

    # No text extracted - try to render first page as image using pymupdf
    image_path = _render_pdf_page_as_image(path, 1, None)
    if image_path:
        return DocumentArtifacts(
            sections=[
                DocumentSection(
                    logical_path=f"{logical_path}::page:0001",
                    title=f"{path.stem} page 1",
                    text="",
                    section_type="page",
                    index=1,
                    content_type="image",
                    image_path=image_path,
                )
            ],
            document_type="pdf",
            extractor="builtin-pdf-fallback",
        )

    logger.warning(
        "No extractable text or images found in PDF: %s. Install recallforge[docs] for richer PDF parsing.",
        path,
    )
    return DocumentArtifacts(sections=[], document_type="pdf", extractor="builtin-pdf-fallback")


def _iter_pdf_streams(raw: bytes) -> Iterable[bytes]:
    pattern = re.compile(rb"<<(?P<dict>.*?)>>\s*stream\r?\n(?P<data>.*?)\r?\nendstream", re.S)
    for match in pattern.finditer(raw):
        dict_bytes = match.group("dict")
        data = match.group("data")
        if b"/FlateDecode" in dict_bytes:
            try:
                data = zlib.decompress(data)
            except Exception:
                continue
        yield data


def _extract_pdf_literal_strings(stream_bytes: bytes) -> List[str]:
    results: List[str] = []
    i = 0
    while i < len(stream_bytes):
        if stream_bytes[i] != 0x28:  # "("
            i += 1
            continue

        i += 1
        depth = 1
        buffer = io.StringIO()
        while i < len(stream_bytes) and depth > 0:
            byte = stream_bytes[i]
            i += 1

            if byte == 0x5C:  # backslash
                if i >= len(stream_bytes):
                    break
                escaped = stream_bytes[i]
                i += 1
                mapping = {
                    0x6E: "\n",  # n
                    0x72: "\r",  # r
                    0x74: "\t",  # t
                    0x62: "\b",  # b
                    0x66: "\f",  # f
                    0x28: "(",
                    0x29: ")",
                    0x5C: "\\",
                }
                if escaped in mapping:
                    buffer.write(mapping[escaped])
                elif 48 <= escaped <= 55:
                    octal_digits = bytes([escaped])
                    while i < len(stream_bytes) and len(octal_digits) < 3 and 48 <= stream_bytes[i] <= 55:
                        octal_digits += bytes([stream_bytes[i]])
                        i += 1
                    try:
                        buffer.write(chr(int(octal_digits, 8)))
                    except Exception:
                        pass
                else:
                    buffer.write(chr(escaped))
                continue

            if byte == 0x28:  # "("
                depth += 1
                buffer.write("(")
                continue
            if byte == 0x29:  # ")"
                depth -= 1
                if depth == 0:
                    break
                buffer.write(")")
                continue

            if 32 <= byte <= 126 or byte in (9, 10, 13):
                buffer.write(chr(byte))

        text = _clean_text(buffer.getvalue())
        if text:
            results.append(text)

    return results
