#!/usr/bin/env python3
"""Generate deterministic office-document fixtures for UAT."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile


def _write_fake_docx(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>Agent memory document about quarterly planning.</w:t></w:r></w:p>
                <w:p><w:r><w:t>Architecture notes mention embeddings, reranking, and local-first MCP deployment.</w:t></w:r></w:p>
              </w:body>
            </w:document>
            """,
        )


def _write_fake_pptx(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                   xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
              <p:cSld><p:spTree><p:sp><p:txBody>
                <a:p><a:r><a:t>Slide one roadmap overview</a:t></a:r></a:p>
              </p:txBody></p:sp></p:spTree></p:cSld>
            </p:sld>
            """,
        )
        archive.writestr(
            "ppt/slides/slide2.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                   xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
              <p:cSld><p:spTree><p:sp><p:txBody>
                <a:p><a:r><a:t>Slide two MCP deployment checklist</a:t></a:r></a:p>
              </p:txBody></p:sp></p:spTree></p:cSld>
            </p:sld>
            """,
        )


def _write_fake_pdf(path: Path) -> None:
    """Generate a valid PDF that pypdf can extract text from.
    
    Uses pypdf's PdfWriter to create a properly-formed PDF with embedded text.
    Falls back to a manual PDF with proper font resources if pypdf write fails.
    """
    try:
        from pypdf import PdfWriter
        from pypdf.generic import (
            ArrayObject,
            DictionaryObject,
            NameObject,
            NumberObject,
            TextStringObject,
            ContentStream,
        )
    except ImportError:
        pass

    # Build a minimal but valid PDF manually with proper font resources
    text_line = "Local-first PDF notes for RecallForge MCP ingestion."
    content = f"BT /F1 18 Tf 72 720 Td ({text_line}) Tj ET".encode("ascii")

    # Helvetica is a built-in PDF font — no embedding required
    font_dict = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    resources = b"<< /Font << /F1 5 0 R >> >>"

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources %s /Contents 4 0 R >>" % resources,
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        font_dict,
    ]

    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii"))
        chunks.append(obj)
        chunks.append(b"\nendobj\n")

    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        chunks.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    chunks.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(b"".join(chunks))


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: generate_test_documents.py <output-dir>", file=sys.stderr)
        return 2

    output_dir = Path(sys.argv[1]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    docx_path = output_dir / "planning_notes.docx"
    pptx_path = output_dir / "deployment_review.pptx"
    pdf_path = output_dir / "mcp_overview.pdf"

    _write_fake_docx(docx_path)
    _write_fake_pptx(pptx_path)
    _write_fake_pdf(pdf_path)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files": {
                    "docx": str(docx_path),
                    "pptx": str(pptx_path),
                    "pdf": str(pdf_path),
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
