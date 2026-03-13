from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from recallforge.documents import extract_document_artifacts, is_document_file


def _write_fake_docx(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>RecallForge document ingestion works locally.</w:t></w:r></w:p>
                <w:p><w:r><w:t>Structured paragraphs become searchable sections.</w:t></w:r></w:p>
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
              <p:cSld>
                <p:spTree>
                  <p:sp>
                    <p:txBody>
                      <a:p><a:r><a:t>Slide one architecture overview</a:t></a:r></a:p>
                    </p:txBody>
                  </p:sp>
                </p:spTree>
              </p:cSld>
            </p:sld>
            """,
        )
        archive.writestr(
            "ppt/slides/slide2.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                   xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
              <p:cSld>
                <p:spTree>
                  <p:sp>
                    <p:txBody>
                      <a:p><a:r><a:t>Slide two deployment checklist</a:t></a:r></a:p>
                    </p:txBody>
                  </p:sp>
                </p:spTree>
              </p:cSld>
            </p:sld>
            """,
        )


def _write_fake_pdf(path: Path) -> None:
    content = b"BT /F1 18 Tf 72 720 Td (Quarterly revenue overview for local-first MCP memory.) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
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


def test_is_document_file():
    assert is_document_file("report.pdf")
    assert is_document_file("slides.pptx")
    assert is_document_file("notes.docx")
    assert not is_document_file("notes.md")


def test_extract_docx_artifacts(tmp_path):
    path = tmp_path / "notes.docx"
    _write_fake_docx(path)

    artifacts = extract_document_artifacts(path, "docs/notes.docx")

    assert artifacts.document_type == "docx"
    assert artifacts.extractor == "ooxml"
    assert len(artifacts.sections) == 1
    assert "RecallForge document ingestion works locally." in artifacts.sections[0].text
    assert artifacts.sections[0].logical_path == "docs/notes.docx::section:0001"


def test_extract_pptx_artifacts(tmp_path):
    path = tmp_path / "slides.pptx"
    _write_fake_pptx(path)

    artifacts = extract_document_artifacts(path, "slides/slides.pptx")

    assert artifacts.document_type == "pptx"
    assert artifacts.extractor == "ooxml"
    assert len(artifacts.sections) == 2
    assert artifacts.sections[0].logical_path == "slides/slides.pptx::slide:0001"
    assert "deployment checklist" in artifacts.sections[1].text.lower()


def test_extract_pdf_artifacts_with_builtin_fallback(tmp_path):
    path = tmp_path / "report.pdf"
    _write_fake_pdf(path)

    artifacts = extract_document_artifacts(path, "reports/report.pdf")

    assert artifacts.document_type == "pdf"
    assert artifacts.sections
    assert artifacts.sections[0].logical_path == "reports/report.pdf::page:0001"
    assert "local-first MCP memory" in artifacts.sections[0].text
