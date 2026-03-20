from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.core.settings import Settings


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_simple_pdf(path: Path, lines: list[str]) -> None:
    """Write a tiny text PDF suitable for parser tests."""

    content_lines = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        escaped = _escape_pdf_text(line)
        if index > 0:
            content_lines.append("0 -18 Td")
        content_lines.append(f"({escaped}) Tj")
    content_lines.append("ET")
    content = "\n".join(content_lines)
    content_bytes = content.encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content_bytes), content_bytes),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
    pdf.extend(f"startxref\n{xref_start}\n%%EOF\n".encode("ascii"))
    path.write_bytes(bytes(pdf))


@pytest.fixture()
def fixture_root(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parent / "fixtures" / "competencies"
    target_root = tmp_path / "competencies"
    shutil.copytree(source_root, target_root)

    manuals_root = target_root / "PT" / "PT_2.6_sand_control" / "manuals"
    write_simple_pdf(
        manuals_root / "01_requirement_for_sand_control.pdf",
        [
            "Requirement for Sand Control",
            "Sand management starts from sanding risk and operating philosophy.",
            "Appraisal inputs include well type, field context, and production philosophy.",
        ],
    )
    write_simple_pdf(
        manuals_root / "02_sand_control_design.pdf",
        [
            "Sand Control Design",
            "Method selection should compare screen, gravel pack, and chemical consolidation.",
            "Design basis must include production rate, completion constraints, and reservoir strength.",
        ],
    )
    return target_root


@pytest.fixture()
def test_settings(tmp_path: Path, fixture_root: Path) -> Settings:
    return Settings(
        COMPETENCIES_ROOT=fixture_root,
        VECTOR_STORE_PATH=tmp_path / "index" / "vector_store.sqlite3",
        EMBEDDINGS_CACHE_PATH=tmp_path / "index" / "embeddings_cache.sqlite3",
        TOP_K_DOCUMENTS=2,
        TOP_K_CHUNKS=4,
        HEURISTIC_SHORTLIST_SIZE=2,
        LLM_ROUTING_ENABLED=False,
        EMBEDDING_PROVIDER="hash",
        EMBEDDING_DIMENSION=64,
    )
