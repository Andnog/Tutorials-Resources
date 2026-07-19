from pathlib import Path

import fitz

from laundry_rag.pdf_benchmark import benchmark_pdf, extract_with_pymupdf


def test_pymupdf_benchmark_records_coverage(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Texto de prueba para el extractor.")
    document.save(pdf)

    result = benchmark_pdf(pdf, "PyMuPDF", extract_with_pymupdf)

    assert result["páginas"] == 1
    assert result["páginas_con_texto"] == 1
    assert result["caracteres"] > 0
