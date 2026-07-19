from pathlib import Path

import fitz

from laundry_rag.ingestion import Page, chunk_pages, extract_pdf


def test_extract_pdf_keeps_page_metadata(tmp_path: Path) -> None:
    pdf = tmp_path / "manual.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72), "SEGURIDAD DE LA LAVADORA\nLea las instrucciones antes de usar el equipo."
    )
    document.save(pdf)
    pages = extract_pdf(pdf)
    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].source == "manual.pdf"
    assert pages[0].text.startswith("SEGURIDAD")


def test_chunking_has_text_and_traceable_metadata() -> None:
    page = Page("p1", "manual.pdf", "Manual", "X1", 3, "Mantenimiento", "palabra " * 600, "pymupdf")
    chunks = chunk_pages([page], chunk_words=100, overlap_words=20)
    assert len(chunks) > 1
    assert all(chunk.text and chunk.page == 3 and chunk.source == "manual.pdf" for chunk in chunks)
