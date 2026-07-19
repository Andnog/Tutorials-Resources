"""Construye de forma reproducible el corpus cacheado y la colección persistente."""

from laundry_rag.ingestion import chunk_pages, ingest_manuals
from laundry_rag.paths import CHROMA_DIR, CORPUS_PATH
from laundry_rag.vectorstore import ChromaManualStore


def main() -> None:
    pages = ingest_manuals(force=True)
    chunks = chunk_pages(pages)
    store = ChromaManualStore()
    store.rebuild(chunks)
    print(f"Corpus: {len(pages)} páginas en {CORPUS_PATH}")
    print(f"ChromaDB: {store.count} chunks en {CHROMA_DIR}")


if __name__ == "__main__":
    main()
