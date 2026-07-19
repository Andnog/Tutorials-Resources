"""Rutas estables, independientes del directorio desde el que se ejecuta el código."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CORPUS_PATH = PROCESSED_DIR / "corpus_pages.json"
CHROMA_DIR = ROOT / "data" / "chroma"
