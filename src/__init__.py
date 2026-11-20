"""Core package for the Helwan National University RAG QA & Evaluation pipeline."""

from pathlib import Path

DEFAULT_DATA_DIR = Path("data")
DEFAULT_CHROMA_DIR = Path("chroma_db")
DEFAULT_COLLECTION_NAME = "helwan_regulations"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

__all__ = [
    "DEFAULT_CHROMA_DIR",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_DATA_DIR",
    "DEFAULT_EMBEDDING_MODEL",
]
