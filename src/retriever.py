"""Vector retrieval and shared multilingual-e5 embedding helpers."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings

from src import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievedChunk:
    """Single retrieved chunk with similarity metrics and source metadata."""

    chunk_id: str
    text: str
    source_file: str
    page_number: int
    distance: float
    similarity: float
    metadata: dict[str, Any]


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Lazily load and cache the SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    *,
    prefix: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> list[list[float]]:
    """Encode texts using normalized embeddings with an E5 task prefix."""
    if not texts:
        return []

    model = get_embedding_model(model_name)
    prefixed_texts = [f"{prefix}{text}" for text in texts]
    vectors = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [[float(value) for value in row] for row in vectors]


def embed_passages(
    texts: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> list[list[float]]:
    """Encode document passages with the E5 `passage: ` prefix."""
    return embed_texts(
        texts,
        prefix="passage: ",
        model_name=model_name,
        batch_size=batch_size,
    )


def embed_query(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float]:
    """Encode a single search query with the E5 `query: ` prefix."""
    return embed_texts([query], prefix="query: ", model_name=model_name, batch_size=1)[0]


class ChromaRetriever:
    """Query interface over a persisted or in-memory ChromaDB collection."""

    def __init__(
        self,
        chroma_dir: Path | str = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embed_fn: Callable[[str], list[float]] | None = None,
        client: Any | None = None,
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.model_name = model_name
        self._embed_fn = embed_fn or (lambda q: embed_query(q, model_name=self.model_name))

        if client is not None:
            self._client = client
        else:
            if not self.chroma_dir.exists():
                raise FileNotFoundError(
                    f"ChromaDB directory '{self.chroma_dir}' does not exist. "
                    "Run `python -m src.ingestion` first to index the PDFs."
                )
            self._client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )

        try:
            self._collection = self._client.get_collection(name=self.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{self.collection_name}' was not found in "
                f"'{self.chroma_dir}'. Run `python -m src.ingestion` first."
            ) from exc

    def count(self) -> int:
        """Return total number of indexed chunks in the collection."""
        return int(self._collection.count())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks with cosine distance and similarity."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        total_chunks = self.count()
        if total_chunks == 0:
            return []

        n_results = min(top_k, total_chunks)
        query_embedding = self._embed_fn(query)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        for raw_id, doc_text, raw_meta, raw_distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            metadata = dict(raw_meta) if raw_meta else {}
            distance = float(raw_distance)
            similarity = max(-1.0, min(1.0, 1.0 - distance))
            results.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id", raw_id)),
                    text=str(doc_text or ""),
                    source_file=str(metadata.get("source_file", "unknown")),
                    page_number=int(metadata.get("page_number", 0)),
                    distance=distance,
                    similarity=similarity,
                    metadata=metadata,
                )
            )

        results.sort(key=lambda item: item.distance)
        return results


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    chroma_dir: Path | str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[RetrievedChunk]:
    """Convenience helper to query ChromaDB directly in one call."""
    retriever = ChromaRetriever(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        model_name=model_name,
    )
    return retriever.retrieve(query=query, top_k=top_k)


def _configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for quick retrieval inspection."""
    _configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Query Helwan regulations ChromaDB index.")
    parser.add_argument("query", help="Arabic or English search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Collection name.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    args = parser.parse_args(argv)

    chunks = retrieve_chunks(
        query=args.query,
        top_k=args.top_k,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        model_name=args.model,
    )
    print(f"Retrieved {len(chunks)} chunks for query: {args.query!r}")
    for rank, chunk in enumerate(chunks, start=1):
        preview = chunk.text.replace("\n", " ")[:180]
        print(
            f"[{rank}] sim={chunk.similarity:.4f} dist={chunk.distance:.4f} "
            f"| {chunk.source_file} p.{chunk.page_number} ({chunk.chunk_id})"
        )
        print(f"    {preview}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
