"""PDF ingestion, Arabic text quality checks, chunking, and ChromaDB persistence."""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import pymupdf
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from src.retriever import embed_passages

_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_PRESENTATION_FORM_RE = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")
_LEADING_TA_MARBUTA_RE = re.compile(r"(?:^|[\s\u060C؛،\(\[])ة[\u0621-\u064A]{2,}")
_REVERSED_FRAGMENT_RE = re.compile(r"(?:^|\s)(?:نم|نع|ىف|ديزي|رهظت)(?:\s|$)")

# Common Arabic regulatory words in visual/reversed order used to flag reversed PDF extraction.
_REVERSED_ARABIC_WORDS = (
    "ةعماج",
    "ةيلك",
    "بلاط",
    "ةدام",
    "ةحئال",
    "جمانرب",
    "تاعاس",
    "لصف",
    "رارق",
    "ررقم",
    "مولع",
    "بساح",
)

# Common Microsoft Word PDF CMap artifacts where Lam-Alef ligature decomposes in reversed order.
_LAM_ALEF_ARTIFACTS = (
    "األ",
    "اإل",
    "اآل",
    "الالئحة",
    "خالل",
    "مجاالت",
    "االهلية",
    "االمتحان",
    "االنسحاب",
)


@dataclass(frozen=True)
class PageDocument:
    """Extracted text and provenance for a single PDF page."""

    source_file: str
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    """Chunked text ready for embedding and storage in ChromaDB."""

    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class ArabicQualityReport:
    """Diagnostic summary of Arabic/English text quality for one PDF page."""

    source_file: str
    page_number: int
    warnings: list[str]
    arabic_char_count: int
    latin_char_count: int


def _char_script(ch: str) -> str:
    """Classify a character for visual-to-logical BiDi token reconstruction."""
    code = ord(ch)
    if ch in "،؛؟":
        return "PUNC"
    if (
        (0x0620 <= code <= 0x065F)
        or (0x066E <= code <= 0x06D3)
        or (0x06D5 <= code <= 0x06FF)
        or (0xFB50 <= code <= 0xFDFF)
        or (0xFE70 <= code <= 0xFEFF)
    ):
        return "AR"
    if (
        ("A" <= ch <= "Z")
        or ("a" <= ch <= "z")
        or ("0" <= ch <= "9")
        or (0x0660 <= code <= 0x0669)
        or ch in "_/@%"
    ):
        return "LTR"
    if ch in ".-+:":
        return "NUM_PUNC"
    return "PUNC"


_Glyph = tuple[float, float, float, float, float, float, str]


def reconstruct_text_from_rawdict(rawdict: dict[str, Any]) -> str:
    """Reconstruct logical reading order from PyMuPDF `rawdict` glyph bounding boxes.

    Justified Arabic paragraphs exported by MS Word store glyphs in visual left-to-right
    chunks, and Lam-Alef ligatures place zero-width Alef at the right edge of Lam.
    Sorting Arabic glyphs right-to-left by x1 descending (tie-breaking equal x1 by x0
    ascending) restores logical Arabic order and eliminates Lam-Alef reversal artifacts.
    """
    glyphs: list[_Glyph] = []
    for block in rawdict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = float(span.get("size", 11.0))
                for char_info in span.get("chars", []):
                    ch = char_info.get("c", "")
                    if not ch or ch.isspace() or ch in ("\x00", "\xad"):
                        continue
                    x0, y0, x1, y1 = (float(v) for v in char_info["bbox"])
                    yc = (y0 + y1) / 2.0
                    glyphs.append((x0, y0, x1, y1, yc, size, ch))

    if not glyphs:
        return ""

    glyphs.sort(key=lambda g: g[4])
    visual_lines: list[tuple[float, list[_Glyph]]] = []
    current_line = [glyphs[0]]
    current_y = glyphs[0][4]
    for glyph in glyphs[1:]:
        tolerance = max(3.0, glyph[5] * 0.45)
        if abs(glyph[4] - current_y) <= tolerance:
            current_line.append(glyph)
            current_y = (current_y * (len(current_line) - 1) + glyph[4]) / len(current_line)
        else:
            visual_lines.append((current_y, current_line))
            current_line = [glyph]
            current_y = glyph[4]
    visual_lines.append((current_y, current_line))
    visual_lines.sort(key=lambda item: item[0])

    out_lines: list[str] = []
    for _, line_glyphs in visual_lines:
        line_glyphs.sort(key=lambda g: (round(g[0], 2), round(g[2], 2)))
        tokens: list[tuple[str, list[_Glyph]]] = []

        for glyph in line_glyphs:
            script = _char_script(glyph[6])
            if not tokens:
                tokens.append((script, [glyph]))
                continue

            prev_script, prev_list = tokens[-1]
            prev_glyph = prev_list[-1]
            gap = glyph[0] - prev_glyph[2]
            space_thresh = max(1.6, min(prev_glyph[5], glyph[5]) * 0.18)
            compatible = (
                (prev_script == script and script in ("AR", "LTR"))
                or (prev_script == "LTR" and script == "NUM_PUNC")
                or (prev_script == "NUM_PUNC" and script == "LTR")
            )
            if gap <= space_thresh and compatible:
                if prev_script == "NUM_PUNC" and script == "LTR":
                    tokens[-1] = ("LTR", prev_list)
                prev_list.append(glyph)
            else:
                tokens.append((script, [glyph]))

        rendered: list[tuple[str, str]] = []
        ar_chars = 0
        lat_chars = 0
        for script, g_list in tokens:
            if script == "AR":
                ar_chars += len(g_list)
                # Tie-break equal x1 with +x0 so full-width Lam precedes zero-width Alef.
                rtl_glyphs = sorted(g_list, key=lambda g: (-round(g[2], 2), round(g[0], 2)))
                rendered.append(("AR", "".join(g[6] for g in rtl_glyphs)))
            elif script == "LTR":
                lat_chars += sum(
                    1 for g in g_list if ("A" <= g[6] <= "Z") or ("a" <= g[6] <= "z")
                )
                rendered.append(("LTR", "".join(g[6] for g in g_list)))
            else:
                rendered.append(("NEU", "".join(g[6] for g in g_list)))

        is_rtl_line = ar_chars > 0 and ar_chars >= lat_chars
        final_words: list[str] = []
        if is_rtl_line:
            rev = list(reversed(rendered))
            idx = 0
            while idx < len(rev):
                if rev[idx][0] == "LTR":
                    end = idx
                    while end < len(rev) and rev[end][0] == "LTR":
                        end += 1
                    final_words.extend(word for _, word in reversed(rev[idx:end]))
                    idx = end
                else:
                    final_words.append(rev[idx][1])
                    idx += 1
        else:
            idx = 0
            while idx < len(rendered):
                if rendered[idx][0] == "AR":
                    end = idx
                    while end < len(rendered) and rendered[end][0] == "AR":
                        end += 1
                    final_words.extend(word for _, word in reversed(rendered[idx:end]))
                    idx = end
                else:
                    final_words.append(rendered[idx][1])
                    idx += 1

        out_lines.append(" ".join(final_words))

    return "\n".join(out_lines)


def extract_page_text_geometric(page: pymupdf.Page) -> str:
    """Extract page text using PyMuPDF glyph coordinates to preserve RTL Arabic order."""
    rawdict = page.get_text(
        "rawdict",
        flags=pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_PRESERVE_WHITESPACE,
    )
    return reconstruct_text_from_rawdict(rawdict)


def clean_extracted_text(text: str) -> str:
    """Normalize PDF text while preserving Arabic, English, numbers, and line breaks."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\x00", "").replace("\xad", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Repair unambiguous reversed Lam-Alef ligature decompositions emitted by MS Word PDFs.
    normalized = (
        normalized.replace("األ", "الأ")
        .replace("اإل", "الإ")
        .replace("اآل", "الآ")
    )

    cleaned_lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        line = re.sub(r"\s+([،؛؟.,:)\]])", r"\1", line)
        line = re.sub(r"([(\[])\s+", r"\1", line)
        cleaned_lines.append(line)

    collapsed = "\n".join(cleaned_lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def detect_arabic_text_issues(text: str) -> list[str]:
    """Inspect extracted text for reversed Arabic words, ligature artifacts, or corruption."""
    warnings: list[str] = []
    if not text or not text.strip():
        warnings.append("Page text is empty (may be a scanned image or blank page).")
        return warnings

    if "\ufffd" in text:
        warnings.append(
            "Replacement character (U+FFFD) found; font encoding may be corrupted."
        )

    if _PRESENTATION_FORM_RE.search(text):
        warnings.append("Unnormalized Arabic Presentation Forms (FB50-FEFF) detected.")

    matched_reversed = [word for word in _REVERSED_ARABIC_WORDS if word in text]
    if matched_reversed:
        warnings.append(
            "Potential reversed Arabic word(s) detected: " + ", ".join(matched_reversed)
        )

    if _REVERSED_FRAGMENT_RE.search(text):
        warnings.append(
            "Fragmented/reversed Arabic token(s) detected (e.g. نم، نع، ديزي)."
        )

    if _LEADING_TA_MARBUTA_RE.search(text):
        warnings.append(
            "Arabic token(s) starting with ta-marbuta (ة) detected, suggesting reversed order."
        )

    matched_ligatures = [token for token in _LAM_ALEF_ARTIFACTS if token in text]
    if matched_ligatures:
        warnings.append(
            "Lam-Alef CMap ordering artifact(s) detected: " + ", ".join(matched_ligatures)
        )

    return warnings


def load_pdfs(
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> tuple[list[PageDocument], list[ArabicQualityReport]]:
    """Load all PDF files from `data_dir` and report per-page Arabic quality diagnostics."""
    directory = Path(data_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Data directory '{directory}' does not exist.")

    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in '{directory}'.")

    pages: list[PageDocument] = []
    reports: list[ArabicQualityReport] = []

    for pdf_path in pdf_paths:
        with pymupdf.open(pdf_path) as document:
            for page_index in range(len(document)):
                page_number = page_index + 1
                raw_text = extract_page_text_geometric(document[page_index])
                cleaned_text = clean_extracted_text(raw_text)
                cleaned_warnings = detect_arabic_text_issues(cleaned_text)

                pages.append(
                    PageDocument(
                        source_file=pdf_path.name,
                        page_number=page_number,
                        text=cleaned_text,
                    )
                )
                reports.append(
                    ArabicQualityReport(
                        source_file=pdf_path.name,
                        page_number=page_number,
                        warnings=cleaned_warnings,
                        arabic_char_count=len(_ARABIC_CHAR_RE.findall(cleaned_text)),
                        latin_char_count=len(_LATIN_CHAR_RE.findall(cleaned_text)),
                    )
                )

    return pages, reports


_LATIN_COURSE_CODE_RE = re.compile(r"\b(ROB|COM|BAS|GEN)\s*-?\s*(\d{3})\b", re.IGNORECASE)
_ARABIC_COURSE_CODE_META_RE = re.compile(
    r"(?:^|[\s(\[،])(بنل|حسب|أسس|اسس|عام)\s*-?\s*(\d{3})(?!\d)"
)
_ARABIC_TO_LATIN_PREFIX: dict[str, str] = {
    "بنل": "ROB",
    "حسب": "COM",
    "أسس": "BAS",
    "اسس": "BAS",
    "عام": "GEN",
}
_ARTICLE_RE = re.compile(r"(?:^|[\s،.(])(?:مادة|المادة)\s*\(?\s*(\d{1,3})\s*\)?")
_LEVEL_CUE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:المستوى\s+الأول|الفرقة\s+الأولى|\bLevel\s*1\b)", re.IGNORECASE), "Level 1"),
    (re.compile(r"(?:المستوى\s+الثاني|الفرقة\s+الثانية|\bLevel\s*2\b)", re.IGNORECASE), "Level 2"),
    (re.compile(r"(?:المستوى\s+الثالث|الفرقة\s+الثالثة|\bLevel\s*3\b)", re.IGNORECASE), "Level 3"),
    (re.compile(r"(?:المستوى\s+الرابع|الفرقة\s+الرابعة|\bLevel\s*4\b)", re.IGNORECASE), "Level 4"),
)
_TABLE_STRUCTURE_RE = re.compile(
    r"(?:المتطلب\s+السابق|المتطلبات\s+السابقة|كود\s+المقرر|جدول\s*\(?\d+\)?|"
    r"توزيع\s+الدرجات|\bPrerequisite\b)"
)
_HOURS_SPLIT_RE = re.compile(r"(?:نظري|عملي|تمارين)")


def extract_chunk_metadata_cues(text: str) -> dict[str, str | int]:
    """Extract optional Chroma scalar metadata cues (courses, articles, levels, table type)."""
    cues: dict[str, str | int] = {}

    codes: list[str] = []
    seen_codes: set[str] = set()
    for match in _LATIN_COURSE_CODE_RE.finditer(text):
        normalized = f"{match.group(1).upper()} {match.group(2)}"
        if normalized not in seen_codes:
            seen_codes.add(normalized)
            codes.append(normalized)
    for match in _ARABIC_COURSE_CODE_META_RE.finditer(text):
        latin = f"{_ARABIC_TO_LATIN_PREFIX[match.group(1)]} {match.group(2)}"
        arabic = f"{match.group(1)} {match.group(2)}"
        for code in (arabic, latin):
            if code not in seen_codes:
                seen_codes.add(code)
                codes.append(code)
    if re.search(r"مقرر\s+المشروع|مشروع\s+التخرج", text):
        for project_code in ("COM 400", "COM 498"):
            if project_code not in seen_codes:
                seen_codes.add(project_code)
                codes.append(project_code)
    if codes:
        cues["course_codes"] = ", ".join(codes)

    articles: list[str] = []
    seen_articles: set[str] = set()
    for match in _ARTICLE_RE.finditer(text):
        num = match.group(1)
        if num not in seen_articles:
            seen_articles.add(num)
            articles.append(num)
    if articles:
        cues["articles"] = ", ".join(articles)

    levels = [label for pattern, label in _LEVEL_CUE_RULES if pattern.search(text)]
    if levels:
        cues["academic_level"] = ", ".join(levels)

    is_table_or_curriculum = bool(
        codes
        or _TABLE_STRUCTURE_RE.search(text)
        or ("ساعات معتمدة" in text and _HOURS_SPLIT_RE.search(text))
    )
    if is_table_or_curriculum:
        cues["chunk_type"] = "table_or_curriculum"

    return cues


def chunk_documents(
    pages: Sequence[PageDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split pages with RecursiveCharacterTextSplitter and attach source/page/chunk metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "،", " ", ""],
    )

    chunks: list[TextChunk] = []
    for page in pages:
        if not page.text or not page.text.strip():
            continue

        split_texts = splitter.split_text(page.text)
        chunk_counter = 0
        for piece in split_texts:
            stripped = piece.strip()
            if not stripped:
                continue
            chunk_counter += 1
            chunk_id = f"{page.source_file}::p{page.page_number:03d}::c{chunk_counter:03d}"
            metadata: dict[str, Any] = {
                "source_file": page.source_file,
                "page_number": int(page.page_number),
                "chunk_id": chunk_id,
            }
            metadata.update(extract_chunk_metadata_cues(stripped))
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=stripped,
                    metadata=metadata,
                )
            )

    return chunks


def persist_chunks_to_chroma(
    chunks: Sequence[TextChunk],
    chroma_dir: Path | str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    reset_collection: bool = True,
) -> int:
    """Embed chunks using E5 passage embeddings and persist them to ChromaDB."""
    if not chunks:
        raise ValueError("Cannot persist an empty chunk list.")

    target_dir = Path(chroma_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(target_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    if reset_collection:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )

    embeddings = embed_passages(
        [chunk.text for chunk in chunks],
        model_name=model_name,
        batch_size=batch_size,
    )

    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch_chunks],
            documents=[chunk.text for chunk in batch_chunks],
            metadatas=[dict(chunk.metadata) for chunk in batch_chunks],
            embeddings=embeddings[start:end],
        )

    return total


def _select_sample_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    step = (total - 1) / (count - 1)
    indices = sorted({round(i * step) for i in range(count)})
    return indices


def ingest_directory(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    chroma_dir: Path | str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    sample_count: int = 4,
    reset_collection: bool = True,
) -> list[TextChunk]:
    """Run the full PDF ingestion pipeline and print quality warnings plus sample chunks."""
    pages, reports = load_pdfs(data_dir=data_dir)
    chunks = chunk_documents(
        pages=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    flagged_reports = [report for report in reports if report.warnings]
    print(f"Loaded {len(pages)} pages from '{data_dir}' -> produced {len(chunks)} chunks.")
    if flagged_reports:
        print(
            f"[WARNING] Arabic text quality checks flagged {len(flagged_reports)}/{len(pages)} "
            "pages (inspect samples below):"
        )
        for report in flagged_reports[:10]:
            joined = "; ".join(report.warnings)
            print(f"  - {report.source_file} p.{report.page_number}: {joined}")
        if len(flagged_reports) > 10:
            print(f"  ... and {len(flagged_reports) - 10} more pages with warnings.")
    else:
        print("Arabic text quality check passed with zero warnings.")

    sample_indices = _select_sample_indices(len(chunks), sample_count)
    if sample_indices:
        print("\n=== Sample Extracted Chunks ===")
        for idx in sample_indices:
            chunk = chunks[idx]
            print(
                f"\n--- Chunk {idx + 1}/{len(chunks)} | {chunk.chunk_id} "
                f"(len={len(chunk.text)}) ---"
            )
            print(chunk.text)

    persisted = persist_chunks_to_chroma(
        chunks=chunks,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        model_name=model_name,
        reset_collection=reset_collection,
    )
    print(
        f"\nPersisted {persisted} chunks to '{chroma_dir}' "
        f"(collection='{collection_name}', model='{model_name}')."
    )
    return chunks


def _configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for PDF ingestion into ChromaDB."""
    _configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Ingest Helwan regulations PDFs into ChromaDB.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory with PDFs.")
    parser.add_argument(
        "--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB persistence path."
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Collection name.")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Text splitter chunk size."
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Text splitter overlap.",
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    parser.add_argument(
        "--samples", type=int, default=4, help="Number of sample chunks to print."
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Upsert into existing collection instead of recreating it.",
    )
    args = parser.parse_args(argv)

    ingest_directory(
        data_dir=args.data_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model,
        sample_count=args.samples,
        reset_collection=not args.no_reset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
