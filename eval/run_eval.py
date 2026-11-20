"""Evaluation runner computing Recall@K, MRR, and generation quality metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL  # noqa: E402
from src.generator import (  # noqa: E402
    ConfigurationError,
    LLMBackend,
    LLMGenerationError,
    RAGGenerator,
    create_llm_backend,
    detect_abstention,
    extract_cited_pages,
)
from src.retriever import ChromaRetriever, RetrievedChunk  # noqa: E402

_TOKEN_RE = re.compile(r"[\u0621-\u064AA-Za-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_EASTERN_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ARABIC_STOPWORDS = frozenset(
    {
        "في",
        "من",
        "على",
        "إلى",
        "الى",
        "عن",
        "أو",
        "و",
        "أن",
        "إن",
        "ما",
        "لا",
        "لم",
        "لن",
        "مع",
        "كل",
        "هذا",
        "هذه",
        "ذلك",
        "التي",
        "الذي",
        "يكون",
        "يتم",
        "حسب",
        "وفق",
        "صفحة",
    }
)
_ENGLISH_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "as",
        "by",
        "from",
        "page",
    }
)


def _normalize_arabic_text(text: str) -> str:
    normalized = text.translate(_EASTERN_ARABIC_DIGITS).lower()
    normalized = (
        normalized.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    return normalized


def _extract_content_tokens(text: str) -> set[str]:
    normalized = _normalize_arabic_text(text)
    tokens = set(_TOKEN_RE.findall(normalized))
    return {
        token
        for token in tokens
        if len(token) > 1
        and token not in _ARABIC_STOPWORDS
        and token not in _ENGLISH_STOPWORDS
    }


def _extract_numbers(text: str) -> set[str]:
    normalized = text.translate(_EASTERN_ARABIC_DIGITS)
    return set(_NUMBER_RE.findall(normalized))


def compute_recall_at_k(
    retrieved_pages: Sequence[int],
    expected_pages: Sequence[int],
    k: int,
) -> float | None:
    """Compute fraction of unique expected pages present in the top-k retrieved pages."""
    expected_set = set(expected_pages)
    if not expected_set:
        return None
    top_pages = set(retrieved_pages[:k])
    return len(top_pages & expected_set) / len(expected_set)


def compute_mrr(
    retrieved_pages: Sequence[int],
    expected_pages: Sequence[int],
    k: int,
) -> float | None:
    """Compute Mean Reciprocal Rank contribution of the first matching retrieved page."""
    expected_set = set(expected_pages)
    if not expected_set:
        return None
    for rank, page in enumerate(retrieved_pages[:k], start=1):
        if page in expected_set:
            return 1.0 / rank
    return 0.0


def evaluate_with_rubric(
    item: dict[str, Any],
    answer: str,
    abstained: bool,
    cited_pages: Sequence[int],
    chunks: Sequence[RetrievedChunk],
) -> dict[str, Any]:
    """Evaluate answer faithfulness, accuracy, and citations using a deterministic rubric."""
    answerable = bool(item.get("answerable", True))
    expected_pages = set(item.get("expected_pages", []))
    cited_set = set(cited_pages)
    retrieved_pages = {chunk.page_number for chunk in chunks}

    if not answerable:
        abstention_correct = bool(abstained)
        score = 1.0 if abstention_correct else 0.0
        citation_score = 1.0 if (abstention_correct and not cited_set) else 0.0
        return {
            "method": "rubric",
            "faithfulness": score,
            "answer_accuracy": score,
            "citation_quality": citation_score,
            "abstention_correct": abstention_correct,
            "rationale": (
                "Correctly abstained on unanswerable question."
                if abstention_correct
                else "Hallucinated or failed to abstain on unanswerable question."
            ),
        }

    if abstained:
        return {
            "method": "rubric",
            "faithfulness": 1.0,
            "answer_accuracy": 0.0,
            "citation_quality": 0.0,
            "abstention_correct": False,
            "rationale": "Abstained on an answerable question (likely retrieval or context miss).",
        }

    expected_answer = str(item.get("expected_answer", ""))
    expected_tokens = _extract_content_tokens(expected_answer)
    answer_tokens = _extract_content_tokens(answer)
    context_tokens = _extract_content_tokens("\n".join(chunk.text for chunk in chunks))

    token_recall = (
        len(expected_tokens & answer_tokens) / len(expected_tokens)
        if expected_tokens
        else 0.0
    )

    expected_numbers = _extract_numbers(expected_answer)
    answer_numbers = _extract_numbers(answer)
    context_numbers = _extract_numbers("\n".join(chunk.text for chunk in chunks))

    if expected_numbers:
        number_recall = len(expected_numbers & answer_numbers) / len(expected_numbers)
        answer_accuracy = round(0.65 * token_recall + 0.35 * number_recall, 4)
    else:
        answer_accuracy = round(token_recall, 4)

    token_support = (
        len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
    )
    if answer_numbers:
        number_support = len(answer_numbers & context_numbers) / len(answer_numbers)
        faithfulness = round(0.7 * token_support + 0.3 * number_support, 4)
    else:
        faithfulness = round(token_support, 4)

    if not cited_set:
        citation_quality = 0.0
    else:
        valid_cited = len(cited_set & retrieved_pages) / len(cited_set)
        expected_cov = (
            len(cited_set & expected_pages) / len(expected_pages) if expected_pages else 1.0
        )
        citation_quality = round(0.5 * valid_cited + 0.5 * expected_cov, 4)

    return {
        "method": "rubric",
        "faithfulness": faithfulness,
        "answer_accuracy": answer_accuracy,
        "citation_quality": citation_quality,
        "abstention_correct": True,
        "rationale": (
            f"token_recall={token_recall:.2f}, context_support={token_support:.2f}, "
            f"cited_pages={sorted(cited_set)}"
        ),
    }


def _parse_judge_json(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response JSON must be an object.")
    return parsed


def _clamp_unit(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(1.0, numeric)), 4)


_RETRYABLE_SIGNALS = (
    "429",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "quota",
    "too many requests",
    "temporarily unavailable",
    "overloaded",
    "503",
    "timeout",
    "timed out",
    "connection reset",
)
_RETRY_HINT_RE = re.compile(
    r"(?:retry\s+(?:in|after)|retrydelay['\":\s]+)\s*(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(daily in message for daily in ("perday", "per day", "daily")):
        return False
    return any(signal in message for signal in _RETRYABLE_SIGNALS)


def _extract_retry_delay(exc: Exception, default_delay: float) -> float:
    match = _RETRY_HINT_RE.search(str(exc))
    if match:
        hinted = float(match.group(1)) + 1.0
        return max(1.0, min(90.0, hinted))
    return max(1.0, min(90.0, default_delay))


def _call_with_retry[T](
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    base_delay: float = 4.0,
) -> T:
    """Invoke an LLM API call with exponential backoff on transient rate-limit errors."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable_error(exc):
                raise
            delay = _extract_retry_delay(exc, base_delay * (2**attempt))
            print(
                f"    [retry {attempt + 1}/{max_retries}] transient rate/network limit; "
                f"sleeping {delay:.1f}s...",
                flush=True,
            )
            time.sleep(delay)
            attempt += 1


def evaluate_with_llm_judge(
    backend: LLMBackend,
    item: dict[str, Any],
    answer: str,
    chunks: Sequence[RetrievedChunk],
    max_retries: int = 5,
) -> dict[str, Any]:
    """Score faithfulness, accuracy, and citation quality via LLM-as-a-judge JSON output."""
    abstained = detect_abstention(answer)
    cited_pages = extract_cited_pages(answer)
    rubric_fallback = evaluate_with_rubric(
        item=item,
        answer=answer,
        abstained=abstained,
        cited_pages=cited_pages,
        chunks=chunks,
    )

    context_preview = "\n\n".join(
        f"[Page {chunk.page_number}] {chunk.text}" for chunk in chunks
    )
    system_prompt = (
        "You are an impartial RAG evaluation judge. Return ONLY valid JSON with keys: "
        '"faithfulness" (0.0-1.0), "answer_accuracy" (0.0-1.0), '
        '"citation_quality" (0.0-1.0), and "rationale" (concise string).'
    )
    user_prompt = (
        f"Question: {item['question']}\n"
        f"Category: {item.get('category')}\n"
        f"Answerable: {item.get('answerable', True)}\n"
        f"Expected Pages: {item.get('expected_pages', [])}\n"
        f"Reference Answer: {item.get('expected_answer', '')}\n\n"
        f"Retrieved Context:\n{context_preview}\n\n"
        f"Candidate Answer:\n{answer}\n\n"
        "Scoring criteria:\n"
        "- If Answerable is false: faithfulness=1.0, answer_accuracy=1.0, and "
        "citation_quality=1.0 ONLY if Candidate Answer clearly abstains without inventing "
        "facts or spurious citations; otherwise 0.0.\n"
        "- If Answerable is true: faithfulness measures whether every claim is grounded in "
        "Retrieved Context; answer_accuracy measures factual completeness against Reference "
        "Answer; citation_quality measures accurate page citations matching Retrieved "
        "Context/Expected Pages."
    )

    try:
        raw_judge = _call_with_retry(
            lambda: backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            ),
            max_retries=max_retries,
        )
        parsed = _parse_judge_json(raw_judge)
        return {
            "method": f"llm_judge:{backend.provider_name}",
            "faithfulness": _clamp_unit(
                parsed.get("faithfulness"), rubric_fallback["faithfulness"]
            ),
            "answer_accuracy": _clamp_unit(
                parsed.get("answer_accuracy"), rubric_fallback["answer_accuracy"]
            ),
            "citation_quality": _clamp_unit(
                parsed.get("citation_quality"), rubric_fallback["citation_quality"]
            ),
            "abstention_correct": rubric_fallback["abstention_correct"],
            "rationale": str(parsed.get("rationale", "")).strip() or rubric_fallback["rationale"],
        }
    except Exception as exc:
        rubric_fallback["rationale"] = f"LLM judge fallback ({exc}); {rubric_fallback['rationale']}"
        return rubric_fallback


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def _summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable_records = [r for r in records if r.get("answerable", True)]
    unanswerable_records = [r for r in records if not r.get("answerable", True)]

    return {
        "count": len(records),
        "answerable_count": len(answerable_records),
        "unanswerable_count": len(unanswerable_records),
        "recall_at_k": _mean_or_none([r.get("recall_at_k") for r in answerable_records]),
        "mrr": _mean_or_none([r.get("mrr") for r in answerable_records]),
        "faithfulness": _mean_or_none(
            [r["generation"]["faithfulness"] for r in records if r.get("generation")]
        ),
        "answer_accuracy": _mean_or_none(
            [r["generation"]["answer_accuracy"] for r in records if r.get("generation")]
        ),
        "citation_quality": _mean_or_none(
            [r["generation"]["citation_quality"] for r in records if r.get("generation")]
        ),
        "abstention_accuracy": _mean_or_none(
            [
                1.0 if r["generation"]["abstention_correct"] else 0.0
                for r in unanswerable_records
                if r.get("generation")
            ]
        ),
    }


def run_evaluation(
    dataset_path: Path | str = Path("eval/test_dataset.json"),
    output_path: Path | str = Path("eval/eval_results.json"),
    chroma_dir: Path | str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
    provider: str | None = None,
    llm_model: str | None = None,
    retrieval_only: bool = False,
    judge_mode: str = "auto",
    request_delay: float = 1.0,
    max_retries: int = 5,
    retriever: ChromaRetriever | None = None,
    generator: RAGGenerator | None = None,
    judge_backend: LLMBackend | None = None,
) -> dict[str, Any]:
    """Execute end-to-end RAG evaluation and write structured JSON metrics."""
    dataset_file = Path(dataset_path)
    items: list[dict[str, Any]] = json.loads(dataset_file.read_text(encoding="utf-8"))

    active_retriever = retriever or ChromaRetriever(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        model_name=model_name,
    )

    active_generator = generator
    active_judge = judge_backend
    resolved_provider: str | None = provider
    resolved_model_name: str | None = llm_model
    if not retrieval_only and active_generator is None:
        resolved_backend = create_llm_backend(provider=provider, model=llm_model)
        resolved_provider = resolved_backend.provider_name
        resolved_model_name = resolved_backend.model_name
        active_generator = RAGGenerator(backend=resolved_backend)
        if judge_mode in ("auto", "llm") and active_judge is None:
            active_judge = resolved_backend

    per_item_results: list[dict[str, Any]] = []
    by_category_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    total_items = len(items)
    for idx, item in enumerate(items, start=1):
        query = str(item["question"])
        expected_pages = [int(p) for p in item.get("expected_pages", [])]
        expected_chunks = [str(c) for c in item.get("expected_chunks", [])]
        answerable = bool(item.get("answerable", True))

        print(f"[{idx}/{total_items}] {item['id']} ({item['category']})...", flush=True)
        chunks = active_retriever.retrieve(query=query, top_k=top_k)
        retrieved_pages = [chunk.page_number for chunk in chunks]
        recall_at_k = compute_recall_at_k(retrieved_pages, expected_pages, k=top_k)
        mrr = compute_mrr(retrieved_pages, expected_pages, k=top_k)

        generation_payload: dict[str, Any] | None = None
        if not retrieval_only and active_generator is not None:
            try:
                gen_result = _call_with_retry(
                    lambda q=query, ch=chunks: active_generator.generate(query=q, chunks=ch),
                    max_retries=max_retries,
                )
                resolved_provider = gen_result.provider
                resolved_model_name = gen_result.model

                cited_pages = [cite.page_number for cite in gen_result.citations]
                if not cited_pages and not gen_result.abstained:
                    cited_pages = extract_cited_pages(gen_result.answer)

                if request_delay > 0:
                    time.sleep(request_delay)

                if judge_mode in ("auto", "llm") and active_judge is not None:
                    scores = evaluate_with_llm_judge(
                        backend=active_judge,
                        item=item,
                        answer=gen_result.answer,
                        chunks=chunks,
                        max_retries=max_retries,
                    )
                else:
                    scores = evaluate_with_rubric(
                        item=item,
                        answer=gen_result.answer,
                        abstained=gen_result.abstained,
                        cited_pages=cited_pages,
                        chunks=chunks,
                    )

                if request_delay > 0 and idx < total_items:
                    time.sleep(request_delay)

                generation_payload = {
                    "answer": gen_result.answer,
                    "abstained": gen_result.abstained,
                    "cited_pages": cited_pages,
                    "provider": gen_result.provider,
                    "model": gen_result.model,
                    **scores,
                }
            except (ConfigurationError, LLMGenerationError) as exc:
                generation_payload = {
                    "answer": "",
                    "abstained": False,
                    "cited_pages": [],
                    "method": "error",
                    "faithfulness": 0.0,
                    "answer_accuracy": 0.0,
                    "citation_quality": 0.0,
                    "abstention_correct": False,
                    "rationale": f"Generation error: {exc}",
                }

        record = {
            "id": item["id"],
            "category": item["category"],
            "language": item.get("language", "ar"),
            "answerable": answerable,
            "question": query,
            "expected_pages": expected_pages,
            "expected_chunks": expected_chunks,
            "retrieved_pages": retrieved_pages,
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "retrieved_scores": [round(chunk.similarity, 4) for chunk in chunks],
            "top_similarity": round(chunks[0].similarity, 4) if chunks else None,
            "recall_at_k": round(recall_at_k, 4) if recall_at_k is not None else None,
            "mrr": round(mrr, 4) if mrr is not None else None,
            "generation": generation_payload,
        }
        per_item_results.append(record)
        by_category_records[str(item["category"])].append(record)

    summary = {
        "overall": _summarize_records(per_item_results),
        "by_category": {
            cat: _summarize_records(recs) for cat, recs in by_category_records.items()
        },
    }

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "dataset_path": str(dataset_file),
            "chroma_dir": str(chroma_dir),
            "collection_name": collection_name,
            "embedding_model": model_name,
            "llm_provider": resolved_provider,
            "llm_model": resolved_model_name,
            "top_k": top_k,
            "retrieval_only": retrieval_only,
            "judge_mode": judge_mode,
            "total_questions": len(per_item_results),
        },
        "summary": summary,
        "results": per_item_results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _fmt_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _print_summary(report: dict[str, Any]) -> None:
    overall = report["summary"]["overall"]
    top_k = report["metadata"]["top_k"]
    print("\n=== Evaluation Summary ===")
    print(f"Questions: {overall['count']} | Top-K: {top_k}")
    print(
        f"Overall -> Recall@{top_k}: {_fmt_metric(overall['recall_at_k'])} | "
        f"MRR: {_fmt_metric(overall['mrr'])} | "
        f"Faithfulness: {_fmt_metric(overall['faithfulness'])} | "
        f"Answer Accuracy: {_fmt_metric(overall['answer_accuracy'])} | "
        f"Abstention Acc: {_fmt_metric(overall['abstention_accuracy'])}"
    )
    print("\nBy Category:")
    for category, metrics in report["summary"]["by_category"].items():
        print(
            f"  - {category:<18} (n={metrics['count']}) | "
            f"Recall@{top_k}={_fmt_metric(metrics['recall_at_k'])} | "
            f"MRR={_fmt_metric(metrics['mrr'])} | "
            f"Faith={_fmt_metric(metrics['faithfulness'])} | "
            f"Acc={_fmt_metric(metrics['answer_accuracy'])} | "
            f"Abst={_fmt_metric(metrics['abstention_accuracy'])}"
        )


def _configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for running retrieval & generation evaluation."""
    _configure_utf8_stdout()

    parser = argparse.ArgumentParser(
        description="Evaluate Helwan National University RAG pipeline."
    )
    parser.add_argument("--dataset", default="eval/test_dataset.json", help="Path to test JSON.")
    parser.add_argument("--output", default="eval/eval_results.json", help="Output JSON path.")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Collection name.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K chunks per query.")
    parser.add_argument("--provider", default=None, help="LLM provider: gemini or groq.")
    parser.add_argument("--llm-model", default=None, help="Optional LLM model override.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval metrics (Recall@K, MRR) without calling an LLM.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["auto", "llm", "rubric"],
        default="auto",
        help="Generation scoring mode.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Seconds to pause between LLM API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries on rate limits or transient network errors.",
    )
    args = parser.parse_args(argv)

    report = run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        model_name=args.model,
        top_k=args.top_k,
        provider=args.provider,
        llm_model=args.llm_model,
        retrieval_only=args.retrieval_only,
        judge_mode=args.judge_mode,
        request_delay=args.request_delay,
        max_retries=args.max_retries,
    )
    _print_summary(report)
    print(f"\nSaved detailed results to '{args.output}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
