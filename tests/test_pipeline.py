"""Offline unit and integration tests for the Helwan RAG QA & Evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pytest

from eval.run_eval import compute_mrr, compute_recall_at_k, evaluate_with_rubric
from src import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.generator import (
    ABSTENTION_PHRASE_AR,
    ABSTENTION_PHRASE_EN,
    ConfigurationError,
    RAGGenerator,
    build_rag_prompts,
    create_llm_backend,
    detect_abstention,
    extract_cited_pages,
)
from src.ingestion import (
    PageDocument,
    chunk_documents,
    clean_extracted_text,
    detect_arabic_text_issues,
    reconstruct_text_from_rawdict,
)
from src.retriever import ChromaRetriever, RetrievedChunk


class _FakeLLMBackend:
    """Deterministic test double implementing the LLMBackend protocol."""

    provider_name = "fake"
    model_name = "fake-model-v1"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, float]] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append((system_prompt, user_prompt, temperature))
        return self.response


def test_chunk_documents_respects_size_and_metadata() -> None:
    sentence = "يشترط لدخول الامتحان النهائي تحقيق نسبة حضور لا تقل عن 75% من المحاضرات. "
    long_text = sentence * 25
    pages = [
        PageDocument(source_file="bylaws.pdf", page_number=8, text=long_text),
        PageDocument(source_file="bylaws.pdf", page_number=9, text="   "),
        PageDocument(source_file="bylaws.pdf", page_number=10, text="CGPA >= 2.0 للنجاح."),
    ]

    chunks = chunk_documents(
        pages=pages,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    assert len(chunks) >= 3
    seen_ids: set[str] = set()
    for chunk in chunks:
        assert len(chunk.text) <= DEFAULT_CHUNK_SIZE
        assert set(chunk.metadata.keys()) == {"source_file", "page_number", "chunk_id"}
        assert chunk.metadata["source_file"] == "bylaws.pdf"
        assert chunk.metadata["page_number"] in {8, 10}
        assert chunk.metadata["chunk_id"] == chunk.chunk_id
        assert chunk.chunk_id not in seen_ids
        seen_ids.add(chunk.chunk_id)


def test_clean_extracted_text_and_arabic_quality_warnings() -> None:
    raw_corrupted_cmap = "جامعة حلوان االهلية\r\nبرنامج )برمجيات اإلنسان اآللي(\u0000"
    cleaned = clean_extracted_text(raw_corrupted_cmap)
    assert "جامعة حلوان االهلية" in cleaned
    assert "الإنسان الآلي" in cleaned
    assert "\x00" not in cleaned

    clean_sample = "تمنح جامعة حلوان الأهلية درجة البكالوريوس في برمجيات الإنسان الآلي (ROB 404)."
    assert detect_arabic_text_issues(clean_sample) == []

    reversed_sample = "ةعماج ناولح ةيلك مولع بساحلا"
    warnings = detect_arabic_text_issues(reversed_sample)
    assert any("reversed" in warning.lower() for warning in warnings)

    fragmented_sample = "بما لا ديزي نع 83"
    frag_warnings = detect_arabic_text_issues(fragmented_sample)
    assert any("reversed" in warning.lower() for warning in frag_warnings)

    replacement_warnings = detect_arabic_text_issues("النص غير مقروء \ufffd")
    assert any("U+FFFD" in warning for warning in replacement_warnings)

    empty_warnings = detect_arabic_text_issues("   ")
    assert any("empty" in warning.lower() for warning in empty_warnings)


def test_reconstruct_text_from_rawdict_handles_rtl_lam_alef_and_ltr() -> None:
    # Build synthetic glyphs emitted out-of-order on one line:
    # Logical reading order (right-to-left): "الدرجة CGPA لا تزيد عن 83"
    # Includes zero-width Alef sharing x1 with Lam to test Lam-Alef tie-breaking.
    chars = [
        # "الدرجة" on the right (x ~ 430..470), emitted scrambled:
        {"c": "د", "bbox": (454.0, 100.0, 459.0, 112.0)},
        {"c": "ر", "bbox": (448.0, 100.0, 454.0, 112.0)},
        {"c": "ا", "bbox": (464.0, 100.0, 467.0, 112.0)},
        {"c": "ل", "bbox": (459.0, 100.0, 464.0, 112.0)},
        {"c": "ة", "bbox": (437.0, 100.0, 442.0, 112.0)},
        {"c": "ج", "bbox": (442.0, 100.0, 448.0, 112.0)},
        # "CGPA" embedded LTR run (x ~ 390..425):
        {"c": "C", "bbox": (390.0, 100.0, 398.0, 112.0)},
        {"c": "G", "bbox": (398.0, 100.0, 407.0, 112.0)},
        {"c": "P", "bbox": (407.0, 100.0, 415.0, 112.0)},
        {"c": "A", "bbox": (415.0, 100.0, 424.0, 112.0)},
        # "لا" with zero-width Alef at right edge of Lam (x1 = 370.0 for both!):
        {"c": "ا", "bbox": (370.0, 100.0, 370.0, 112.0)},
        {"c": "ل", "bbox": (362.0, 100.0, 370.0, 112.0)},
        # "83" on the left (x ~ 330..346):
        {"c": "8", "bbox": (330.0, 100.0, 338.0, 112.0)},
        {"c": "3", "bbox": (338.0, 100.0, 346.0, 112.0)},
    ]
    rawdict = {
        "blocks": [
            {
                "type": 0,
                "lines": [{"spans": [{"size": 11.0, "chars": chars}]}],
            }
        ]
    }
    reconstructed = reconstruct_text_from_rawdict(rawdict)
    assert reconstructed == "الدرجة CGPA لا 83"


def test_chroma_retriever_interface_and_scoring() -> None:
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="test_helwan_regs",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=["doc::p008::c001", "doc::p012::c001", "doc::p046::c001"],
        documents=[
            "نسبة الحضور المطلوبة 75% لدخول الامتحان النهائي.",
            "يتطلب التخرج اجتياز 138 ساعة معتمدة بمعدل تراكمي لا يقل عن 2.",
            "ROB 404 covers mobile robot control architectures and navigation.",
        ],
        metadatas=[
            {"source_file": "bylaws.pdf", "page_number": 8, "chunk_id": "doc::p008::c001"},
            {"source_file": "bylaws.pdf", "page_number": 12, "chunk_id": "doc::p012::c001"},
            {"source_file": "bylaws.pdf", "page_number": 46, "chunk_id": "doc::p046::c001"},
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    def fake_embed_fn(query: str) -> list[float]:
        if "ساعة" in query or "الساعات" in query or "138" in query:
            return [0.0, 1.0, 0.0]
        if "ROB 404" in query:
            return [0.0, 0.0, 1.0]
        return [1.0, 0.0, 0.0]

    retriever = ChromaRetriever(
        collection_name="test_helwan_regs",
        embed_fn=fake_embed_fn,
        client=client,
    )

    assert retriever.count() == 3
    results = retriever.retrieve("ما عدد الساعات المعتمدة للتخرج؟", top_k=10)
    assert len(results) == 3
    assert results[0].chunk_id == "doc::p012::c001"
    assert results[0].page_number == 12
    assert results[0].source_file == "bylaws.pdf"
    assert results[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert results[0].distance == pytest.approx(0.0, abs=1e-5)
    assert results[0].distance <= results[1].distance <= results[2].distance

    with pytest.raises(ValueError):
        retriever.retrieve("   ", top_k=3)

    with pytest.raises(ValueError):
        retriever.retrieve("test", top_k=0)


def test_prompts_citations_and_abstention_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="bylaws.pdf::p008::c001",
            text="يتطلب دخول الامتحان النهائي نسبة حضور 75% على الأقل.",
            source_file="bylaws.pdf",
            page_number=8,
            distance=0.05,
            similarity=0.95,
            metadata={"source_file": "bylaws.pdf", "page_number": 8},
        )
    ]
    sys_prompt, user_prompt = build_rag_prompts("ما نسبة الحضور؟", chunks)
    assert ABSTENTION_PHRASE_AR in sys_prompt
    assert "صفحة: 8" in user_prompt
    assert "bylaws.pdf" in user_prompt

    assert extract_cited_pages("يجب حضور 75% [صفحة 8] وراجع [صفحة ١٢، 13] و [Page 46].") == [
        8,
        12,
        13,
        46,
    ]
    assert detect_abstention(ABSTENTION_PHRASE_AR) is True
    assert detect_abstention(ABSTENTION_PHRASE_EN) is True
    assert detect_abstention("نسبة الحضور 75% [صفحة 8].") is False

    fake_backend = _FakeLLMBackend("يجب حضور 75% على الأقل [صفحة 8] ولا توجد بيانات [صفحة 99].")
    generator = RAGGenerator(backend=fake_backend)
    result = generator.generate("ما نسبة الحضور؟", chunks)
    assert result.abstained is False
    assert len(result.citations) == 1
    assert result.citations[0].page_number == 8
    assert result.citations[0].chunk_id == "bylaws.pdf::p008::c001"

    # Empty context triggers immediate abstention without calling backend.
    empty_ar = generator.generate("ما قيمة المصروفات؟", [])
    assert empty_ar.abstained is True
    assert empty_ar.answer == ABSTENTION_PHRASE_AR
    assert empty_ar.citations == []
    assert len(fake_backend.calls) == 1

    empty_en = generator.generate("What are the annual tuition fees?", [])
    assert empty_en.abstained is True
    assert empty_en.answer == ABSTENTION_PHRASE_EN
    assert empty_en.citations == []

    # Backend abstention clears citations even if stray numbers exist.
    abstaining_backend = _FakeLLMBackend(ABSTENTION_PHRASE_AR)
    abst_gen = RAGGenerator(backend=abstaining_backend)
    abst_res = abst_gen.generate("ما قيمة المصروفات؟", chunks)
    assert abst_res.abstained is True
    assert abst_res.citations == []

    # Missing keys raise ConfigurationError cleanly without hardcoded fallback.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        create_llm_backend()


def test_eval_metrics_and_starter_dataset_schema() -> None:
    assert compute_recall_at_k([12, 8, 5], [8, 12], k=3) == pytest.approx(1.0)
    assert compute_recall_at_k([12, 8, 5], [8, 19], k=3) == pytest.approx(0.5)
    assert compute_recall_at_k([1, 2], [], k=2) is None

    assert compute_mrr([5, 8, 12], [8, 12], k=3) == pytest.approx(0.5)
    assert compute_mrr([5, 8, 12], [19], k=3) == pytest.approx(0.0)
    assert compute_mrr([5], [], k=3) is None

    unanswerable_item = {
        "id": "u1",
        "category": "unanswerable",
        "answerable": False,
        "expected_pages": [],
        "expected_answer": ABSTENTION_PHRASE_AR,
    }
    abstain_eval = evaluate_with_rubric(
        item=unanswerable_item,
        answer=ABSTENTION_PHRASE_AR,
        abstained=True,
        cited_pages=[],
        chunks=[],
    )
    assert abstain_eval["abstention_correct"] is True
    assert abstain_eval["faithfulness"] == 1.0

    dataset_path = Path("eval/test_dataset.json")
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert len(items) >= 20
    categories = {item["category"] for item in items}
    assert categories == {
        "direct_factual",
        "multi_hop",
        "tabular_numerical",
        "unanswerable",
    }
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))
