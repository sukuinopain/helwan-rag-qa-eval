"""Offline unit and integration tests for the Helwan RAG QA & Evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings

from eval.generate_plots import generate_all_plots
from eval.run_eval import compute_mrr, compute_recall_at_k, evaluate_with_rubric
from src import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.generator import (
    ABSTENTION_PHRASE_AR,
    ABSTENTION_PHRASE_EN,
    ConfigurationError,
    GeminiBackend,
    GroqBackend,
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
    extract_chunk_metadata_cues,
    reconstruct_text_from_rawdict,
)
from src.retriever import ChromaRetriever, RetrievedChunk, normalize_and_expand_query


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
    monkeypatch.setattr("src.generator.load_dotenv", lambda *args, **kwargs: False)
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
    for item in items:
        assert isinstance(item.get("expected_chunks"), list)
        if item["answerable"]:
            assert len(item["expected_chunks"]) >= 1
        else:
            assert item["expected_chunks"] == []


def test_generate_all_plots_writes_three_pngs(tmp_path: Path) -> None:
    synthetic_report = {
        "metadata": {
            "top_k": 5,
            "embedding_model": "intfloat/multilingual-e5-base",
            "llm_provider": "gemini",
            "llm_model": "gemini-2.5-flash",
            "judge_mode": "auto",
        },
        "summary": {
            "overall": {
                "answerable_count": 3,
                "recall_at_k": 0.85,
                "mrr": 0.9,
            },
            "by_category": {
                "direct_factual": {
                    "recall_at_k": 1.0,
                    "mrr": 1.0,
                    "faithfulness": 0.95,
                    "answer_accuracy": 0.9,
                },
                "multi_hop": {
                    "recall_at_k": 0.75,
                    "mrr": 0.8,
                    "faithfulness": 0.85,
                    "answer_accuracy": 0.8,
                },
                "tabular_numerical": {
                    "recall_at_k": 0.8,
                    "mrr": 0.9,
                    "faithfulness": 0.9,
                    "answer_accuracy": 0.85,
                },
                "unanswerable": {
                    "faithfulness": 1.0,
                    "answer_accuracy": 1.0,
                    "abstention_accuracy": 1.0,
                },
            },
        },
        "results": [
            {"id": "q01", "answerable": True, "top_similarity": 0.89},
            {"id": "q14", "answerable": True, "top_similarity": 0.83},
            {"id": "q19", "answerable": False, "top_similarity": 0.85},
            {"id": "q24", "answerable": False, "top_similarity": 0.78},
        ],
    }
    results_file = tmp_path / "eval_results.json"
    results_file.write_text(json.dumps(synthetic_report), encoding="utf-8")

    out_dir = tmp_path / "plots"
    created = generate_all_plots(results_path=results_file, output_dir=out_dir)

    assert len(created) == 3
    for png_path in created:
        assert png_path.exists()
        assert png_path.suffix == ".png"
        assert png_path.stat().st_size > 1000


def test_runtime_api_key_and_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.generator.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GROQ_API_KEY", "env-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Runtime api_key and model_name override env vars for Gemini and Groq.
    runtime_gemini = create_llm_backend(
        provider="gemini",
        api_key="ui-gemini-key",
        model_name="custom-gemini-model",
    )
    assert isinstance(runtime_gemini, GeminiBackend)
    assert runtime_gemini._api_key == "ui-gemini-key"
    assert runtime_gemini.model_name == "custom-gemini-model"

    runtime_groq = create_llm_backend(
        provider="groq",
        api_key="ui-groq-key",
        model_name="custom-groq-model",
    )
    assert isinstance(runtime_groq, GroqBackend)
    assert runtime_groq._api_key == "ui-groq-key"
    assert runtime_groq.model_name == "custom-groq-model"

    # Empty/whitespace runtime inputs cleanly fall back to environment variables.
    env_gemini = create_llm_backend(provider="gemini", api_key="   ", model_name=" ")
    assert isinstance(env_gemini, GeminiBackend)
    assert env_gemini._api_key == "env-gemini-key"
    assert env_gemini.model_name == "gemini-2.5-flash"

    # RAGGenerator passes explicit api_key / model_name through and stays backward-compatible.
    gen_gemini = RAGGenerator(
        provider="gemini",
        api_key="ui-session-gemini",
        model_name="gemini-2.5-flash-runtime",
    )
    gemini_backend = gen_gemini._get_backend()
    assert isinstance(gemini_backend, GeminiBackend)
    assert gemini_backend._api_key == "ui-session-gemini"
    assert gemini_backend.model_name == "gemini-2.5-flash-runtime"

    gen_runtime = RAGGenerator(
        provider="groq",
        api_key="ui-session-groq",
        model_name="mixtral-8x7b",
    )
    backend = gen_runtime._get_backend()
    assert isinstance(backend, GroqBackend)
    assert backend._api_key == "ui-session-groq"
    assert backend.model_name == "mixtral-8x7b"

    # Whitespace-only model_name falls back cleanly to legacy `model` argument.
    gen_whitespace_fallback = RAGGenerator(
        provider="gemini",
        model="gemini-legacy-fallback",
        model_name="   ",
    )
    assert gen_whitespace_fallback._get_backend().model_name == "gemini-legacy-fallback"

    gen_legacy = RAGGenerator(provider="gemini", model="gemini-legacy-arg")
    legacy_backend = gen_legacy._get_backend()
    assert isinstance(legacy_backend, GeminiBackend)
    assert legacy_backend._api_key == "env-gemini-key"
    assert legacy_backend.model_name == "gemini-legacy-arg"

    from app import _build_generator

    app_gen = _build_generator(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key="ui-key",
    )
    app_backend = app_gen._get_backend()
    assert isinstance(app_backend, GeminiBackend)
    assert app_backend._api_key == "ui-key"
    assert app_backend.model_name == "gemini-2.5-flash"


def test_colloquial_and_level_query_normalization() -> None:
    q_colloquial = "ايه مواد سنه تالته؟"
    expanded = normalize_and_expand_query(q_colloquial)
    assert expanded.startswith(q_colloquial)
    assert "المستوى الثالث" in expanded
    assert "الفرقة الثالثة" in expanded
    assert "المقررات الدراسية" in expanded

    q_digit = normalize_and_expand_query("مقررات سنه 3 في اللائحة")
    assert "المستوى الثالث" in q_digit
    assert "الفرقة الثالثة" in q_digit

    q_prereq = normalize_and_expand_query("هل يوجد سمر كورس وما متطلب بنل 404؟")
    assert "الفصل الدراسي الصيفي" in q_prereq
    assert "المتطلب السابق" in q_prereq
    assert "ROB 404" in q_prereq

    q_project = normalize_and_expand_query("توزيع درجات COM 400")
    assert "مقرر المشروع" in q_project

    formal = "ما هي شروط التخرج؟"
    assert normalize_and_expand_query(formal) == formal


def test_retriever_diversity_and_fetch_k(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "chroma_diversity"
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name="test_diversity",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=["p010::c001", "p010::c002", "p010::c003", "p009::c001", "p012::c001"],
        documents=[
            "CGPA chunk 1 page 10",
            "CGPA chunk 2 page 10",
            "CGPA chunk 3 page 10",
            "Warning article 17 page 9",
            "Course repeat rules page 12",
        ],
        metadatas=[
            {"source_file": "bylaws.pdf", "page_number": 10, "chunk_id": "p010::c001"},
            {"source_file": "bylaws.pdf", "page_number": 10, "chunk_id": "p010::c002"},
            {"source_file": "bylaws.pdf", "page_number": 10, "chunk_id": "p010::c003"},
            {"source_file": "bylaws.pdf", "page_number": 9, "chunk_id": "p009::c001"},
            {"source_file": "bylaws.pdf", "page_number": 12, "chunk_id": "p012::c001"},
        ],
        embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.2, 0.0, 0.0],
            [0.95, 0.3, 0.0, 0.0],
            [0.9, 0.4, 0.0, 0.0],
            [0.85, 0.5, 0.0, 0.0],
        ],
    )

    seen_queries: list[str] = []

    def fake_embed_fn(text: str) -> list[float]:
        seen_queries.append(text)
        return [1.0, 0.0, 0.0, 0.0]

    retriever = ChromaRetriever(
        chroma_dir=chroma_dir,
        collection_name="test_diversity",
        embed_fn=fake_embed_fn,
    )

    diverse = retriever.retrieve("ايه مواد سنه 3؟", top_k=3, fetch_k=5, max_per_page=2)
    assert [chunk.chunk_id for chunk in diverse] == ["p010::c001", "p010::c002", "p009::c001"]
    assert "المستوى الثالث" in seen_queries[-1]

    # Backfill kicks in cleanly when top_k exceeds max_per_page * distinct_pages.
    backfilled = retriever.retrieve(
        "ما قواعد CGPA؟",
        top_k=4,
        fetch_k=5,
        max_per_page=1,
        expand_query=False,
    )
    assert seen_queries[-1] == "ما قواعد CGPA؟"
    assert len(backfilled) == 4
    assert backfilled[:3] == [
        backfilled[0],
        backfilled[1],
        backfilled[2],
    ]
    assert {chunk.page_number for chunk in backfilled[:3]} == {10, 9, 12}


def test_chunk_metadata_enrichment_for_tables_and_courses() -> None:
    pages = [
        PageDocument(
            source_file="bylaws.pdf",
            page_number=16,
            text=(
                "مادة (26)\n"
                "المستوى الثالث - جدول المقررات:\n"
                "ROB 303 | نظري 2 | عملي 2 | ساعات معتمدة 3 | المتطلب السابق COM 201"
            ),
        ),
        PageDocument(
            source_file="bylaws.pdf",
            page_number=9,
            text="مادة (18): توزع درجات مقرر المشروع بنسبة 50% أعمال فصلية و50% مناقشة.",
        ),
        PageDocument(
            source_file="bylaws.pdf",
            page_number=5,
            text="يشترط للتخرج استيفاء شروط اللائحة الداخلية.",
        ),
    ]

    chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 3

    table_meta = chunks[0].metadata
    assert "ROB 303" in str(table_meta["course_codes"])
    assert "COM 201" in str(table_meta["course_codes"])
    assert table_meta["articles"] == "26"
    assert table_meta["academic_level"] == "Level 3"
    assert table_meta["chunk_type"] == "table_or_curriculum"

    proj_meta = chunks[1].metadata
    assert proj_meta["articles"] == "18"
    assert "COM 400" in str(proj_meta["course_codes"])
    assert "COM 498" in str(proj_meta["course_codes"])
    assert proj_meta["chunk_type"] == "table_or_curriculum"

    # Plain narrative keeps only base keys and all cue values remain scalars.
    assert set(chunks[2].metadata.keys()) == {"source_file", "page_number", "chunk_id"}
    assert extract_chunk_metadata_cues(chunks[2].text) == {}
    for chunk in chunks:
        for val in chunk.metadata.values():
            assert isinstance(val, (str, int))
