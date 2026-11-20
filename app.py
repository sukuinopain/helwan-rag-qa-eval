"""Streamlit inspection UI for Helwan National University RAG QA & failure analysis."""

from __future__ import annotations

import html
import importlib
import inspect
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.generator as generator_module  # noqa: E402
from src import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL  # noqa: E402
from src.retriever import (  # noqa: E402
    ChromaRetriever,
    RetrievedChunk,
    get_embedding_model,
    normalize_and_expand_query,
)

SAMPLE_QUERIES: list[tuple[str, str]] = [
    (
        "Direct Factual (AR)",
        "ما هي نسبة الحضور المطلوبة لدخول الامتحان النهائي وما عقوبة تجاوز الغياب؟",
    ),
    (
        "Tabular Numerical (AR)",
        "ما هي عدد الساعات المعتمدة المطلوبة للتخرج وكيف تتوزع على متطلبات البرنامج؟",
    ),
    (
        "Multi-hop (AR)",
        "طالب معدله التراكمي 1.8، ما الحد الأقصى للساعات المسجلة له "
        "وما قواعد إعادة المقررات لرفع معدله؟",
    ),
    ("Course Details (EN)", "What topics are covered in ROB 404 Mobile robots?"),
    (
        "Unanswerable (AR)",
        "ما قيمة المصروفات الدراسية السنوية للالتحاق ببرنامج برمجيات الإنسان الآلي؟",
    ),
]


@st.cache_resource(show_spinner="Loading ChromaDB index and multilingual-e5-base...")
def load_retriever(
    chroma_dir: str,
    collection_name: str,
    model_name: str,
) -> ChromaRetriever:
    """Load and cache the ChromaRetriever and embedding model across Streamlit reruns."""
    get_embedding_model(model_name)
    return ChromaRetriever(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        model_name=model_name,
    )


def _format_bidi_html(text: str) -> str:
    escaped = html.escape(text).replace("\n", "<br>")
    return f'<div class="rtl-auto">{escaped}</div>'


_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash"),
    "groq": ("GROQ_API_KEY", "GROQ_MODEL", "llama-3.3-70b-versatile"),
}


def _set_sample_query(sample_text: str) -> None:
    st.session_state["query_input"] = sample_text


def _clear_session_state_key(state_key: str) -> None:
    st.session_state[state_key] = ""


def _default_provider_index() -> int:
    configured = os.getenv("LLM_PROVIDER", "").strip().lower()
    if configured == "groq":
        return 1
    if configured == "gemini":
        return 0
    if os.getenv("GROQ_API_KEY", "").strip() and not os.getenv("GEMINI_API_KEY", "").strip():
        return 1
    return 0


def _build_generator(
    provider: str,
    model_name: str | None,
    api_key: str | None,
) -> generator_module.RAGGenerator:
    """Instantiate RAGGenerator, reloading src.generator if Streamlit cached a stale module."""
    params = inspect.signature(generator_module.RAGGenerator.__init__).parameters
    if "model_name" not in params or "api_key" not in params:
        importlib.reload(generator_module)
    return generator_module.RAGGenerator(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
    )


def _render_chunks(chunks: list[RetrievedChunk]) -> None:
    st.subheader("Retrieved Chunks & Similarity Scores")
    if not chunks:
        st.info("No chunks were retrieved for this query.")
        return

    top_sim = max(chunk.similarity for chunk in chunks)
    unique_pages = sorted({chunk.page_number for chunk in chunks})
    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks Retrieved", len(chunks))
    m2.metric("Top Similarity", f"{top_sim:.4f}")
    m3.metric("Unique PDF Pages", ", ".join(str(p) for p in unique_pages))

    for rank, chunk in enumerate(chunks, start=1):
        title = (
            f"#{rank} | Page {chunk.page_number} | "
            f"Similarity: {chunk.similarity:.4f} | Distance: {chunk.distance:.4f} | "
            f"{chunk.chunk_id}"
        )
        with st.expander(title, expanded=(rank <= 2)):
            st.progress(max(0.0, min(1.0, chunk.similarity)))
            meta_parts = [
                f"Source: `{chunk.source_file}`",
                f"Page: `{chunk.page_number}`",
                f"Chunk ID: `{chunk.chunk_id}`",
            ]
            for label, key in (
                ("Level", "academic_level"),
                ("Courses", "course_codes"),
                ("Articles", "articles"),
                ("Type", "chunk_type"),
            ):
                value = chunk.metadata.get(key)
                if value:
                    meta_parts.append(f"{label}: `{value}`")
            st.caption(" | ".join(meta_parts))
            st.markdown(_format_bidi_html(chunk.text), unsafe_allow_html=True)


def main() -> None:
    """Render the Streamlit RAG QA and retrieval inspection application."""
    load_dotenv()
    st.set_page_config(
        page_title="Helwan National University RAG QA",
        page_icon="🎓",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .rtl-auto {
            direction: auto;
            unicode-bidi: plaintext;
            line-height: 1.65;
            font-size: 1.0rem;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            background-color: rgba(128, 128, 128, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Helwan National University — Regulations RAG QA & Inspector")
    st.caption(
        "Inspect grounded answers, source/page citations, and vector retrieval scores "
        "over the Robotics Software Program Bylaws (Arabic + English)."
    )

    with st.sidebar:
        st.header("Configuration")
        chroma_dir = st.text_input("ChromaDB Directory", value=str(DEFAULT_CHROMA_DIR))
        collection_name = st.text_input("Collection Name", value=DEFAULT_COLLECTION_NAME)
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=15, value=8)

        st.divider()
        retrieval_only = st.checkbox(
            "Retrieval-only mode (inspect chunks without LLM)",
            value=False,
        )
        provider_choice = st.selectbox(
            "LLM Provider",
            options=["gemini", "groq"],
            index=_default_provider_index(),
            key="llm_provider",
            disabled=retrieval_only,
        )
        env_key_var, env_model_var, default_model = _PROVIDER_DEFAULTS[provider_choice]
        default_model_hint = os.getenv(env_model_var, "").strip() or default_model
        model_override = st.text_input(
            "Model Name (optional)",
            key=f"llm_model_{provider_choice}",
            placeholder=default_model_hint,
            help=(
                f"Leave blank to use {env_model_var} or default '{default_model}'."
            ),
            disabled=retrieval_only,
        )
        api_key_state_key = f"api_key_{provider_choice}"
        runtime_api_key = st.text_input(
            f"{provider_choice.capitalize()} API Key (session only)",
            type="password",
            key=api_key_state_key,
            placeholder=f"Paste your {provider_choice.capitalize()} API key...",
            help=(
                "Stored only in memory (st.session_state) for this active session; "
                "never written to disk or .env."
            ),
            disabled=retrieval_only,
        )
        has_runtime_key = bool(runtime_api_key.strip())
        has_env_key = bool(os.getenv(env_key_var, "").strip())

        if not retrieval_only:
            if has_runtime_key:
                st.success(
                    f"Using your session `{provider_choice}` API key (memory-only)."
                )
                st.button(
                    "Clear session key",
                    on_click=_clear_session_state_key,
                    args=(api_key_state_key,),
                    use_container_width=True,
                )
            elif has_env_key:
                st.info(
                    f"Using `{env_key_var}` from environment. You can paste your own "
                    "key above to override it for this session."
                )
            else:
                st.info(
                    f"No `{provider_choice}` API key entered yet. Paste your own API key "
                    "above to enable live grounded answers — retrieved chunks, scores, "
                    "and PDF pages still work without a key."
                )

    try:
        retriever = load_retriever(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            model_name=DEFAULT_EMBEDDING_MODEL,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(
            f"{exc}\n\nRun ingestion first:\n```bash\npython -m src.ingestion\n```"
        )
        return

    with st.sidebar:
        st.divider()
        st.subheader("Index Status")
        st.write(f"**Collection:** `{retriever.collection_name}`")
        st.write(f"**Indexed Chunks:** `{retriever.count()}`")
        st.write(f"**Embedding Model:** `{retriever.model_name}`")

    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""

    st.markdown("**Sample Queries:**")
    btn_cols = st.columns(len(SAMPLE_QUERIES))
    for col, (label, sample_text) in zip(btn_cols, SAMPLE_QUERIES, strict=True):
        col.button(
            label,
            on_click=_set_sample_query,
            args=(sample_text,),
            use_container_width=True,
        )

    query = st.text_area(
        "Enter your question (Arabic or English):",
        key="query_input",
        height=90,
        placeholder="اكتب سؤالك حول اللوائح الدراسية هنا أو بالإنجليزية...",
    )
    run_clicked = st.button("Ask / Retrieve", type="primary")

    if not run_clicked:
        return

    if not query.strip():
        st.warning("Please enter a non-empty question.")
        return

    cleaned_query = query.strip()
    expanded_query = normalize_and_expand_query(cleaned_query)
    if expanded_query != cleaned_query:
        st.caption(f"Expanded retrieval query: `{expanded_query}`")

    with st.spinner("Retrieving relevant regulation chunks..."):
        chunks = retriever.retrieve(query=cleaned_query, top_k=top_k)

    if not retrieval_only:
        st.subheader("Generated Answer & Citations")
        cleaned_runtime_key = runtime_api_key.strip() or None
        cleaned_model = model_override.strip() or None

        if cleaned_runtime_key is None and not has_env_key:
            st.info(
                f"No `{provider_choice}` API key provided yet. Paste your own API key "
                "in the sidebar to generate live grounded answers. Retrieved chunks, "
                "similarity scores, and PDF page references are shown below."
            )
        else:
            try:
                generator = _build_generator(
                    provider=provider_choice,
                    model_name=cleaned_model,
                    api_key=cleaned_runtime_key,
                )
                with st.spinner("Generating grounded answer..."):
                    result = generator.generate(query=query.strip(), chunks=chunks)

                if result.abstained:
                    st.warning(
                        f"Status: **Abstained** | Provider: `{result.provider}` "
                        f"(`{result.model}`)"
                    )
                else:
                    st.success(
                        f"Status: **Grounded Answer** | Provider: `{result.provider}` "
                        f"(`{result.model}`)"
                    )

                st.markdown(_format_bidi_html(result.answer), unsafe_allow_html=True)

                st.markdown("**Cited Pages / Sources:**")
                if result.citations:
                    citation_rows = [
                        {
                            "Source File": cite.source_file,
                            "PDF Page": cite.page_number,
                            "Chunk ID": cite.chunk_id,
                        }
                        for cite in result.citations
                    ]
                    st.dataframe(citation_rows, use_container_width=True, hide_index=True)
                elif result.abstained:
                    st.info("No citations attached because the model abstained.")
                else:
                    st.info(
                        "No explicit inline page citations (`[صفحة X]` / `[Page X]`) were parsed "
                        "from the response. Inspect the retrieved chunks below."
                    )
            except (
                generator_module.ConfigurationError,
                generator_module.LLMGenerationError,
            ) as exc:
                st.warning(
                    f"LLM generation skipped/failed: {exc}\n\n"
                    "Retrieved chunks are still displayed below for inspection."
                )

    st.divider()
    _render_chunks(chunks)


if __name__ == "__main__":
    main()
