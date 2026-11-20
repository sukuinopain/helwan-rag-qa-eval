"""Grounded answer generation with source citations and explicit abstention."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv

from src.retriever import ChromaRetriever, RetrievedChunk

load_dotenv()

ABSTENTION_PHRASE_AR = "لا تتوفر معلومات كافية في اللوائح المسترجعة للإجابة على هذا السؤال."
ABSTENTION_PHRASE_EN = (
    "Insufficient information in the retrieved regulations to answer this question."
)

_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_EASTERN_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_PAGE_CITE_GROUP_RE = re.compile(
    r"(?:صفحة|صفحات|ص\.?|pages?|pp?\.?)\s*[:\-]?\s*([0-9]+(?:\s*[,،&و\-–]\s*[0-9]+)*)",
    re.IGNORECASE,
)

_ABSTENTION_SIGNALS = (
    "لا تتوفر معلومات كافية",
    "لا توجد معلومات كافية",
    "غير مذكور في اللوائح",
    "لم يرد في السياق",
    "لا يمكن الإجابة من السياق",
    "insufficient information",
    "not mentioned in the provided",
    "not available in the retrieved",
    "cannot answer based on the provided",
    "does not contain information",
)


class ConfigurationError(RuntimeError):
    """Raised when required LLM provider credentials or configuration are missing."""


class LLMGenerationError(RuntimeError):
    """Raised when the configured LLM provider fails to produce a valid response."""


@dataclass(frozen=True)
class Citation:
    """Verified page citation linked back to a retrieved chunk."""

    source_file: str
    page_number: int
    chunk_id: str


@dataclass(frozen=True)
class GenerationResult:
    """Grounded generation output including verified citations and abstention status."""

    answer: str
    citations: list[Citation]
    abstained: bool
    provider: str
    model: str
    retrieved_chunks: list[RetrievedChunk]


class LLMBackend(Protocol):
    """Minimal chat/generation interface implemented by LLM providers."""

    provider_name: str
    model_name: str

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Return model text generated from system and user prompts."""


class GeminiBackend:
    """Google Gemini backend using the `google-genai` SDK and `GEMINI_API_KEY`."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        resolved_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        if not resolved_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Add it to your environment or .env file."
            )
        self._api_key = resolved_key
        self.model_name = (
            model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        ).strip()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Generate text with Gemini API."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                ),
            )
            text = (response.text or "").strip()
        except Exception as exc:
            raise LLMGenerationError(f"Gemini generation failed: {exc}") from exc

        if not text:
            raise LLMGenerationError("Gemini returned an empty response.")
        return text


class GroqBackend:
    """Groq backend using the `groq` SDK and `GROQ_API_KEY`."""

    provider_name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        resolved_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        if not resolved_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file."
            )
        self._api_key = resolved_key
        self.model_name = (
            model_name or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
        ).strip()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Generate text with Groq Chat Completions API."""
        try:
            from groq import Groq

            client = Groq(api_key=self._api_key)
            completion = client.chat.completions.create(
                model=self.model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            message = completion.choices[0].message.content if completion.choices else ""
            text = (message or "").strip()
        except Exception as exc:
            raise LLMGenerationError(f"Groq generation failed: {exc}") from exc

        if not text:
            raise LLMGenerationError("Groq returned an empty response.")
        return text


def create_llm_backend(
    provider: str | None = None,
    model: str | None = None,
) -> LLMBackend:
    """Instantiate an LLM backend from explicit arguments or environment variables."""
    load_dotenv()
    selected = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()

    if not selected or selected == "auto":
        if (os.getenv("GEMINI_API_KEY") or "").strip():
            selected = "gemini"
        elif (os.getenv("GROQ_API_KEY") or "").strip():
            selected = "groq"
        else:
            raise ConfigurationError(
                "No LLM API key found. Set GEMINI_API_KEY or GROQ_API_KEY in your .env "
                "file or environment variables."
            )

    if selected == "gemini":
        return GeminiBackend(model_name=model)
    if selected == "groq":
        return GroqBackend(model_name=model)
    raise ConfigurationError(
        f"Unsupported LLM provider '{selected}'. Supported providers: 'gemini', 'groq'."
    )


def is_arabic_text(text: str) -> bool:
    """Return True when the text contains Arabic script characters."""
    return bool(_ARABIC_CHAR_RE.search(text))


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks with explicit source, page, and chunk metadata."""
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        header = (
            f"[{idx}] المصدر: {chunk.source_file} | صفحة: {chunk.page_number} "
            f"| chunk_id: {chunk.chunk_id} | similarity: {chunk.similarity:.4f}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_rag_prompts(
    query: str,
    chunks: Sequence[RetrievedChunk],
) -> tuple[str, str]:
    """Construct strict grounding system and user prompts for regulatory QA."""
    arabic_query = is_arabic_text(query)
    abstention_phrase = ABSTENTION_PHRASE_AR if arabic_query else ABSTENTION_PHRASE_EN
    citation_example = "[صفحة X]" if arabic_query else "[Page X]"

    system_prompt = (
        "أنت مساعد أكاديمي دقيق متخصص في لوائح جامعة حلوان الأهلية.\n"
        "You are a strict regulatory QA assistant for Helwan National University.\n\n"
        "RULES (STRICTLY ENFORCED):\n"
        "1. Rely ONLY on the retrieved excerpts below. Do NOT use external knowledge, guess, "
        "or extrapolate numbers, fees, dates, or names not explicitly stated in the excerpts.\n"
        "2. Answer in the same language as the user's question (Arabic for Arabic questions, "
        "English for English questions).\n"
        f"3. Cite exact PDF page numbers inline after every claim using {citation_example} "
        "(using the `صفحة` number shown in each excerpt header).\n"
        "4. Preserve exact numbers, percentages, course codes, and table values verbatim.\n"
        "5. If the retrieved excerpts do NOT contain enough information to answer the question "
        "completely and reliably, you MUST abstain and output ONLY this exact sentence:\n"
        f"{abstention_phrase}"
    )

    context_block = format_context(chunks)
    user_prompt = (
        f"السؤال / Question:\n{query.strip()}\n\n"
        f"المقتطفات المسترجعة / Retrieved Excerpts:\n{context_block}\n\n"
        "الإجابة الموثقة بالمصادر / Grounded Answer:"
    )
    return system_prompt, user_prompt


def detect_abstention(answer: str) -> bool:
    """Return True if the generated answer explicitly abstains from answering."""
    normalized = re.sub(r"\s+", " ", answer).strip().lower()
    if not normalized:
        return True
    return any(signal in normalized for signal in _ABSTENTION_SIGNALS)


def extract_cited_pages(answer: str) -> list[int]:
    """Extract deduplicated page numbers cited inline in Arabic or English."""
    normalized = answer.translate(_EASTERN_ARABIC_DIGITS)
    pages: list[int] = []
    seen: set[int] = set()

    for match in _PAGE_CITE_GROUP_RE.finditer(normalized):
        group_text = match.group(1)
        for num_str in re.findall(r"\d+", group_text):
            page_num = int(num_str)
            if page_num > 0 and page_num not in seen:
                seen.add(page_num)
                pages.append(page_num)

    return pages


def _map_citations_to_chunks(
    cited_pages: Sequence[int],
    chunks: Sequence[RetrievedChunk],
) -> list[Citation]:
    """Match extracted page numbers against retrieved chunks to prevent fabricated pages."""
    page_lookup: dict[int, RetrievedChunk] = {}
    for chunk in chunks:
        if chunk.page_number not in page_lookup:
            page_lookup[chunk.page_number] = chunk

    citations: list[Citation] = []
    for page_num in cited_pages:
        matched = page_lookup.get(page_num)
        if matched is not None:
            citations.append(
                Citation(
                    source_file=matched.source_file,
                    page_number=matched.page_number,
                    chunk_id=matched.chunk_id,
                )
            )
    return citations


class RAGGenerator:
    """Grounded answer generator with citation extraction and empty-context abstention."""

    def __init__(
        self,
        backend: LLMBackend | None = None,
        provider: str | None = None,
        model: str | None = None,
        min_similarity: float | None = None,
    ) -> None:
        self._backend = backend
        self._provider = provider
        self._model = model
        self.min_similarity = min_similarity

    def _get_backend(self) -> LLMBackend:
        if self._backend is None:
            self._backend = create_llm_backend(provider=self._provider, model=self._model)
        return self._backend

    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate a grounded answer with page citations, or abstain when unsupported."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        filtered = [
            chunk
            for chunk in chunks
            if self.min_similarity is None or chunk.similarity >= self.min_similarity
        ]

        if not filtered:
            abstention = ABSTENTION_PHRASE_AR if is_arabic_text(query) else ABSTENTION_PHRASE_EN
            provider_name = (
                self._backend.provider_name if self._backend else (self._provider or "none")
            )
            model_name = self._backend.model_name if self._backend else (self._model or "none")
            return GenerationResult(
                answer=abstention,
                citations=[],
                abstained=True,
                provider=provider_name,
                model=model_name,
                retrieved_chunks=list(chunks),
            )

        backend = self._get_backend()
        system_prompt, user_prompt = build_rag_prompts(query=query, chunks=filtered)
        raw_answer = backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        ).strip()

        abstained = detect_abstention(raw_answer)
        citations = (
            []
            if abstained
            else _map_citations_to_chunks(extract_cited_pages(raw_answer), filtered)
        )

        return GenerationResult(
            answer=raw_answer,
            citations=citations,
            abstained=abstained,
            provider=backend.provider_name,
            model=backend.model_name,
            retrieved_chunks=list(filtered),
        )


def answer_question(
    query: str,
    retriever: ChromaRetriever,
    generator: RAGGenerator,
    top_k: int = 5,
) -> GenerationResult:
    """Retrieve top-k chunks and generate a grounded answer."""
    chunks = retriever.retrieve(query=query, top_k=top_k)
    return generator.generate(query=query, chunks=chunks)
