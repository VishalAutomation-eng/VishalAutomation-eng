"""Streamlit entry point for comparing Old/Page/Graph RAG strategies."""

from __future__ import annotations

import json
import os
import sys
from typing import List, Sequence

from common import (
    EMBEDDING_API_BASE,
    EMBEDDING_MODEL,
    LOGGER,
    PDFExtractionError,
    RetrievedDoc,
    extract_pdf_pages,
    format_pdf_backend_help,
    get_pdf_backend_status,
    require_module,
)
from graph_rag import GraphRAGIndex
from old_rag import build_old_rag_index
from page_rag import build_page_rag_index

DEFAULT_API_BASE = os.getenv("VISION_API_BASE", "http://111.118.189.124:8100/v1")
DEFAULT_MODEL = os.getenv("VISION_MODEL", "/home/expadmin/Mayank/Qwen")


class LLMRequestError(RuntimeError):
    """Raised when the LLM API request fails or returns malformed output."""


def build_index(
    rag_type: str,
    pages: Sequence[str],
    embedding_api_base: str,
    embedding_model: str,
):
    """Build retrieval index for selected RAG strategy."""
    if rag_type.startswith("Old"):
        return build_old_rag_index(
            pages,
            embedding_api_base=embedding_api_base,
            embedding_model=embedding_model,
        )
    if rag_type.startswith("Page"):
        return build_page_rag_index(
            pages,
            embedding_api_base=embedding_api_base,
            embedding_model=embedding_model,
        )
    return GraphRAGIndex(
        pages,
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
    )


def run_pipeline(
    pdf_bytes: bytes,
    rag_type: str,
    query: str,
    top_k: int,
    embedding_api_base: str,
    embedding_model: str,
) -> List[RetrievedDoc]:
    """Execute extraction + index creation + retrieval."""
    pages = extract_pdf_pages(pdf_bytes)
    index = build_index(rag_type, pages, embedding_api_base, embedding_model)
    return index.search(query, k=top_k)


def build_prompt(query: str, contexts: Sequence[RetrievedDoc]) -> str:
    """Create a grounded prompt from retrieved contexts."""
    context_block = "\n\n".join([f"[{i + 1}] {doc.content}" for i, doc in enumerate(contexts)])
    return (
        "You are a precise RAG assistant. Use ONLY the context snippets below. "
        "If answer is not in context, say so.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )


def call_qwen(api_base: str, model: str, query: str, contexts: Sequence[RetrievedDoc]) -> str:
    """Call Qwen-compatible chat completions endpoint and return answer text."""
    requests = require_module("requests")
    prompt = build_prompt(query, contexts)
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer clearly with short bullet points."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        LOGGER.exception("LLM API request failed")
        raise LLMRequestError(f"LLM request failed: {exc}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        LOGGER.exception("Malformed LLM API response")
        raw = response.text if "response" in locals() else "<no response>"
        snippet = raw[:800] if isinstance(raw, str) else json.dumps(raw)[:800]
        raise LLMRequestError(f"Malformed LLM response: {exc}. Response snippet: {snippet}") from exc


def main() -> None:
    """Render and run Streamlit UI."""
    st = require_module("streamlit")
    st.set_page_config(page_title="RAG Comparison Demo", page_icon="📄", layout="wide")
    st.title("📄 RAG Comparison Demo (Old vs Page vs Graph)")
    st.caption("Semantic chunking + embedding retrieval + graph/hybrid search by RAG type.")

    with st.sidebar:
        st.header("Model Configuration")
        api_base = st.text_input("VISION_API_BASE", value=DEFAULT_API_BASE)
        model = st.text_input("VISION_MODEL", value=DEFAULT_MODEL)
        embedding_api_base = st.text_input("EMBEDDING_API_BASE", value=EMBEDDING_API_BASE)
        embedding_model = st.text_input("EMBEDDING_MODEL", value=EMBEDDING_MODEL)
        top_k = st.slider("Top-K Retrieval", 1, 8, 4)

        st.markdown("---")
        st.subheader("Runtime Info")
        st.caption(f"Python: {sys.version.split()[0]}")
        st.code(sys.executable)

        st.subheader("PDF Backend Check")
        backend_status = get_pdf_backend_status()
        for backend_name, installed in backend_status.items():
            icon = "✅" if installed else "❌"
            st.write(f"{icon} {backend_name}")
        if not any(backend_status.values()):
            st.warning("No PDF backend installed.")
            st.code(format_pdf_backend_help())

    uploaded_pdf = st.file_uploader("Upload user PDF", type=["pdf"])
    query = st.text_area("Ask a question about the PDF", height=90)
    rag_type = st.radio(
        "Choose RAG Type",
        ["Old RAG (semantic chunking)", "Page RAG (embedding pages)", "Graph RAG (hybrid graph+embedding)"],
    )

    if st.button("Run RAG", type="primary"):
        if not uploaded_pdf:
            st.warning("Please upload a PDF first.")
            return
        if not query.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner("Extracting, chunking/indexing, and retrieving..."):
                contexts = run_pipeline(
                    uploaded_pdf.read(),
                    rag_type,
                    query,
                    top_k,
                    embedding_api_base,
                    embedding_model,
                )
        except (PDFExtractionError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
            st.info("Quick fix: run the install commands below in the same terminal/env where Streamlit runs.")
            st.code(format_pdf_backend_help())
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Unexpected pipeline error")
            st.error(f"Unexpected error during pipeline execution: {exc}")
            return

        left, right = st.columns(2)
        with left:
            st.subheader("Retrieved Context")
            for i, doc in enumerate(contexts, start=1):
                st.markdown(f"**{i}. score={doc.score:.3f} | metadata={doc.metadata}**")
                st.write(doc.content[:1000] + ("..." if len(doc.content) > 1000 else ""))

        with right:
            st.subheader("LLM Answer")
            try:
                answer = call_qwen(api_base, model, query, contexts)
                st.success("Answer generated")
                st.write(answer)
            except LLMRequestError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
