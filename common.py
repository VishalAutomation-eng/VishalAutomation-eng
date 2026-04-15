"""Shared helpers for PDF extraction, semantic chunking, and embedding retrieval."""

from __future__ import annotations

import importlib
import io
import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


class PDFExtractionError(RuntimeError):
    """Raised when all PDF extraction backends fail."""


class EmbeddingError(RuntimeError):
    """Raised when embedding API call fails."""


@dataclass
class RetrievedDoc:
    """Container for retrieval result snippets."""

    content: str
    score: float
    metadata: Dict


PDF_BACKEND_MODULES = {
    "PyMuPDF": "fitz",
    "pypdf": "pypdf",
    "pdfminer.six": "pdfminer.high_level",
}


def load_optional_module(module_name: str):
    """Load a module by name and return it if installed, otherwise None."""
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None
    if spec is None:
        return None

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def require_module(module_name: str):
    """Load required module or raise a clear RuntimeError."""
    module = load_optional_module(module_name)
    if module is None:
        raise RuntimeError(f"Required dependency is not installed: {module_name}")
    return module


def get_pdf_backend_status() -> Dict[str, bool]:
    """Return installed/not-installed status for supported PDF extractors."""
    return {name: load_optional_module(module) is not None for name, module in PDF_BACKEND_MODULES.items()}


def format_pdf_backend_help() -> str:
    """Return a practical install message for missing PDF dependencies."""
    return (
        "Install PDF dependencies in the same Python environment used by Streamlit:\n"
        "1) python -m pip install -r requirements.txt\n"
        "2) python -m pip install PyMuPDF pypdf pdfminer.six\n"
        "3) Restart Streamlit (Ctrl+C, then run streamlit again)."
    )


def preprocess_text(text: str) -> str:
    """Normalize whitespace and remove null bytes."""
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like segments."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _l2norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def cosine_similarity_pair(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    d = sum(a * b for a, b in zip(v1, v2))
    n = _l2norm(v1) * _l2norm(v2)
    return d / n if n else 0.0


_LOCAL_EMBEDDER = None


def _get_local_embedder(model_name: str):
    """Load and cache a local open-source sentence-transformers model."""
    global _LOCAL_EMBEDDER
    if _LOCAL_EMBEDDER is not None:
        return _LOCAL_EMBEDDER

    st_module = require_module("sentence_transformers")
    _LOCAL_EMBEDDER = st_module.SentenceTransformer(model_name)
    return _LOCAL_EMBEDDER


def _embed_with_local_model(texts: Sequence[str], model_name: str) -> List[List[float]]:
    """Embed text with local open-source model."""
    model = _get_local_embedder(model_name)
    vectors = model.encode(list(texts), normalize_embeddings=True)
    return [list(map(float, row)) for row in vectors]


def _embed_with_remote_api(texts: Sequence[str], api_base: str, model: str) -> List[List[float]]:
    """Embed text with OpenAI-compatible remote embedding endpoint."""
    requests = require_module("requests")
    payload = {"input": list(texts), "model": model}
    endpoints = [f"{api_base.rstrip('/')}/v1/embeddings", f"{api_base.rstrip('/')}/embeddings"]
    errors: List[str] = []

    for endpoint in endpoints:
        try:
            response = requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            vectors = [row["embedding"] for row in data["data"]]
            if len(vectors) != len(texts):
                raise EmbeddingError("Embedding response length mismatch.")
            return vectors
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{endpoint}: {exc}")

    raise EmbeddingError("Embedding API failed. " + " | ".join(errors))


def call_embedding_api(
    texts: Sequence[str],
    api_base: str = EMBEDDING_API_BASE,
    model: str = EMBEDDING_MODEL,
) -> List[List[float]]:
    """Get embeddings using local OSS model by default, remote API optionally."""
    if not texts:
        return []

    # Default and recommended: local open-source model.
    if not api_base:
        return _embed_with_local_model(texts, model_name=model)

    # Optional: try remote endpoint first; fallback to local model if remote fails.
    try:
        return _embed_with_remote_api(texts, api_base=api_base, model=model)
    except EmbeddingError as remote_error:
        LOGGER.warning("Remote embedding failed, falling back to local model: %s", remote_error)
        return _embed_with_local_model(texts, model_name=model)


def semantic_chunk_text(
    text: str,
    max_sentences_per_chunk: int = 5,
    similarity_threshold: float = 0.72,
    embedding_api_base: str = EMBEDDING_API_BASE,
    embedding_model: str = EMBEDDING_MODEL,
) -> List[str]:
    """Create semantic chunks using sentence embeddings and similarity continuity."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    sent_embeddings = call_embedding_api(sentences, api_base=embedding_api_base, model=embedding_model)
    chunks: List[List[str]] = [[sentences[0]]]

    for idx in range(1, len(sentences)):
        prev_vec = sent_embeddings[idx - 1]
        cur_vec = sent_embeddings[idx]
        sim = cosine_similarity_pair(prev_vec, cur_vec)

        if sim >= similarity_threshold and len(chunks[-1]) < max_sentences_per_chunk:
            chunks[-1].append(sentences[idx])
        else:
            chunks.append([sentences[idx]])

    return [" ".join(chunk) for chunk in chunks]


def extract_with_pymupdf(pdf_bytes: bytes) -> List[str]:
    fitz = load_optional_module("fitz")
    if fitz is None:
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [(page.get_text("text") or "") for page in doc]
    finally:
        doc.close()


def extract_with_pypdf(pdf_bytes: bytes) -> List[str]:
    pypdf = load_optional_module("pypdf")
    if pypdf is None:
        return []

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return [(pg.extract_text() or "") for pg in reader.pages]


def extract_with_pdfminer(pdf_bytes: bytes) -> List[str]:
    pdfminer_high_level = load_optional_module("pdfminer.high_level")
    if pdfminer_high_level is None:
        return []

    text = pdfminer_high_level.extract_text(io.BytesIO(pdf_bytes))
    return [text or ""]


def extract_pdf_pages(pdf_bytes: bytes) -> List[str]:
    errors: List[str] = []
    extractors = [
        ("PyMuPDF", extract_with_pymupdf),
        ("pypdf", extract_with_pypdf),
        ("pdfminer", extract_with_pdfminer),
    ]

    for backend_name, extractor in extractors:
        try:
            pages = extractor(pdf_bytes)
            cleaned_pages = [preprocess_text(p) for p in pages if p and p.strip()]
            if cleaned_pages:
                LOGGER.info("PDF extraction succeeded with %s (%d chunks)", backend_name, len(cleaned_pages))
                return cleaned_pages
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Extractor %s failed", backend_name)
            errors.append(f"{backend_name} failed: {exc}")

    message = (
        "No usable PDF text extractor found. Install at least one backend: "
        "PyMuPDF, pypdf, or pdfminer.six.\n\n"
        + format_pdf_backend_help()
    )
    if errors:
        message += "\n\nBackend errors:\n- " + "\n- ".join(errors)
    raise PDFExtractionError(message)


class EmbeddingIndex:
    """Simple in-memory embedding index using cosine similarity."""

    def __init__(
        self,
        docs: Sequence[Dict],
        embedding_api_base: str = EMBEDDING_API_BASE,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        if not docs:
            raise ValueError("EmbeddingIndex requires at least one document.")

        self.docs = list(docs)
        self.embedding_api_base = embedding_api_base
        self.embedding_model = embedding_model
        self.doc_vectors = call_embedding_api(
            [d["content"] for d in self.docs],
            api_base=self.embedding_api_base,
            model=self.embedding_model,
        )

    def search(self, query: str, k: int = 4) -> List[RetrievedDoc]:
        """Return top-k documents by embedding cosine similarity."""
        query_vec = call_embedding_api(
            [query], api_base=self.embedding_api_base, model=self.embedding_model
        )[0]

        scored: List[Tuple[int, float]] = []
        for idx, vec in enumerate(self.doc_vectors):
            scored.append((idx, cosine_similarity_pair(query_vec, vec)))

        top_ids = [idx for idx, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]
        score_map = dict(scored)
        return [
            RetrievedDoc(
                content=self.docs[idx]["content"],
                score=float(score_map[idx]),
                metadata=self.docs[idx].get("metadata", {}),
            )
            for idx in top_ids
        ]


class QdrantEmbeddingIndex:
    """Advanced vector database index powered by Qdrant."""

    def __init__(
        self,
        docs: Sequence[Dict],
        embedding_api_base: str = EMBEDDING_API_BASE,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str | None = None,
    ):
        if not docs:
            raise ValueError("QdrantEmbeddingIndex requires at least one document.")

        qdrant_client_mod = require_module("qdrant_client")
        models = require_module("qdrant_client.models")

        self.docs = list(docs)
        self.embedding_api_base = embedding_api_base
        self.embedding_model = embedding_model

        self.doc_vectors = call_embedding_api(
            [d["content"] for d in self.docs],
            api_base=self.embedding_api_base,
            model=self.embedding_model,
        )
        vector_size = len(self.doc_vectors[0])

        self.collection_name = collection_name or f"rag_{uuid.uuid4().hex[:10]}"
        self.client = qdrant_client_mod.QdrantClient(":memory:")

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

        points = [
            models.PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "content": self.docs[idx]["content"],
                    "metadata": self.docs[idx].get("metadata", {}),
                },
            )
            for idx, vector in enumerate(self.doc_vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, k: int = 4) -> List[RetrievedDoc]:
        """Search Qdrant collection using query embeddings."""
        query_vector = call_embedding_api(
            [query], api_base=self.embedding_api_base, model=self.embedding_model
        )[0]

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
        )

        out: List[RetrievedDoc] = []
        for hit in result.points:
            payload = hit.payload or {}
            out.append(
                RetrievedDoc(
                    content=payload.get("content", ""),
                    score=float(hit.score or 0.0),
                    metadata=payload.get("metadata", {}),
                )
            )
        return out


def build_vector_index(
    docs: Sequence[Dict],
    embedding_api_base: str = EMBEDDING_API_BASE,
    embedding_model: str = EMBEDDING_MODEL,
):
    """Build advanced vector DB index (Qdrant) with safe fallback to in-memory index."""
    if load_optional_module("qdrant_client") is not None and load_optional_module("qdrant_client.models") is not None:
        return QdrantEmbeddingIndex(
            docs,
            embedding_api_base=embedding_api_base,
            embedding_model=embedding_model,
        )

    LOGGER.warning("qdrant-client not installed; using in-memory EmbeddingIndex fallback.")
    return EmbeddingIndex(
        docs,
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
    )
