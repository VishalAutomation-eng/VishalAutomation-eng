"""Graph RAG pipeline with hybrid retrieval (graph signal + embeddings)."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from common import (
    EMBEDDING_API_BASE,
    EMBEDDING_MODEL,
    RetrievedDoc,
    call_embedding_api,
    cosine_similarity_pair,
)


class GraphRAGIndex:
    """Hybrid Graph RAG: embedding similarity + entity-overlap boost."""

    ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

    def __init__(
        self,
        pages: Sequence[str],
        embedding_api_base: str = EMBEDDING_API_BASE,
        embedding_model: str = EMBEDDING_MODEL,
        graph_weight: float = 0.35,
        embedding_weight: float = 0.65,
    ):
        self.embedding_api_base = embedding_api_base
        self.embedding_model = embedding_model
        self.graph_weight = graph_weight
        self.embedding_weight = embedding_weight

        self.sentences: List[Dict] = []
        self.entity_to_sent_ids: Dict[str, List[int]] = {}

        sent_id = 0
        for page_num, page_text in enumerate(pages, start=1):
            raw_sentences = re.split(r"(?<=[.!?])\s+", page_text)
            for sentence in raw_sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue

                entities = sorted(set(self.ENTITY_RE.findall(sentence)))
                self.sentences.append(
                    {
                        "id": sent_id,
                        "content": sentence,
                        "metadata": {"page": page_num, "entities": entities},
                    }
                )
                for entity in entities:
                    self.entity_to_sent_ids.setdefault(entity.lower(), []).append(sent_id)
                sent_id += 1

        if not self.sentences:
            raise ValueError("GraphRAGIndex requires at least one usable sentence.")

        self.sent_vectors = call_embedding_api(
            [s["content"] for s in self.sentences],
            api_base=self.embedding_api_base,
            model=self.embedding_model,
        )

    def search(self, query: str, k: int = 4) -> List[RetrievedDoc]:
        """Search by hybrid score: embedding similarity + graph entity overlap."""
        query_entities = {entity.lower() for entity in self.ENTITY_RE.findall(query)}

        entity_hits: Dict[int, int] = {}
        for query_entity in query_entities:
            for sent_id in self.entity_to_sent_ids.get(query_entity, []):
                entity_hits[sent_id] = entity_hits.get(sent_id, 0) + 1

        query_vec = call_embedding_api(
            [query], api_base=self.embedding_api_base, model=self.embedding_model
        )[0]

        scored: List[Tuple[int, float]] = []
        for sent_id, sent_vec in enumerate(self.sent_vectors):
            emb_score = cosine_similarity_pair(query_vec, sent_vec)
            graph_score = float(entity_hits.get(sent_id, 0))
            hybrid = (self.embedding_weight * emb_score) + (self.graph_weight * graph_score)
            scored.append((sent_id, hybrid))

        top_ids = [sid for sid, _ in sorted(scored, key=lambda row: row[1], reverse=True)[:k]]
        score_map = dict(scored)
        return [
            RetrievedDoc(
                content=self.sentences[sid]["content"],
                score=float(score_map[sid]),
                metadata=self.sentences[sid]["metadata"],
            )
            for sid in top_ids
        ]
