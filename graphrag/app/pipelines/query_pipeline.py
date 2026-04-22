"""Query pipeline for Cypher generation, graph retrieval, and answer generation."""

import asyncio
import logging
from typing import Any, Dict, List

import numpy as np

from app.services.graph_service import GraphService
from app.services.llm_service import LLMService
from app.utils.prompt_templates import ANSWER_GENERATION_PROMPT, CYPHER_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Pipeline to answer questions from a graph-backed knowledge base."""

    def __init__(self, llm_service: LLMService, graph_service: GraphService, embedding_model_name: str) -> None:
        self._llm_service = llm_service
        self._graph_service = graph_service
        self._embedding_model_name = embedding_model_name
        self._embedding_model: Any | None = None

    async def _get_embedding_model(self) -> Any:
        """Lazy-load sentence-transformers model.

        :return: Loaded embedding model.
        """

        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = await asyncio.to_thread(SentenceTransformer, self._embedding_model_name)
        return self._embedding_model

    async def _question_to_cypher(self, question: str) -> str:
        """Translate natural language question to Cypher.

        :param question: User question.
        :return: Cypher query.
        """

        schema_hint = await self._graph_service.graph_schema_hint()
        prompt = CYPHER_GENERATION_PROMPT.format(schema_hint=schema_hint, question=question)
        cypher = await self._llm_service.generate(prompt=prompt, temperature=0.0)
        return cypher.strip().strip("```").replace("cypher", "").strip()

    @staticmethod
    def _build_graph_context(records: List[Dict[str, Any]]) -> str:
        """Convert Cypher rows into a textual context block.

        :param records: Graph records.
        :return: Context text.
        """

        if not records:
            return "No graph facts returned."
        lines = ["Graph query results:"]
        for idx, row in enumerate(records, start=1):
            lines.append(f"{idx}. {row}")
        return "\n".join(lines)

    async def _hybrid_chunk_context(self, question: str, k: int = 5) -> str:
        """Retrieve semantic context from chunks for improved answer robustness.

        :param question: User question.
        :param k: Number of top chunks.
        :return: Context string.
        """

        chunks = await self._graph_service.fetch_chunks(limit=200)
        if not chunks:
            return ""

        model = await self._get_embedding_model()
        q_emb = await asyncio.to_thread(model.encode, [question], normalize_embeddings=True)
        text_emb = await asyncio.to_thread(
            model.encode,
            [c.get("text", "") for c in chunks],
            normalize_embeddings=True,
        )
        scores = np.dot(np.asarray(text_emb), np.asarray(q_emb)[0])
        top_idx = np.argsort(scores)[::-1][:k]

        selected = [chunks[i].get("text", "") for i in top_idx if chunks[i].get("text")]
        if not selected:
            return ""
        return "\n".join(f"Chunk {i + 1}: {txt}" for i, txt in enumerate(selected))

    async def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer user question using graph retrieval and LLM reasoning.

        :param question: User question.
        :return: Dict with answer and intermediate details.
        """

        cypher_query = await self._question_to_cypher(question)
        records = await self._graph_service.run_query(cypher_query)
        graph_context = self._build_graph_context(records)
        hybrid_context = await self._hybrid_chunk_context(question)

        combined_context = graph_context if not hybrid_context else f"{graph_context}\n\nSemantic chunk context:\n{hybrid_context}"
        answer_prompt = ANSWER_GENERATION_PROMPT.format(question=question, context=combined_context)
        answer = await self._llm_service.generate(prompt=answer_prompt, temperature=0.0)

        return {
            "answer": answer.strip(),
            "cypher": cypher_query,
            "records": records,
        }
