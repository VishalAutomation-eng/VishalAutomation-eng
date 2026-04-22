"""Extraction service for entities and relationships."""

import logging
from typing import Any, Dict

from app.services.llm_service import LLMService
from app.utils.prompt_templates import ENTITY_RELATION_PROMPT

logger = logging.getLogger(__name__)


class ExtractionService:
    """Service responsible for structured IE from chunk text."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm_service = llm_service

    async def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities and relations.

        :param text: Chunk text input.
        :return: JSON extraction payload.
        """

        prompt = ENTITY_RELATION_PROMPT.format(text=text)
        data = await self._llm_service.generate_json(prompt)

        entities = data.get("entities", []) if isinstance(data, dict) else []
        relations = data.get("relations", []) if isinstance(data, dict) else []

        if not isinstance(entities, list) or not isinstance(relations, list):
            logger.warning("LLM extraction schema mismatch: %s", data)
            return {"entities": [], "relations": []}

        return {"entities": entities, "relations": relations}
