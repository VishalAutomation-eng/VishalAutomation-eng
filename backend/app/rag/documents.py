"""Document DTOs for retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedDocument:
    """Single retrieved chunk from a vector database."""

    text: str
    score: float
    source_db: str
    metadata: dict[str, Any] = field(default_factory=dict)
