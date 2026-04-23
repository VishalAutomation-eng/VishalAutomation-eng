"""Graph persistence and retrieval service for Neo4j."""

import logging
from typing import Any, Dict, List

from config.settings import settings

logger = logging.getLogger(__name__)


class GraphService:
    """Service for Neo4j graph operations."""

    def __init__(self) -> None:
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def close(self) -> None:
        """Close Neo4j driver."""

        await self._driver.close()

    async def ensure_constraints(self) -> None:
        """Create base constraints/indexes for graph entities."""

        constraints = [
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
        ]
        async with self._driver.session(database=settings.neo4j_database) as session:
            for query in constraints:
                await session.run(query)

    async def upsert_extraction(self, chunk_id: str, chunk_text: str, extraction: Dict[str, Any]) -> Dict[str, int]:
        """Upsert entities and relationships extracted from text.

        :param chunk_id: Unique chunk ID.
        :param chunk_text: Original chunk text.
        :param extraction: Dict with entities and relations.
        :return: Write statistics.
        """

        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        async with self._driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $chunk_text
                """,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
            )

            for entity in entities:
                name = str(entity.get("name", "")).strip()
                entity_type = str(entity.get("type", "Entity")).strip() or "Entity"
                if not name:
                    continue

                safe_label = "".join(ch for ch in entity_type if ch.isalnum()) or "Entity"
                await session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $entity_type
                    """,
                    name=name,
                    entity_type=entity_type,
                )
                await session.run(
                    f"MATCH (e:Entity {{name: $name}}) SET e:{safe_label}",
                    name=name,
                )
                await session.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id}), (e:Entity {name: $name})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    chunk_id=chunk_id,
                    name=name,
                )

            for relation in relations:
                source = str(relation.get("source", "")).strip()
                target = str(relation.get("target", "")).strip()
                rel_type = str(relation.get("type", "RELATED_TO")).strip().upper().replace(" ", "_")
                if not source or not target:
                    continue
                safe_rel_type = "".join(ch for ch in rel_type if ch.isalnum() or ch == "_") or "RELATED_TO"
                query = f"""
                MATCH (s:Entity {{name: $source}}), (t:Entity {{name: $target}})
                MERGE (s)-[r:{safe_rel_type}]->(t)
                """
                await session.run(query, source=source, target=target)

        return {
            "chunks": 1,
            "entities": len(entities),
            "relations": len(relations),
        }

    async def run_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute a read-only Cypher query.

        :param cypher_query: Cypher to execute.
        :return: Query records.
        :raises ValueError: If query is not read-only.
        """

        blocked_tokens = ("CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "LOAD CSV")
        upper = cypher_query.upper()
        if any(token in upper for token in blocked_tokens):
            raise ValueError("Non read-only Cypher query rejected")

        records: List[Dict[str, Any]] = []
        async with self._driver.session(database=settings.neo4j_database) as session:
            result = await session.run(cypher_query)
            async for record in result:
                records.append(record.data())

        return records

    async def graph_schema_hint(self) -> str:
        """Return coarse schema hints for query generation.

        :return: Schema hint string.
        """

        query = """
        MATCH (e:Entity)
        WITH collect(DISTINCT e.type)[0..20] AS labels
        MATCH ()-[r]->()
        WITH labels, collect(DISTINCT type(r))[0..40] AS rels
        RETURN labels, rels
        """
        async with self._driver.session(database=settings.neo4j_database) as session:
            result = await session.run(query)
            row = await result.single()
            if not row:
                return "Labels: [Entity], Relationships: [MENTIONS, RELATED_TO]"
            labels = row.get("labels", [])
            rels = row.get("rels", [])
            return f"Labels: {labels}; Relationships: {rels}"

    async def fetch_chunks(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch chunk nodes to support optional semantic reranking.

        :param limit: Max chunks.
        :return: Chunk records.
        """

        query = "MATCH (c:Chunk) RETURN c.id AS id, c.text AS text LIMIT $limit"
        rows: List[Dict[str, Any]] = []
        async with self._driver.session(database=settings.neo4j_database) as session:
            result = await session.run(query, limit=limit)
            async for record in result:
                rows.append(record.data())
        return rows
