"""Reusable prompt templates for GraphRAG workflows."""

ENTITY_RELATION_PROMPT = """
You are an information extraction engine.
Extract entities and relations from the text.
Rules:
1) Use only facts explicitly present in the text.
2) Do not infer or hallucinate missing facts.
3) Use canonical entity names.
4) Entity types should be concise labels such as Person, Company, Location, Product, Event.
5) Relation type must be UPPER_SNAKE_CASE.
6) Output JSON only with this exact schema:
{{
  "entities": [{{"name": "", "type": ""}}],
  "relations": [{{"source": "", "target": "", "type": ""}}]
}}
If nothing is found, return empty arrays.

Text:
{text}
""".strip()


CYPHER_GENERATION_PROMPT = """
You translate user questions into Neo4j Cypher.
Rules:
1) Return ONLY a valid read-only Cypher query.
2) NEVER use CREATE, MERGE, DELETE, SET, REMOVE, DROP, CALL dbms, LOAD CSV.
3) Prefer MATCH patterns using labels and relationship types.
4) Include LIMIT 25 if no limit is present.
5) Use existing schema hints:
{schema_hint}

Question:
{question}
""".strip()


ANSWER_GENERATION_PROMPT = """
You are a QA assistant.
Answer using ONLY the provided graph context.
If the answer is not in the context, reply exactly: I don't know
Be concise and factual.

Question:
{question}

Graph Context:
{context}
""".strip()
