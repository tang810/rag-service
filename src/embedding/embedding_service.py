from __future__ import annotations

from typing import Iterable

import asyncpg

from src.clients.config import db_config


class EmbeddingService:
    def __init__(self):
        self._dsn = db_config.url.replace("postgresql+asyncpg://", "postgresql://", 1)

    async def _connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(self._dsn)

    async def get_chunks_without_embedding(self, limit: int) -> list[tuple[str, str, str]]:
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT c.chunk_id, c.doc_id, c.content
                FROM chunks c
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM embeddings e
                    WHERE e.chunk_id = c.chunk_id
                )
                ORDER BY c.id ASC
                LIMIT $1
                """,
                limit,
            )
            return [(row["chunk_id"], row["doc_id"], row["content"]) for row in rows]
        except asyncpg.PostgresError as exc:
            raise RuntimeError(f"Failed to query chunks without embedding: {exc}") from exc
        finally:
            await conn.close()

    @staticmethod
    def _vector_to_pg(vector: Iterable[float]) -> str:
        # pgvector accepts text format like: [0.1,0.2,...]
        return "[" + ",".join(f"{float(v):.10f}" for v in vector) + "]"

    async def insert_embeddings(self, records: list[tuple[str, str, list[float]]]) -> int:
        if not records:
            return 0

        conn = await self._connect()
        try:
            payload = [
                (chunk_id, doc_id, self._vector_to_pg(embedding_vector))
                for chunk_id, doc_id, embedding_vector in records
            ]
            await conn.executemany(
                """
                INSERT INTO embeddings (chunk_id, doc_id, embedding)
                VALUES ($1, $2, $3::vector)
                """,
                payload,
            )
            return len(payload)
        except asyncpg.PostgresError as exc:
            raise RuntimeError(f"Failed to insert embeddings: {exc}") from exc
        finally:
            await conn.close()
