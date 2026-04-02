from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.clients.embedding_client import EmbeddingAPIError, get_embeddings
from src.embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def build_embeddings(batch_size: int = 32, limit: int = 1000) -> dict[str, Any]:
    """Build embeddings for chunks that do not have vectors yet."""
    return asyncio.run(_build_embeddings_async(batch_size=batch_size, limit=limit))


async def _build_embeddings_async(batch_size: int = 32, limit: int = 1000) -> dict[str, Any]:
    service = EmbeddingService()

    total_chunks = 0
    total_inserted = 0
    total_api_time = 0.0
    start_at = time.perf_counter()

    while True:
        try:
            chunk_rows = await service.get_chunks_without_embedding(limit=limit)
        except Exception as exc:
            logger.exception("Database read error: %s", exc)
            raise

        if not chunk_rows:
            break

        for i in range(0, len(chunk_rows), batch_size):
            batch = chunk_rows[i : i + batch_size]
            texts = [item[2] for item in batch]

            api_start = time.perf_counter()
            try:
                vectors = get_embeddings(texts, batch_size=min(batch_size, len(texts)))
            except EmbeddingAPIError as exc:
                logger.error("Embedding API failed for batch starting at %d: %s", i, exc)
                continue
            except Exception as exc:
                logger.exception("Unexpected embedding API error: %s", exc)
                continue
            total_api_time += time.perf_counter() - api_start

            records = [
                (chunk_id, doc_id, vector)
                for (chunk_id, doc_id, _content), vector in zip(batch, vectors, strict=True)
            ]

            try:
                inserted = await service.insert_embeddings(records)
            except Exception as exc:
                logger.error("Database write error for batch starting at %d: %s", i, exc)
                continue

            total_chunks += len(batch)
            total_inserted += inserted
            logger.info("Processed batch=%d, inserted=%d", len(batch), inserted)

        if len(chunk_rows) < limit:
            break

    elapsed = time.perf_counter() - start_at

    print(f"处理chunk数量: {total_chunks}")
    print(f"embedding生成耗时: {total_api_time:.2f}s")
    print(f"写入数据库数量: {total_inserted}")

    return {
        "chunks_processed": total_chunks,
        "api_time_seconds": round(total_api_time, 3),
        "inserted_count": total_inserted,
        "elapsed_seconds": round(elapsed, 3),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    build_embeddings()
