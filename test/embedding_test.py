from __future__ import annotations

import asyncio
import json
import logging
import time

import asyncpg

from clients.config import db_config
from clients.embedding_client import get_embeddings
from src.embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


async def _fetch_test_chunks(limit: int = 10) -> list[tuple[str, str, str]]:
	dsn = db_config.url.replace("postgresql+asyncpg://", "postgresql://", 1)
	conn = await asyncpg.connect(dsn)
	try:
		rows = await conn.fetch(
			"""
			SELECT c.chunk_id, c.doc_id, c.content
			FROM chunks c
			WHERE NOT EXISTS (
				SELECT 1 FROM embeddings e WHERE e.chunk_id = c.chunk_id
			)
			ORDER BY c.id ASC
			LIMIT $1
			""",
			limit,
		)
		return [(row["chunk_id"], row["doc_id"], row["content"]) for row in rows]
	finally:
		await conn.close()


async def _run_test() -> None:
	result: dict[str, object] = {
		"chunks_requested": 10,
		"chunks_fetched": 0,
		"embedding_size": 0,
		"api_batch_size": 10,
		"api_time_seconds": 0.0,
		"chunks_processed": 0,
		"inserted_count": 0,
		"insert_success": False,
	}

	service = EmbeddingService()
	logger.info("[1/4] 从 chunks 表读取测试数据")
	chunks = await _fetch_test_chunks(limit=10)
	result["chunks_fetched"] = len(chunks)

	if not chunks:
		logger.info("未找到可用于测试的 chunks（可能都已生成 embedding）")
		print(json.dumps(result, ensure_ascii=False, indent=2))
		return

	texts = [item[2] for item in chunks]
	logger.info("[2/4] 调用 Embedding API")
	api_start = time.perf_counter()
	vectors = get_embeddings(texts, batch_size=10)
	result["api_time_seconds"] = round(time.perf_counter() - api_start, 3)

	if not vectors:
		raise RuntimeError("embedding API returned empty vectors")

	result["embedding_size"] = len(vectors[0])
	print(f"embedding size: {len(vectors[0])}")

	records = [
		(chunk_id, doc_id, vector)
		for (chunk_id, doc_id, _content), vector in zip(chunks, vectors, strict=True)
	]

	logger.info("[3/4] 写入 embeddings 表")
	inserted = await service.insert_embeddings(records)
	result["chunks_processed"] = len(chunks)
	result["inserted_count"] = inserted
	result["insert_success"] = inserted > 0
	result["sample_chunk"] = {
		"chunk_id": chunks[0][0],
		"doc_id": chunks[0][1],
		"content_preview": chunks[0][2][:120],
	}
	result["sample_embedding_preview"] = vectors[0][:8]

	print(f"chunks processed: {len(chunks)}")
	if inserted > 0:
		print("insert success")
	else:
		print("insert skipped")

	logger.info("[4/4] 输出测试结果")
	print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
	asyncio.run(_run_test())


if __name__ == "__main__":
	main()
