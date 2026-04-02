from __future__ import annotations

import argparse
import asyncio
import json
import logging
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path

import asyncpg

from src.chunk.chunk_processor import MarkdownChunkProcessor
from clients.config import db_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class InputRecord:
    filename: str
    preview_url: str
    module: str


TEST_INPUT = InputRecord(
    filename="2020电动飞机传动系统磁性齿轮的设计.pdf",
    preview_url="http://36.103.203.113:2300/alpha/pdf/2026/03/05/bc06e781fe5f46aca8613b5be16a6ce6.pdf",
    module="飞行器",
)


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self):
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def execute(self, query: str, *args):
        self.executed.append((query, args))
        return "OK"

    async def fetch(self, query: str, *args):
        return []

    async def close(self):
        self.closed = True


class MarkdownChunkProcessorTests(unittest.TestCase):
    def test_generate_chunk_id_is_stable(self):
        processor = MarkdownChunkProcessor(markdown_dir=Path("."))
        cid_1 = processor.generate_chunk_id("docA", 7)
        cid_2 = processor.generate_chunk_id("docA", 7)
        cid_3 = processor.generate_chunk_id("docA", 8)

        self.assertEqual(cid_1, cid_2)
        self.assertNotEqual(cid_1, cid_3)
        self.assertEqual(64, len(cid_1))

    def test_parse_markdown_extracts_heading_path_and_page(self):
        markdown = """<!-- Page 3 -->
# Introduction
## Background
Some text line 1.
Some text line 2.

### Detail
More details.
"""
        processor = MarkdownChunkProcessor(markdown_dir=Path("."))
        segments = processor.parse_markdown(markdown)

        self.assertEqual("Introduction", segments[0]["text"])
        self.assertEqual(["Introduction"], segments[0]["section_path"])
        self.assertEqual(3, segments[0]["page"])
        self.assertEqual(["Introduction", "Background"], segments[2]["section_path"])

    def test_save_chunks_to_db_executes_delete_and_inserts(self):
        processor = MarkdownChunkProcessor(markdown_dir=Path("."))
        fake_conn = _FakeConnection()

        async def fake_connect():
            return fake_conn

        processor._connect = fake_connect  # type: ignore[method-assign]

        chunks = [
            {
                "chunk_id": "a" * 64,
                "doc_id": "doc_x",
                "content": "chunk 1",
                "page": 3,
                "section_path": ["A", "B"],
                "chunk_index": 0,
            },
            {
                "chunk_id": "b" * 64,
                "doc_id": "doc_x",
                "content": "chunk 2",
                "page": None,
                "section_path": [],
                "chunk_index": 1,
            },
        ]

        inserted_count = asyncio.run(processor.save_chunks_to_db("doc_x", chunks))

        self.assertEqual(2, inserted_count)
        self.assertTrue(fake_conn.closed)
        self.assertEqual(3, len(fake_conn.executed))


def ensure_data_dirs(project_root: Path) -> Path:
    markdown_dir = project_root / "data" / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    return markdown_dir


def resolve_markdown_path(markdown_dir: Path, file_stem: str) -> Path:
    candidates = [
        markdown_dir / f"{file_stem}.md",
        markdown_dir / file_stem / f"{file_stem}.md",
        markdown_dir / file_stem / "vlm" / f"{file_stem}.md",
    ]

    for candidate in candidates:
        logger.info("查找Markdown文件: %s", candidate)
        if candidate.exists():
            logger.info("找到markdown文件: %s", candidate)
            return candidate

    raise FileNotFoundError(f"Markdown 文件不存在: {candidates[-1]}")


async def _fetch_doc_id_by_source_url(source_url: str) -> str | None:
    dsn = db_config.url.replace("postgresql+asyncpg://", "postgresql://", 1)
    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT doc_id
            FROM documents
            WHERE source_url = $1
            ORDER BY create_time DESC NULLS LAST
            LIMIT 1
            """,
            source_url,
        )
        if row is None:
            return None
        return row["doc_id"]
    finally:
        await conn.close()


def resolve_doc_id_from_documents(source_url: str) -> str:
    doc_id = asyncio.run(_fetch_doc_id_by_source_url(source_url))
    if not doc_id:
        raise RuntimeError(
            "documents 表中不存在 source_url 对应记录，请先执行 pdf_test.py 完成 documents 入库后再执行 chunks_test.py"
        )
    return doc_id


def ensure_document_exists(doc_id: str) -> None:
    logger.info("documents 主记录存在: doc_id=%s", doc_id)


def run_test(write_db: bool = True) -> dict[str, object]:
    project_root = Path(__file__).resolve().parent
    markdown_dir = ensure_data_dirs(project_root)

    logger.info("[1/5] 从 documents 表获取 doc_id")
    doc_id = resolve_doc_id_from_documents(TEST_INPUT.preview_url)

    logger.info("[2/5] 定位 Markdown 文件")
    markdown_path = resolve_markdown_path(markdown_dir, Path(TEST_INPUT.filename).stem)

    logger.info("[3/5] 检查 documents 主记录")
    ensure_document_exists(doc_id)

    logger.info("[4/5] 解析 Markdown 并切分 chunks")
    processor = MarkdownChunkProcessor(markdown_dir=markdown_dir)
    document = processor.load_markdown_file(markdown_path, doc_id=doc_id)
    parsed_segments, chunks = processor.build_chunks(doc_id=doc_id, markdown_text=document["content"])
    logger.info("Parsed segments: %d", len(parsed_segments))
    logger.info("Generated chunks: %d", len(chunks))

    db_written = False
    inserted_count = 0
    if write_db:
        logger.info("[5/5] 写入 chunks 表")
        inserted_count = asyncio.run(processor.save_chunks_to_db(doc_id=doc_id, chunks=chunks))
        logger.info("Inserted chunks into database: %d", inserted_count)
        db_written = True

    sample_chunk = None
    if chunks:
        sample_chunk = {
            "chunk_id": chunks[0]["chunk_id"],
            "page": chunks[0]["page"],
            "section_path": chunks[0]["section_path"],
            "chunk_index": chunks[0]["chunk_index"],
            "content_preview": chunks[0]["content"][:120],
        }

    result = {
        "input": asdict(TEST_INPUT),
        "doc_id": doc_id,
        "markdown_path": str(markdown_path),
        "parsed_segment_count": len(parsed_segments),
        "chunk_count": len(chunks),
        "inserted_count": inserted_count,
        "sample_chunk": sample_chunk,
        "db_written": db_written,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-case test for Markdown chunk pipeline")
    parser.add_argument("--skip-db", action="store_true", help="Only test chunk parsing; skip chunks upsert")
    parser.add_argument("--unit", action="store_true", help="Run unit tests instead of the integration-style script")
    args = parser.parse_args()

    if args.unit:
        unittest.main(argv=["chunks_test.py"], verbosity=2)
    else:
        output = run_test(write_db=not args.skip_db)
        print(json.dumps(output, ensure_ascii=False, indent=2))
