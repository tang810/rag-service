from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import asyncpg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunk.chunk_processor import MarkdownChunkProcessor
from src.clients.config import db_config
from src.clients.embedding_client import EmbeddingAPIError, get_embeddings
from src.embedding.embedding_service import EmbeddingService
from src.extract.offline_extract import run_offline_extract_for_doc
from src.pdf.doc_id_generator import generate_doc_id
from src.pdf.document_writer import DocumentRecord, DocumentWriter
from src.pdf.metadata_extractor import MetadataExtractor, save_metadata_json
from src.pdf.pdf_downloader import download_pdf
from src.pdf.pdf_to_md import extract_pdf_to_md


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class InputRecord:
    filename: str
    preview_url: str
    module: str


TEST_INPUT = InputRecord(
    filename="轻型倾转旋翼机总体设计与参数优化.pdf",
    preview_url="http://36.103.203.113:2300/alpha/pdf/2026/03/05/bf89d0e8a84543859072cdf4f38344e0.pdf",
    module="aircraft",
)


def ensure_data_dirs(project_root: Path) -> tuple[Path, Path, Path, Path]:
    data_root = project_root / "data"
    pdf_dir = data_root / "pdf"
    markdown_dir = data_root / "markdown"
    parsed_dir = data_root / "parsed"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    return data_root, pdf_dir, markdown_dir, parsed_dir


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


async def _fetch_source_type_by_doc_id(doc_id: str) -> str | None:
    dsn = db_config.url.replace("postgresql+asyncpg://", "postgresql://", 1)
    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT source_type
            FROM documents
            WHERE doc_id = $1
            LIMIT 1
            """,
            doc_id,
        )
        if row is None:
            return None
        return row["source_type"]
    finally:
        await conn.close()


def resolve_doc_id_from_documents(source_url: str) -> str:
    doc_id = asyncio.run(_fetch_doc_id_by_source_url(source_url))
    if not doc_id:
        raise RuntimeError("documents 表中不存在 source_url 对应记录，无法确定 doc_id")
    return doc_id


def _is_aircraft_source(source_type: str | None) -> bool:
    if not source_type:
        return False
    normalized = source_type.strip().lower()
    return normalized in {"aircraft", "飞行器"}


def run_pdf_workflow(test_input: InputRecord, write_pdf_db: bool = True) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    _, pdf_dir, markdown_dir, parsed_dir = ensure_data_dirs(project_root)

    logger.info("[1/13] PDF工作: 生成 doc_id")
    generated_doc_id = generate_doc_id(test_input.preview_url)

    logger.info("[2/13] PDF工作: 下载 PDF")
    pdf_path = download_pdf(url=test_input.preview_url, output_dir=pdf_dir, original_name=test_input.filename)

    logger.info("[3/13] PDF工作: MinerU 解析 Markdown")
    markdown_path = extract_pdf_to_md(
        input_path=str(pdf_path),
        output_dir=str(markdown_dir),
        cleanup=True,
        keep_images=False,
    )
    if markdown_path is None or not Path(markdown_path).exists():
        raise RuntimeError("Markdown 解析失败，未找到输出文件")

    logger.info("[4/13] PDF工作: LLM 提取 metadata")
    markdown_text = Path(markdown_path).read_text(encoding="utf-8")
    metadata = MetadataExtractor().extract(markdown_text)

    logger.info("[5/13] PDF工作: 保存 metadata JSON")
    metadata_json_path = parsed_dir / f"{generated_doc_id}.json"
    save_metadata_json(metadata_json_path, metadata)

    pdf_db_written = False
    doc_id = generated_doc_id
    if write_pdf_db:
        logger.info("[6/13] PDF工作: 写入 documents 表")
        document = DocumentRecord(
            doc_id=generated_doc_id,
            title=metadata.title,
            authors=metadata.authors,
            keywords=metadata.keywords,
            journal_conference=metadata.journal_conference,
            publish_year=metadata.publish_year,
            abstract=metadata.abstract,
            source_url=test_input.preview_url,
            source_type=test_input.module,
            doc_type=metadata.doc_type,
        )
        DocumentWriter().upsert_document(document)
        pdf_db_written = True

        logger.info("[7/13] PDF工作: 从 documents 表读取 doc_id")
        doc_id = resolve_doc_id_from_documents(test_input.preview_url)
        logger.info("documents doc_id: %s", doc_id)

    return {
        "doc_id": doc_id,
        "generated_doc_id": generated_doc_id,
        "pdf_path": str(pdf_path),
        "markdown_path": str(markdown_path),
        "metadata_json_path": str(metadata_json_path),
        "metadata": metadata.to_dict(),
        "pdf_db_written": pdf_db_written,
    }


def run_chunks_workflow(doc_id: str, markdown_path: str, write_chunks_db: bool = True) -> dict[str, object]:
    logger.info("[8/13] chunks工作: 构建 chunks")
    processor = MarkdownChunkProcessor(markdown_dir=Path(markdown_path).parent)
    document = processor.load_markdown_file(markdown_path=Path(markdown_path), doc_id=doc_id)
    parsed_segments, chunks = processor.build_chunks(doc_id=doc_id, markdown_text=document["content"])

    logger.info("[9/13] chunks工作: Parsed segments=%d, Generated chunks=%d", len(parsed_segments), len(chunks))

    inserted_count = 0
    chunks_db_written = False
    if write_chunks_db:
        logger.info("[10/13] chunks工作: 写入 chunks 表")
        inserted_count = asyncio.run(processor.save_chunks_to_db(doc_id=doc_id, chunks=chunks))
        chunks_db_written = True
        logger.info("Inserted chunks into database: %d", inserted_count)

    sample_chunk = None
    if chunks:
        sample_chunk = {
            "chunk_id": chunks[0]["chunk_id"],
            "page": chunks[0]["page"],
            "section_path": chunks[0]["section_path"],
            "chunk_index": chunks[0]["chunk_index"],
            "content_preview": chunks[0]["content"][:120],
        }

    return {
        "parsed_segment_count": len(parsed_segments),
        "chunk_count": len(chunks),
        "inserted_count": inserted_count,
        "sample_chunk": sample_chunk,
        "chunks_db_written": chunks_db_written,
    }


async def _fetch_doc_chunks_without_embedding(doc_id: str, limit: int) -> list[tuple[str, str, str]]:
    dsn = db_config.url.replace("postgresql+asyncpg://", "postgresql://", 1)
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT c.chunk_id, c.doc_id, c.content
            FROM chunks c
            WHERE c.doc_id = $1
              AND NOT EXISTS (
                  SELECT 1
                  FROM embeddings e
                  WHERE e.chunk_id = c.chunk_id
              )
            ORDER BY c.chunk_index ASC NULLS LAST, c.id ASC
            LIMIT $2
            """,
            doc_id,
            limit,
        )
        return [(row["chunk_id"], row["doc_id"], row["content"]) for row in rows]
    finally:
        await conn.close()


async def _run_embedding_stage(
    doc_id: str,
    write_embeddings_db: bool = True,
    batch_size: int = 32,
    limit: int = 1000,
) -> dict[str, object]:
    logger.info("[11/13] embedding工作: 读取未向量化 chunks 并生成向量")
    service = EmbeddingService()

    chunks_fetched = 0
    chunks_processed = 0
    inserted_count = 0
    api_time_seconds = 0.0
    embedding_size = 0
    sample_embedding_preview: list[float] | None = None

    while True:
        chunk_rows = await _fetch_doc_chunks_without_embedding(doc_id=doc_id, limit=limit)
        if not chunk_rows:
            break

        chunks_fetched += len(chunk_rows)

        for i in range(0, len(chunk_rows), batch_size):
            batch = chunk_rows[i : i + batch_size]
            texts = [item[2] for item in batch]

            api_start = time.perf_counter()
            try:
                vectors = get_embeddings(texts, batch_size=min(batch_size, len(texts)))
            except EmbeddingAPIError as exc:
                logger.error("Embedding API failed for doc_id=%s, batch_start=%d: %s", doc_id, i, exc)
                continue
            api_time_seconds += time.perf_counter() - api_start

            if vectors:
                embedding_size = len(vectors[0])
                if sample_embedding_preview is None:
                    sample_embedding_preview = vectors[0][:8]

            records = [
                (chunk_id, row_doc_id, vector)
                for (chunk_id, row_doc_id, _content), vector in zip(batch, vectors, strict=True)
            ]

            chunks_processed += len(records)

            if write_embeddings_db:
                inserted = await service.insert_embeddings(records)
                inserted_count += inserted

        if len(chunk_rows) < limit:
            break

    return {
        "doc_id": doc_id,
        "embedding_batch_size": batch_size,
        "embedding_query_limit": limit,
        "embedding_chunks_fetched": chunks_fetched,
        "embedding_chunks_processed": chunks_processed,
        "embedding_size": embedding_size,
        "embedding_api_time_seconds": round(api_time_seconds, 3),
        "embedding_inserted_count": inserted_count,
        "embeddings_db_written": write_embeddings_db,
        "sample_embedding_preview": sample_embedding_preview,
    }


def run_pdf_chunks_embeddiing_extract_test(
    write_pdf_db: bool = True,
    write_chunks_db: bool = True,
    write_embeddings_db: bool = True,
    run_extract_if_aircraft: bool = True,
    embedding_batch_size: int = 32,
    embedding_limit: int = 1000,
) -> dict[str, object]:
    if run_extract_if_aircraft and not write_pdf_db:
        raise RuntimeError("提取阶段依赖 documents 表中的 doc_id/source_type；请勿同时使用 --skip-pdf-db 与提取")

    pdf_output = run_pdf_workflow(TEST_INPUT, write_pdf_db=write_pdf_db)
    chunks_output = run_chunks_workflow(
        doc_id=str(pdf_output["doc_id"]),
        markdown_path=str(pdf_output["markdown_path"]),
        write_chunks_db=write_chunks_db,
    )

    doc_id = str(pdf_output["doc_id"])
    embedding_output = asyncio.run(
        _run_embedding_stage(
            doc_id=doc_id,
            write_embeddings_db=write_embeddings_db,
            batch_size=embedding_batch_size,
            limit=embedding_limit,
        )
    )

    source_type = asyncio.run(_fetch_source_type_by_doc_id(doc_id))
    if run_extract_if_aircraft and not source_type:
        raise RuntimeError(f"doc_id={doc_id} 在 documents 表中未查询到 source_type，无法判断是否执行提取")

    should_extract = run_extract_if_aircraft and _is_aircraft_source(source_type)

    extract_output: dict[str, object]
    if should_extract:
        logger.info("[12/13] 提取工作: source_type=%s，执行公式/物理量提取并入库", source_type)
        extract_output = asyncio.run(run_offline_extract_for_doc(doc_id=doc_id, filename=TEST_INPUT.filename))
    else:
        logger.info("[12/13] 提取工作: source_type=%s，跳过离线提取", source_type)
        extract_output = {
            "success": True,
            "skipped": True,
            "reason": "source_type is not aircraft",
            "source_type": source_type,
        }

    logger.info("[13/13] 流程结束")

    return {
        "input": asdict(TEST_INPUT),
        "doc_id": doc_id,
        "generated_doc_id": pdf_output["generated_doc_id"],
        "source_type": source_type,
        "pdf_workflow": pdf_output,
        "chunks_workflow": chunks_output,
        "embedding_workflow": embedding_output,
        "extract_workflow": extract_output,
        "summary": {
            "pdf_done": True,
            "chunks_done": True,
            "embeddings_done": True,
            "extract_done": bool(extract_output.get("success", False)),
            "extract_skipped": bool(extract_output.get("skipped", False)),
            "chunk_count": chunks_output["chunk_count"],
            "embedding_chunks_processed": embedding_output["embedding_chunks_processed"],
            "embedding_inserted_count": embedding_output["embedding_inserted_count"],
            "formulas_written": extract_output.get("formulas_written", 0),
            "quantities_written": extract_output.get("quantities_written", 0),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end PDF + chunks + embedding + extract integration test")
    parser.add_argument("--skip-pdf-db", action="store_true", help="Skip documents upsert in PDF stage")
    parser.add_argument("--skip-chunks-db", action="store_true", help="Skip chunks upsert in chunks stage")
    parser.add_argument("--skip-embeddings-db", action="store_true", help="Skip embeddings write in embedding stage")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction stage even if source_type is aircraft")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Embedding API batch size")
    parser.add_argument("--embedding-limit", type=int, default=1000, help="Per-loop max chunks fetched for this doc")
    args = parser.parse_args()

    result = run_pdf_chunks_embeddiing_extract_test(
        write_pdf_db=not args.skip_pdf_db,
        write_chunks_db=not args.skip_chunks_db,
        write_embeddings_db=not args.skip_embeddings_db,
        run_extract_if_aircraft=not args.skip_extract,
        embedding_batch_size=args.embedding_batch_size,
        embedding_limit=args.embedding_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
