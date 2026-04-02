from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import asyncpg

from src.chunk.chunk_processor import MarkdownChunkProcessor
from clients.config import db_config
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
    filename="2020电动飞机传动系统磁性齿轮的设计.pdf",
    preview_url="http://36.103.203.113:2300/alpha/pdf/2026/03/05/bc06e781fe5f46aca8613b5be16a6ce6.pdf",
    module="飞行器",
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


def resolve_doc_id_from_documents(source_url: str) -> str:
    doc_id = asyncio.run(_fetch_doc_id_by_source_url(source_url))
    if not doc_id:
        raise RuntimeError("documents 表中不存在 source_url 对应记录，无法为 chunks 阶段确定 doc_id")
    return doc_id


def run_pdf_chunks_test(write_pdf_db: bool = True, write_chunks_db: bool = True) -> dict[str, object]:
    project_root = Path(__file__).resolve().parent
    _, pdf_dir, markdown_dir, parsed_dir = ensure_data_dirs(project_root)

    logger.info("[1/8] 生成 doc_id")
    generated_doc_id = generate_doc_id(TEST_INPUT.preview_url)

    logger.info("[2/8] 下载 PDF")
    pdf_path = download_pdf(
        url=TEST_INPUT.preview_url,
        output_dir=pdf_dir,
        original_name=TEST_INPUT.filename,
    )

    logger.info("[3/8] MinerU 解析 Markdown")
    markdown_path = extract_pdf_to_md(
        input_path=str(pdf_path),
        output_dir=str(markdown_dir),
        cleanup=True,
        keep_images=False,
    )
    if markdown_path is None or not Path(markdown_path).exists():
        raise RuntimeError("Markdown 解析失败，未找到输出文件")

    logger.info("[4/8] LLM 提取 metadata")
    markdown_text = Path(markdown_path).read_text(encoding="utf-8")
    metadata = MetadataExtractor().extract(markdown_text)

    logger.info("[5/8] 保存 metadata JSON")
    metadata_json_path = parsed_dir / f"{generated_doc_id}.json"
    save_metadata_json(metadata_json_path, metadata)

    pdf_db_written = False
    if write_pdf_db:
        logger.info("[6/8] 写入 documents 表")
        document = DocumentRecord(
            doc_id=generated_doc_id,
            title=metadata.title,
            authors=metadata.authors,
            keywords=metadata.keywords,
            journal_conference=metadata.journal_conference,
            publish_year=metadata.publish_year,
            abstract=metadata.abstract,
            source_url=TEST_INPUT.preview_url,
            source_type=TEST_INPUT.module,
            doc_type=metadata.doc_type,
        )
        DocumentWriter().upsert_document(document)
        pdf_db_written = True

    logger.info("[7/8] 从 documents 表读取 doc_id")
    doc_id = resolve_doc_id_from_documents(TEST_INPUT.preview_url)
    logger.info("documents doc_id: %s", doc_id)

    logger.info("[8/8] 构建并写入 chunks")
    processor = MarkdownChunkProcessor(markdown_dir=Path(markdown_path).parent)
    document = processor.load_markdown_file(markdown_path=Path(markdown_path), doc_id=doc_id)
    parsed_segments, chunks = processor.build_chunks(doc_id=doc_id, markdown_text=document["content"])
    logger.info("Parsed segments: %d", len(parsed_segments))
    logger.info("Generated chunks: %d", len(chunks))

    inserted_count = 0
    chunks_db_written = False
    if write_chunks_db:
        inserted_count = asyncio.run(processor.save_chunks_to_db(doc_id=doc_id, chunks=chunks))
        logger.info("Inserted chunks into database: %d", inserted_count)
        chunks_db_written = True

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
        "input": asdict(TEST_INPUT),
        "doc_id": doc_id,
        "generated_doc_id": generated_doc_id,
        "pdf_path": str(pdf_path),
        "markdown_path": str(markdown_path),
        "metadata_json_path": str(metadata_json_path),
        "metadata": metadata.to_dict(),
        "pdf_db_written": pdf_db_written,
        "parsed_segment_count": len(parsed_segments),
        "chunk_count": len(chunks),
        "inserted_count": inserted_count,
        "sample_chunk": sample_chunk,
        "chunks_db_written": chunks_db_written,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end PDF + chunks integration test")
    parser.add_argument("--skip-pdf-db", action="store_true", help="Skip documents upsert in PDF stage")
    parser.add_argument("--skip-chunks-db", action="store_true", help="Skip chunks upsert in chunks stage")
    args = parser.parse_args()

    output = run_pdf_chunks_test(
        write_pdf_db=not args.skip_pdf_db,
        write_chunks_db=not args.skip_chunks_db,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
