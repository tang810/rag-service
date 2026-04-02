from __future__ import annotations

"""Batch ingest local PDF files into the RAG database.

Pipeline per paper:
1. PDF -> Markdown (MinerU)
2. Metadata extraction -> documents table
3. Markdown chunking -> chunks table
4. Embeddings -> embeddings table
5. Optional formula extraction -> formulas / physical_quantities tables

Default PDF directory: src/mypdf
"""

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from src.chunk.chunk_processor import MarkdownChunkProcessor
from src.clients.config import db_config
from src.clients.embedding_client import EmbeddingAPIError, get_embeddings
from src.embedding.embedding_service import EmbeddingService
from src.extract.offline_extract import run_offline_extract_for_doc
from src.pdf.doc_id_generator import generate_doc_id
from src.pdf.document_writer import DocumentRecord, DocumentWriter
from src.pdf.metadata_extractor import MetadataExtractor, save_metadata_json
from src.pdf.pdf_to_md import extract_pdf_to_md

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PDF_DIR = Path(__file__).resolve().parent / "src" / "mypdf"


@dataclass
class LocalPaper:
    filename: str
    source_type: str


def _get_dsn() -> str:
    return db_config.url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _normalized_markdown_filename(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[^\w\-]", "_", stem)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    normalized = cleaned or stem
    return f"{normalized}.md"


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
    conn = await asyncpg.connect(_get_dsn())
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
        return row["doc_id"] if row else None
    finally:
        await conn.close()


def resolve_doc_id(source_url: str) -> str:
    doc_id = asyncio.run(_fetch_doc_id_by_source_url(source_url))
    if not doc_id:
        raise RuntimeError(f"documents table has no row for source_url={source_url}")
    return doc_id


async def _fetch_chunks_without_embedding(doc_id: str, limit: int) -> list[tuple[str, str, str]]:
    conn = await asyncpg.connect(_get_dsn())
    try:
        rows = await conn.fetch(
            """
            SELECT c.chunk_id, c.doc_id, c.content
            FROM chunks c
            WHERE c.doc_id = $1
              AND NOT EXISTS (
                  SELECT 1 FROM embeddings e WHERE e.chunk_id = c.chunk_id
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


async def _check_doc_exists(source_url: str) -> bool:
    conn = await asyncpg.connect(_get_dsn())
    try:
        row = await conn.fetchrow("SELECT 1 FROM documents WHERE source_url = $1 LIMIT 1", source_url)
        return row is not None
    finally:
        await conn.close()


async def _get_stats() -> list[dict]:
    conn = await asyncpg.connect(_get_dsn())
    try:
        rows = await conn.fetch(
            """
            SELECT
                d.source_type,
                COUNT(DISTINCT d.doc_id) AS doc_count,
                COUNT(DISTINCT c.chunk_id) AS chunk_count,
                COUNT(DISTINCT e.chunk_id) AS embedding_count
            FROM documents d
            LEFT JOIN chunks c ON d.doc_id = c.doc_id
            LEFT JOIN embeddings e ON d.doc_id = e.doc_id
            GROUP BY d.source_type
            ORDER BY d.source_type
            """
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def run_embedding_stage(doc_id: str, batch_size: int = 32, limit: int = 1000) -> dict:
    service = EmbeddingService()
    total_fetched = 0
    total_processed = 0
    total_inserted = 0

    while True:
        chunk_rows = await _fetch_chunks_without_embedding(doc_id, limit)
        if not chunk_rows:
            break

        total_fetched += len(chunk_rows)

        for i in range(0, len(chunk_rows), batch_size):
            batch = chunk_rows[i : i + batch_size]
            texts = [item[2] for item in batch]

            try:
                vectors = get_embeddings(texts, batch_size=min(batch_size, len(texts)))
            except EmbeddingAPIError as exc:
                logger.error("Embedding API failed doc_id=%s batch=%d: %s", doc_id, i, exc)
                continue

            records = [
                (chunk_id, row_doc_id, vector)
                for (chunk_id, row_doc_id, _), vector in zip(batch, vectors, strict=True)
            ]
            total_processed += len(records)
            inserted = await service.insert_embeddings(records)
            total_inserted += inserted

        if len(chunk_rows) < limit:
            break

    return {
        "chunks_fetched": total_fetched,
        "chunks_processed": total_processed,
        "inserted_count": total_inserted,
    }


def process_one_paper(
    paper: LocalPaper,
    pdf_dir: Path,
    markdown_dir: Path,
    parsed_dir: Path,
    run_extract: bool = True,
) -> dict:
    pdf_path = pdf_dir / paper.filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    source_url = f"local://{paper.filename}"

    # If the document already exists, optionally backfill formula extraction.
    if asyncio.run(_check_doc_exists(source_url)):
        logger.info("  Already ingested, skip document build: %s", paper.filename)
        existing_doc_id = resolve_doc_id(source_url)

        extract_result: dict[str, object] = {"success": True, "skipped": True, "reason": "disabled by flag"}
        if run_extract:
            logger.info("  Backfilling formula extraction for existing doc")
            try:
                extract_result = asyncio.run(
                    run_offline_extract_for_doc(
                        doc_id=existing_doc_id,
                        filename=_normalized_markdown_filename(paper.filename),
                    )
                )
                logger.info(
                    "  Backfill done: formulas=%s, quantities=%s",
                    extract_result.get("formulas_written", 0),
                    extract_result.get("quantities_written", 0),
                )
            except BaseException as exc:
                logger.warning("  Formula extraction failed for %s: %s", paper.filename, exc)
                extract_result = {"success": False, "error": str(exc)}

        return {
            "doc_id": existing_doc_id,
            "skipped": True,
            "extract_success": bool(extract_result.get("success", False)),
            "formulas_written": int(extract_result.get("formulas_written", 0) or 0),
            "quantities_written": int(extract_result.get("quantities_written", 0) or 0),
        }

    logger.info("  [1/7] Generate doc_id")
    generated_doc_id = generate_doc_id(paper.filename)

    logger.info("  [2/7] PDF -> Markdown (MinerU)")
    markdown_path = extract_pdf_to_md(
        input_path=str(pdf_path),
        output_dir=str(markdown_dir),
        cleanup=True,
        keep_images=False,
    )
    if not markdown_path or not Path(markdown_path).exists():
        raise RuntimeError("Markdown parse failed")

    logger.info("  [3/7] Extract metadata with LLM")
    markdown_text = Path(markdown_path).read_text(encoding="utf-8")
    metadata = MetadataExtractor().extract(markdown_text)

    logger.info("  [4/7] Save metadata JSON")
    save_metadata_json(parsed_dir / f"{generated_doc_id}.json", metadata)

    logger.info("  [5/7] Writing documents table")
    DocumentWriter().upsert_document(
        DocumentRecord(
            doc_id=generated_doc_id,
            title=metadata.title,
            authors=metadata.authors,
            keywords=metadata.keywords,
            journal_conference=metadata.journal_conference,
            publish_year=metadata.publish_year,
            abstract=metadata.abstract,
            source_url=source_url,
            source_type=paper.source_type,
            doc_type=metadata.doc_type,
        )
    )

    logger.info("  [6/7] Resolve persisted doc_id")
    doc_id = resolve_doc_id(source_url)

    logger.info("  [7/7] Build chunks")
    processor = MarkdownChunkProcessor(markdown_dir=Path(markdown_path).parent)
    document = processor.load_markdown_file(markdown_path=Path(markdown_path), doc_id=doc_id)
    _, chunks = processor.build_chunks(doc_id=doc_id, markdown_text=document["content"])

    chunk_count = len(chunks)
    inserted_chunks = asyncio.run(processor.save_chunks_to_db(doc_id=doc_id, chunks=chunks))
    logger.info("  Chunks: %d, inserted: %d", chunk_count, inserted_chunks)

    logger.info("  Build embeddings")
    emb_result = asyncio.run(run_embedding_stage(doc_id=doc_id))
    logger.info("  Embeddings: processed=%d, inserted=%d", emb_result["chunks_processed"], emb_result["inserted_count"])

    extract_result: dict[str, object] = {"success": True, "skipped": True, "reason": "disabled by flag"}
    if run_extract:
        logger.info("  Running formula extraction")
        try:
            extract_result = asyncio.run(
                run_offline_extract_for_doc(doc_id=doc_id, filename=Path(markdown_path).name)
            )
            logger.info(
                "  Extraction done: formulas=%s, quantities=%s",
                extract_result.get("formulas_written", 0),
                extract_result.get("quantities_written", 0),
            )
        except BaseException as exc:
            logger.warning("  Formula extraction failed for %s: %s", paper.filename, exc)
            extract_result = {"success": False, "error": str(exc)}

    return {
        "doc_id": doc_id,
        "title": metadata.title,
        "chunk_count": chunk_count,
        "embedding_count": emb_result["inserted_count"],
        "markdown_path": str(markdown_path),
        "extract_success": bool(extract_result.get("success", False)),
        "formulas_written": int(extract_result.get("formulas_written", 0) or 0),
        "quantities_written": int(extract_result.get("quantities_written", 0) or 0),
    }


def scan_pdf_dir(pdf_dir: Path, source_type: str) -> list[LocalPaper]:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    papers = [LocalPaper(filename=f.name, source_type=source_type) for f in pdf_files]
    logger.info("Discovered %d PDF files (source_type=%s)", len(papers), source_type)
    for paper in papers:
        logger.info("  - %s", paper.filename)
    return papers


def run_batch(
    papers: list[LocalPaper],
    pdf_dir: Path,
    markdown_dir: Path,
    parsed_dir: Path,
    run_extract: bool = True,
):
    results: list[dict] = []
    total = len(papers)

    for i, paper in enumerate(papers, 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] %s (source_type=%s)", i, total, paper.filename, paper.source_type)
        logger.info("=" * 60)

        try:
            info = process_one_paper(
                paper=paper,
                pdf_dir=pdf_dir,
                markdown_dir=markdown_dir,
                parsed_dir=parsed_dir,
                run_extract=run_extract,
            )
            if info.get("skipped"):
                results.append({"name": paper.filename, "status": "skipped", **info})
            else:
                results.append({"name": paper.filename, "status": "success", **info})
        except Exception as exc:
            logger.error("[%d/%d] Failed: %s", i, total, exc)
            results.append({"name": paper.filename, "status": "failed", "error": str(exc)})

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    logger.info("=" * 60)
    logger.info("Finished: success=%d, skipped=%d, failed=%d, total=%d", success, skipped, failed, total)
    logger.info("=" * 60)

    print(json.dumps(results, ensure_ascii=False, indent=2))


def show_stats():
    stats = asyncio.run(_get_stats())
    if not stats:
        print("No documents found in database.")
        return

    print("\n" + "=" * 60)
    print("  Ingestion Stats")
    print("=" * 60)
    print(f"  {'SourceType':<20} {'Docs':<10} {'Chunks':<10} {'Embeddings':<12}")
    print("-" * 60)
    for row in stats:
        source_type = row["source_type"] or "(unset)"
        print(
            f"  {source_type:<20} {row['doc_count']:<10} "
            f"{row['chunk_count']:<10} {row['embedding_count']:<12}"
        )
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch ingest local PDF files into the RAG database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Scan default directory src/mypdf
  python batch_upload.py --source-type MyDomain

  # Scan a custom directory
  python batch_upload.py --dir src/mypdf --source-type MyDomain

  # Ingest a single file
  python batch_upload.py --file src/mypdf/paper.pdf --source-type MyDomain

  # Show current stats
  python batch_upload.py --stats

  # Current default PDF directory
  {DEFAULT_PDF_DIR}
        """,
    )

    parser.add_argument("--dir", type=str, help="Directory to scan for .pdf files")
    parser.add_argument("--file", type=str, help="Single PDF file path")
    parser.add_argument("--source-type", type=str, help="Domain label, e.g. MyDomain")
    parser.add_argument("--stats", action="store_true", help="Show ingestion stats only")
    parser.add_argument("--skip-extract", action="store_true", help="Skip formula extraction stage")

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if not args.source_type:
        parser.error("Missing required argument: --source-type")

    if args.dir and args.file:
        parser.error("Arguments --dir and --file are mutually exclusive")

    project_root = Path(__file__).resolve().parent
    _, _, markdown_dir, parsed_dir = ensure_data_dirs(project_root)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            parser.error(f"File does not exist: {file_path}")
        papers = [LocalPaper(filename=file_path.name, source_type=args.source_type)]
        pdf_dir = file_path.parent
    else:
        pdf_dir = Path(args.dir) if args.dir else DEFAULT_PDF_DIR
        logger.info("Using PDF directory: %s", pdf_dir)
        papers = scan_pdf_dir(pdf_dir, args.source_type)

    run_batch(
        papers=papers,
        pdf_dir=pdf_dir,
        markdown_dir=markdown_dir,
        parsed_dir=parsed_dir,
        run_extract=not args.skip_extract,
    )


if __name__ == "__main__":
    main()


