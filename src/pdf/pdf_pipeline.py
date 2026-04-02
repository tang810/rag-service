from __future__ import annotations

import logging
from pathlib import Path

from pdf.doc_id_generator import generate_doc_id
from pdf.document_writer import DocumentRecord, DocumentWriter
from pdf.metadata_extractor import MetadataExtractor, save_metadata_json
from pdf.pdf_downloader import download_pdf
from pdf.pdf_to_md import extract_pdf_to_md

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _project_root() -> Path:
    # src/pdf_pipeline.py -> project root
    return Path(__file__).resolve().parents[1]


def _ensure_data_dirs() -> tuple[Path, Path, Path]:
    # Keep runtime artifacts under top-level data/ (parallel to src/).
    root = _project_root() / "data"
    pdf_dir = root / "pdf"
    markdown_dir = root / "markdown"
    parsed_dir = root / "parsed"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    return pdf_dir, markdown_dir, parsed_dir


def _resolve_markdown_path(md_output: Path, downloaded_pdf: Path) -> Path:
    if md_output.exists():
        return md_output
    stem = downloaded_pdf.stem
    fallback = md_output.parent / stem / f"{stem}.md"
    return fallback


def run_pdf_ingestion(limit: int | None = None) -> dict[str, int]:
    """
    Main pipeline:
    uploaded_files -> download PDF -> MinerU Markdown -> LLM metadata -> documents
    """
    pdf_dir, markdown_dir, parsed_dir = _ensure_data_dirs()
    writer = DocumentWriter()
    extractor = MetadataExtractor()

    rows = writer.fetch_uploaded_pdf_rows(limit=limit)
    total = len(rows)
    success = 0
    failed = 0

    logger.info("Start PDF ingestion: total=%d", total)

    for idx, row in enumerate(rows, start=1):
        logger.info("[%d/%d] Processing URL: %s", idx, total, row.preview_url)
        try:
            doc_id = generate_doc_id(row.preview_url)

            pdf_path = download_pdf(
                url=row.preview_url,
                output_dir=pdf_dir,
                original_name=row.filename or f"{doc_id}.pdf",
            )

            md_output = extract_pdf_to_md(
                input_path=str(pdf_path),
                output_dir=str(markdown_dir),
                cleanup=True,
                keep_images=False,
            )
            if md_output is None:
                raise RuntimeError("MinerU failed to produce markdown")

            md_path = _resolve_markdown_path(Path(md_output), pdf_path)
            if not md_path.exists():
                raise FileNotFoundError(f"Markdown not found: {md_path}")

            markdown_text = md_path.read_text(encoding="utf-8")
            metadata = extractor.extract(markdown_text)

            parsed_json_path = parsed_dir / f"{doc_id}.json"
            save_metadata_json(parsed_json_path, metadata)

            doc = DocumentRecord(
                doc_id=doc_id,
                title=metadata.title,
                authors=metadata.authors,
                keywords=metadata.keywords,
                journal_conference=metadata.journal_conference,
                publish_year=metadata.publish_year,
                abstract=metadata.abstract,
                source_url=row.preview_url,
                source_type=row.module,
                doc_type=metadata.doc_type,
            )
            writer.upsert_document(doc)
            success += 1
        except Exception as exc:
            failed += 1
            logger.exception("Failed processing URL %s: %s", row.preview_url, exc)

    logger.info("PDF ingestion done: success=%d failed=%d total=%d", success, failed, total)
    return {"total": total, "success": success, "failed": failed}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PDF ingestion pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Max number of uploaded_files rows")
    args = parser.parse_args()

    summary = run_pdf_ingestion(limit=args.limit)
    print(summary)
