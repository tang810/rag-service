from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

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


def ensure_data_dirs(project_root: Path) -> tuple[Path, Path, Path]:
	data_root = project_root / "data"
	pdf_dir = data_root / "pdf"
	markdown_dir = data_root / "markdown"
	parsed_dir = data_root / "parsed"

	pdf_dir.mkdir(parents=True, exist_ok=True)
	markdown_dir.mkdir(parents=True, exist_ok=True)
	parsed_dir.mkdir(parents=True, exist_ok=True)

	return pdf_dir, markdown_dir, parsed_dir


def run_test(write_db: bool = True) -> dict:
	project_root = Path(__file__).resolve().parent
	pdf_dir, markdown_dir, parsed_dir = ensure_data_dirs(project_root)

	logger.info("[1/6] 生成 doc_id")
	doc_id = generate_doc_id(TEST_INPUT.preview_url)

	logger.info("[2/6] 下载 PDF")
	pdf_path = download_pdf(
		url=TEST_INPUT.preview_url,
		output_dir=pdf_dir,
		original_name=TEST_INPUT.filename,
	)

	logger.info("[3/6] MinerU 解析 Markdown")
	md_path = extract_pdf_to_md(
		input_path=str(pdf_path),
		output_dir=str(markdown_dir),
		cleanup=True,
		keep_images=False,
	)
	if md_path is None or not Path(md_path).exists():
		raise RuntimeError("Markdown 解析失败，未找到输出文件")

	logger.info("[4/6] LLM 提取 metadata")
	markdown_text = Path(md_path).read_text(encoding="utf-8")
	metadata = MetadataExtractor().extract(markdown_text)

	logger.info("[5/6] 保存 metadata JSON")
	metadata_json_path = parsed_dir / f"{doc_id}.json"
	save_metadata_json(metadata_json_path, metadata)

	db_written = False
	if write_db:
		logger.info("[6/6] 写入 documents 表")
		document = DocumentRecord(
			doc_id=doc_id,
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
		db_written = True

	result = {
		"input": asdict(TEST_INPUT),
		"doc_id": doc_id,
		"pdf_path": str(pdf_path),
		"markdown_path": str(md_path),
		"metadata_json_path": str(metadata_json_path),
		"metadata": metadata.to_dict(),
		"db_written": db_written,
	}
	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Single-case test for PDF ingestion pipeline")
	parser.add_argument("--skip-db", action="store_true", help="Only test parsing; skip documents upsert")
	args = parser.parse_args()

	output = run_test(write_db=not args.skip_db)
	print(json.dumps(output, ensure_ascii=False, indent=2))

