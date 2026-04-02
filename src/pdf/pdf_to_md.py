from __future__ import annotations

"""
PDF/image -> Markdown via mineru CLI (fixed HTTP backend).
"""

import asyncio
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader, PdfWriter

from src.clients.MinerUParser import MinerUParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MINERU_URL = "http://www.science42.vip:40093"


@dataclass
class MinerURunConfig:
    input_path: str
    output_dir: str
    docker_url: Optional[str] = None
    cleanup: bool = True
    keep_images: bool = False

    @classmethod
    def from_args(
        cls,
        input_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        docker_url: Optional[str] = None,
        cleanup: bool = True,
        keep_images: bool = False,
    ) -> "MinerURunConfig":
        if not input_path:
            raise ValueError("input_path is required")
        if not output_dir:
            raise ValueError("output_dir is required")

        resolved_output = str(Path(output_dir).resolve())
        resolved_input = str(Path(input_path).resolve())
        url = docker_url or DEFAULT_MINERU_URL

        return cls(
            input_path=resolved_input,
            output_dir=resolved_output,
            docker_url=url,
            cleanup=cleanup,
            keep_images=keep_images,
        )


def _sanitize_stem(stem: str) -> str:
    """Match MinerUParser filename sanitization for directory naming."""
    cleaned = re.sub(r"[^\w\-]", "_", stem)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or stem


def _cleanup_keep_only_md(parent_dir: Path, md_filename: str) -> None:
    """Remove everything under parent_dir except md_filename."""
    if not parent_dir.exists() or not parent_dir.is_dir():
        return

    for child in parent_dir.iterdir():
        if child.name == md_filename:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.warning("Cleanup failed for %s: %s", child, e)


async def parse_pdf_to_markdown_with_page(pdf_path: str, docker_url: Optional[str] = None) -> str:
    """
    在 PDF 转 Markdown 过程中注入页码标记（<!-- Page N -->），用于后续 chunk 切分阶段提取物理页码，实现 RAG 检索结果的可定位性。
    """

    parser = MinerUParser(docker_url=docker_url or DEFAULT_MINERU_URL)
    reader = PdfReader(pdf_path)

    page_sections: list[str] = []

    with tempfile.TemporaryDirectory(prefix="mineru_pages_") as tmp_dir:
        tmp_root = Path(tmp_dir)

        for page_idx, page in enumerate(reader.pages):
            page_no = page_idx + 1
            logger.info("Processing page %d", page_no)

            page_pdf_path = tmp_root / f"page_{page_no}.pdf"
            page_output_dir = tmp_root / f"out_{page_no}"

            writer = PdfWriter()
            writer.add_page(page)
            with page_pdf_path.open("wb") as f:
                writer.write(f)

            marker = f"<!-- Page {page_no} -->"
            try:
                success, page_md, _ = await parser.parse_file(
                    file_path=str(page_pdf_path),
                    output_dir=str(page_output_dir),
                    cleanup=True,
                    keep_images=False,
                )
                if not success:
                    page_sections.append(f"{marker}\n[Page parse error]")
                    continue

                normalized_page_md = (page_md or "").strip()
                page_sections.append(f"{marker}\n{normalized_page_md}" if normalized_page_md else marker)
            except Exception:
                logger.exception("Page %d parse failed", page_no)
                page_sections.append(f"{marker}\n[Page parse error]")

    if not page_sections:
        return ""

    return "\n\n".join(page_sections).rstrip() + "\n"


async def extract_pdf_to_md_async(
    input_path: str,
    output_dir: str,
    docker_url: Optional[str] = None,
    cleanup: bool = True,
    keep_images: bool = False,
) -> Optional[Path]:
    """
    Use MinerUParser to convert PDF/image -> Markdown (async wrapper).
    Returns the markdown path or None.
    """

    cfg = MinerURunConfig.from_args(
        input_path=input_path,
        output_dir=output_dir,
        docker_url=docker_url,
        cleanup=cleanup,
        keep_images=keep_images,
    )

    md_content = await parse_pdf_to_markdown_with_page(
        pdf_path=cfg.input_path,
        docker_url=cfg.docker_url,
    )
    if not md_content:
        logger.error("MinerU parse failed")
        return None

    raw_stem = Path(cfg.input_path).stem
    pdf_stem = _sanitize_stem(raw_stem)
    out_md_dir = Path(cfg.output_dir) / pdf_stem
    out_md_path = out_md_dir / f"{pdf_stem}.md"
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(md_content, encoding="utf-8")

    logger.info("Markdown saved to %s", out_md_path)

    if cfg.cleanup:
        _cleanup_keep_only_md(out_md_dir, out_md_path.name)
    return out_md_path


def extract_pdf_to_md(
    input_path: str,
    output_dir: str,
    docker_url: Optional[str] = None,
    cleanup: bool = True,
    keep_images: bool = False,
) -> Optional[Path]:
    """
    Sync wrapper. If an event loop exists, use extract_pdf_to_md_async.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            extract_pdf_to_md_async(
                input_path=input_path,
                output_dir=output_dir,
                docker_url=docker_url,
                cleanup=cleanup,
                keep_images=keep_images,
            )
        )
    else:
        logger.error("Detected running event loop; use extract_pdf_to_md_async")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MinerU CLI wrapper: PDF/image -> Markdown")
    parser.add_argument("input_path", help="PDF/image path")
    parser.add_argument("output_dir", help="Markdown output directory")
    parser.add_argument(
        "--docker-url",
        dest="docker_url",
        default=DEFAULT_MINERU_URL,
        help=f"MinerU API URL (default: {DEFAULT_MINERU_URL})",
    )
    parser.add_argument(
        "--keep-artifacts",
        dest="cleanup",
        action="store_false",
        help="Keep mineru artifacts (default: only keep the final .md)",
    )
    parser.add_argument(
        "--keep-images",
        dest="keep_images",
        action="store_true",
        help="Keep images if artifacts are kept",
    )

    args = parser.parse_args()

    md_path = extract_pdf_to_md(
        input_path=args.input_path,
        output_dir=args.output_dir,
        docker_url=args.docker_url,
        cleanup=args.cleanup,
        keep_images=args.keep_images,
    )

    if md_path:
        print(f"Markdown saved to: {md_path}")
    else:
        print("MinerU parse failed; check logs")
