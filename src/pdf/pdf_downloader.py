from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-.]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "document.pdf"


def _derive_filename(url: str, original_name: str | None = None) -> str:
    if original_name:
        base = _sanitize_filename(original_name)
        if not base.lower().endswith(".pdf"):
            base = f"{base}.pdf"
        return base

    parsed = urlparse(url)
    tail = Path(parsed.path).name or "document.pdf"
    tail = _sanitize_filename(tail)
    if not tail.lower().endswith(".pdf"):
        tail = f"{tail}.pdf"
    return tail


def download_pdf(url: str, output_dir: Path, original_name: str | None = None, timeout: int = 120) -> Path:
    """Download PDF from URL and save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = _derive_filename(url, original_name)
    target_path = output_dir / file_name

    logger.info("Downloading PDF: %s", url)
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" not in content_type and target_path.suffix.lower() != ".pdf":
        target_path = target_path.with_suffix(".pdf")

    with target_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logger.info("PDF saved: %s", target_path)
    return target_path
