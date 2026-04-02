from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import asyncpg

from src.clients.config import db_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


PAGE_PATTERN = re.compile(r"^\s*<!--\s*Page\s+(\d+)\s*-->\s*$", re.IGNORECASE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownChunkProcessor:
    def __init__(
        self,
        markdown_dir: Path | None = None,
        chunk_size: int = 800,
        overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

        root = Path(__file__).resolve().parents[2]
        self.markdown_dir = markdown_dir or (root / "data" / "markdown")
        self._dsn = db_config.url.replace("postgresql+asyncpg://", "postgresql://", 1)

    async def _connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(self._dsn)

    def load_markdown_file(self, markdown_path: Path, doc_id: str | None = None) -> dict[str, Any]:
        path = Path(markdown_path)
        return {
            "doc_id": doc_id or path.stem,
            "path": path,
            "content": path.read_text(encoding="utf-8"),
        }

    def load_markdown_files(self) -> list[dict[str, Any]]:
        if not self.markdown_dir.exists():
            logger.warning("Markdown directory does not exist: %s", self.markdown_dir)
            return []

        markdown_files = sorted(self.markdown_dir.rglob("*.md"))

        documents: list[dict[str, Any]] = []
        for path in markdown_files:
            doc_id = path.stem
            text = path.read_text(encoding="utf-8")
            documents.append({"doc_id": doc_id, "path": path, "content": text})

        return documents

    def build_chunks(self, doc_id: str, markdown_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        parsed_segments = self.parse_markdown(markdown_text)
        raw_chunks = self.split_into_chunks(parsed_segments)

        chunks: list[dict[str, Any]] = []
        for index, chunk in enumerate(raw_chunks):
            chunks.append(
                {
                    "chunk_id": self.generate_chunk_id(doc_id, index),
                    "doc_id": doc_id,
                    "content": chunk["content"],
                    "page": chunk.get("page"),
                    "section_path": chunk.get("section_path", []),
                    "chunk_index": index,
                }
            )

        return parsed_segments, chunks

    def parse_markdown(self, markdown_text: str) -> list[dict[str, Any]]:
        lines = markdown_text.splitlines()

        current_page: int | None = None
        section_stack: list[str] = []
        paragraph_lines: list[str] = []
        segments: list[dict[str, Any]] = []

        def flush_paragraph() -> None:
            if not paragraph_lines:
                return
            text = "\n".join(paragraph_lines).strip()
            paragraph_lines.clear()
            if not text:
                return
            segments.append(
                {
                    "text": text,
                    "page": current_page,
                    "section_path": list(section_stack),
                }
            )

        for line in lines:
            page_match = PAGE_PATTERN.match(line)
            if page_match:
                flush_paragraph()
                current_page = int(page_match.group(1))
                continue

            heading_match = HEADING_PATTERN.match(line)
            if heading_match:
                flush_paragraph()

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip().rstrip("#").strip()

                if level <= len(section_stack):
                    section_stack = section_stack[: level - 1]
                section_stack.append(title)

                segments.append(
                    {
                        "text": title,
                        "page": current_page,
                        "section_path": list(section_stack),
                    }
                )
                continue

            if line.strip() == "":
                flush_paragraph()
                continue

            paragraph_lines.append(line)

        flush_paragraph()
        return segments

    def _split_long_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        step = max(1, self.chunk_size - self.overlap)
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            window = text[start:end]

            if end < text_len:
                split_at = window.rfind("\n")
                if split_at < int(self.chunk_size * 0.5):
                    split_at = window.rfind(" ")
                if split_at > 0:
                    end = start + split_at
                    window = text[start:end]

            cleaned = window.strip()
            if cleaned:
                chunks.append(cleaned)

            if end >= text_len:
                break
            start = max(end - self.overlap, start + step)

        return chunks

    def split_into_chunks(self, parsed_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pieces: list[dict[str, Any]] = []

        for segment in parsed_segments:
            text = str(segment["text"]).strip()
            if not text:
                continue

            if len(text) <= self.chunk_size:
                pieces.append(segment)
                continue

            for part in self._split_long_text(text):
                pieces.append(
                    {
                        "text": part,
                        "page": segment.get("page"),
                        "section_path": segment.get("section_path", []),
                    }
                )

        chunks: list[dict[str, Any]] = []
        current_text = ""
        current_page: int | None = None
        current_section_path: list[str] = []

        for piece in pieces:
            piece_text = piece["text"].strip()
            separator = "\n\n" if current_text else ""
            candidate = f"{current_text}{separator}{piece_text}" if current_text else piece_text

            if len(candidate) <= self.chunk_size:
                if not current_text:
                    current_page = piece.get("page")
                    current_section_path = list(piece.get("section_path") or [])
                current_text = candidate
                continue

            if current_text:
                chunks.append(
                    {
                        "content": current_text,
                        "page": current_page,
                        "section_path": current_section_path,
                    }
                )

            overlap_text = current_text[-self.overlap :].strip() if current_text else ""
            if overlap_text:
                next_text = f"{overlap_text}\n\n{piece_text}".strip()
            else:
                next_text = piece_text

            if len(next_text) > self.chunk_size:
                next_text = next_text[-self.chunk_size :]

            current_text = next_text
            current_page = piece.get("page")
            current_section_path = list(piece.get("section_path") or [])

        if current_text:
            chunks.append(
                {
                    "content": current_text,
                    "page": current_page,
                    "section_path": current_section_path,
                }
            )

        return chunks

    def generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        source = f"{doc_id}_{chunk_index}".encode("utf-8")
        return hashlib.sha256(source).hexdigest()

    async def save_chunks_to_db(self, doc_id: str, chunks: list[dict[str, Any]]) -> int:
        conn = await self._connect()
        try:
            async with conn.transaction():
                await conn.execute("DELETE FROM chunks WHERE doc_id = $1", doc_id)

                for index, chunk in enumerate(chunks):
                    chunk_id = chunk.get("chunk_id") or self.generate_chunk_id(doc_id, index)
                    chunk_index = chunk.get("chunk_index", index)
                    await conn.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id,
                            doc_id,
                            content,
                            page,
                            section_path,
                            chunk_index
                        )
                        VALUES ($1,$2,$3,$4,$5,$6)
                        """,
                        chunk_id,
                        doc_id,
                        chunk["content"],
                        chunk.get("page"),
                        json.dumps(chunk.get("section_path", []), ensure_ascii=False),
                        chunk_index,
                    )
            return len(chunks)
        finally:
            await conn.close()

    async def fetch_chunks_by_doc_id(self, doc_id: str) -> list[asyncpg.Record]:
        conn = await self._connect()
        try:
            return await conn.fetch(
                """
                SELECT chunk_id, doc_id, content, page, section_path, chunk_index
                FROM chunks
                WHERE doc_id = $1
                ORDER BY chunk_index ASC
                """,
                doc_id,
            )
        finally:
            await conn.close()

    async def process_document(self, doc_id: str, markdown_path: Path) -> dict[str, Any]:
        document = self.load_markdown_file(markdown_path, doc_id=doc_id)
        parsed_segments, chunks = self.build_chunks(doc_id=doc_id, markdown_text=document["content"])
        inserted_count = await self.save_chunks_to_db(doc_id=doc_id, chunks=chunks)
        return {
            "doc_id": doc_id,
            "markdown_path": str(document["path"]),
            "parsed_segment_count": len(parsed_segments),
            "chunk_count": len(chunks),
            "inserted_count": inserted_count,
            "chunks": chunks,
        }

    async def process_all_documents(self) -> list[dict[str, Any]]:
        documents = self.load_markdown_files()
        if not documents:
            logger.info("No markdown files found in %s", self.markdown_dir)
            return []

        results: list[dict[str, Any]] = []

        for document in documents:
            doc_id = document["doc_id"]
            logger.info("Processing document: %s", doc_id)

            parsed, chunks = self.build_chunks(doc_id=doc_id, markdown_text=document["content"])

            logger.info("Generated chunks: %d", len(chunks))
            inserted_count = await self.save_chunks_to_db(doc_id, chunks)
            logger.info("Inserted chunks into database")
            results.append(
                {
                    "doc_id": doc_id,
                    "markdown_path": str(document["path"]),
                    "parsed_segment_count": len(parsed),
                    "chunk_count": len(chunks),
                    "inserted_count": inserted_count,
                }
            )

        return results


async def process_all_documents() -> list[dict[str, Any]]:
    processor = MarkdownChunkProcessor()
    return await processor.process_all_documents()


if __name__ == "__main__":
    asyncio.run(process_all_documents())
