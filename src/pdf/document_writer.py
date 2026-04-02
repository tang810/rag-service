from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import asyncpg

from src.clients.config import db_config

logger = logging.getLogger(__name__)


@dataclass
class UploadedFileRecord:
    filename: str | None
    preview_url: str
    module: str | None


@dataclass
class DocumentRecord:
    doc_id: str
    title: str | None
    authors: str | None
    keywords: list[str]
    journal_conference: str | None
    publish_year: int | None
    abstract: str | None
    source_url: str
    source_type: str | None
    doc_type: str | None


class DocumentWriter:
    """Read uploaded PDF rows and upsert parsed data into documents table."""

    def __init__(self):
        self._dsn = db_config.url.replace("postgresql+asyncpg://", "postgresql://", 1)

    async def _connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(self._dsn)

    async def _fetch_uploaded_pdf_rows_async(self, limit: int | None = None) -> list[UploadedFileRecord]:
        conn = await self._connect()
        try:
            base_sql = (
                "SELECT filename, preview_url, module "
                "FROM uploaded_files "
                "WHERE preview_url IS NOT NULL "
                "AND ("
                "module ILIKE '%pdf%' "
                "OR preview_url ILIKE '%.pdf%'"
                ") "
                "ORDER BY id ASC"
            )
            if limit is not None:
                rows = await conn.fetch(f"{base_sql} LIMIT $1", limit)
            else:
                rows = await conn.fetch(base_sql)

            result = [
                UploadedFileRecord(
                    filename=row.get("filename"),
                    preview_url=row["preview_url"],
                    module=row.get("module"),
                )
                for row in rows
            ]
            logger.info("Fetched %d uploaded PDF rows", len(result))
            return result
        finally:
            await conn.close()

    async def _upsert_document_async(self, doc: DocumentRecord) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO documents (
                    doc_id,
                    title,
                    authors,
                    keywords,
                    journal_conference,
                    publish_year,
                    abstract,
                    source_url,
                    source_type,
                    doc_type
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10
                )
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    keywords = EXCLUDED.keywords,
                    journal_conference = EXCLUDED.journal_conference,
                    publish_year = EXCLUDED.publish_year,
                    abstract = EXCLUDED.abstract,
                    source_url = EXCLUDED.source_url,
                    source_type = EXCLUDED.source_type,
                    doc_type = EXCLUDED.doc_type
                """,
                doc.doc_id,
                doc.title,
                doc.authors,
                doc.keywords,
                doc.journal_conference,
                doc.publish_year,
                doc.abstract,
                doc.source_url,
                doc.source_type,
                doc.doc_type,
            )
            logger.info("Upserted document doc_id=%s", doc.doc_id)
        finally:
            await conn.close()

    def fetch_uploaded_pdf_rows(self, limit: int | None = None) -> list[UploadedFileRecord]:
        return asyncio.run(self._fetch_uploaded_pdf_rows_async(limit=limit))

    def upsert_document(self, doc: DocumentRecord) -> None:
        asyncio.run(self._upsert_document_async(doc))
