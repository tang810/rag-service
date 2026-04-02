from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    title: str | None
    authors: str | None
    keywords: list[str]
    journal_conference: str | None
    publish_year: int | None
    abstract: str | None
    doc_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetadataExtractor:
    """Extract document metadata from markdown using existing LLM client."""

    def __init__(self, llm_client: LLMClient | None = None, max_excerpt_chars: int = 9000):
        self.llm_client = llm_client or LLMClient()
        self.max_excerpt_chars = max_excerpt_chars

    def _build_excerpt(self, markdown_text: str) -> str:
        # Use only the beginning section to control token cost (roughly first 2-3 pages).
        text = markdown_text.strip()
        return text[: self.max_excerpt_chars]

    def _build_prompt(self, excerpt: str) -> str:
        return (
            "璇蜂粠浠ヤ笅 Markdown 鏂囨湰涓彁鍙栨枃妗?metadata銆俓\n"
            "瑕佹眰锛歕\n"
            "1) 浠呰繑鍥?JSON锛屼笉瑕佽繑鍥炰换浣曡В閲娿€俓\n"
            "2) 瀛楁蹇呴』鍖呭惈锛歵itle, authors, keywords, journal_conference, publish_year, abstract, doc_type銆俓\n"
            "3) authors銆乯ournal_conference銆乸ublish_year 鍙兘涓嶅瓨鍦紝涓嶅瓨鍦ㄦ椂杩斿洖 null銆俓\n"
            "4) keywords 蹇呴』涓哄瓧绗︿覆鏁扮粍銆俓\n"
            "5) doc_type 鍙栧€间紭鍏? paper, book, report, manual, standard, other銆俓\n"
            "6) publish_year 蹇呴』鏄暣鏁版垨 null銆俓\n\\n"
            "Markdown鍐呭濡備笅锛歕\n"
            f"{excerpt}"
        )

    def _extract_json(self, llm_output: str) -> dict[str, Any]:
        def _repair_invalid_escapes(s: str) -> str:
            # Escape stray backslashes that are not valid JSON escapes.
            return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)

        raw = llm_output.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\\s*", "", raw)
            raw = re.sub(r"\\s*```$", "", raw)

        for candidate in (raw, _repair_invalid_escapes(raw)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found in LLM output", raw, 0)

        body = match.group(0)
        for candidate in (body, _repair_invalid_escapes(body)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise json.JSONDecodeError("Failed to parse metadata JSON", body, 0)

    def _normalize(self, data: dict[str, Any]) -> DocumentMetadata:
        def as_nullable_str(val: Any) -> str | None:
            if val is None:
                return None
            s = str(val).strip()
            return s if s else None

        keywords = data.get("keywords")
        if isinstance(keywords, list):
            keywords_list = [str(x).strip() for x in keywords if str(x).strip()]
        elif isinstance(keywords, str):
            keywords_list = [x.strip() for x in re.split(r"[,;锛岋紱]", keywords) if x.strip()]
        else:
            keywords_list = []

        publish_year = data.get("publish_year")
        if publish_year is not None:
            try:
                publish_year = int(publish_year)
            except (TypeError, ValueError):
                publish_year = None

        doc_type = as_nullable_str(data.get("doc_type"))
        allowed_doc_types = {"paper", "book", "report", "manual", "standard", "other"}
        if doc_type and doc_type not in allowed_doc_types:
            doc_type = "other"

        return DocumentMetadata(
            title=as_nullable_str(data.get("title")),
            authors=as_nullable_str(data.get("authors")),
            keywords=keywords_list,
            journal_conference=as_nullable_str(data.get("journal_conference")),
            publish_year=publish_year,
            abstract=as_nullable_str(data.get("abstract")),
            doc_type=doc_type,
        )

    def extract(self, markdown_text: str) -> DocumentMetadata:
        excerpt = self._build_excerpt(markdown_text)
        prompt = self._build_prompt(excerpt)
        llm_output = self.llm_client.completion_text(
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=1200,
            system_prompt="You are a strict metadata extractor. Output valid JSON only.",
        )
        payload = self._extract_json(llm_output)
        metadata = self._normalize(payload)
        logger.info("Metadata extracted: title=%s, doc_type=%s", metadata.title, metadata.doc_type)
        return metadata


def save_metadata_json(output_path: Path, metadata: DocumentMetadata) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

