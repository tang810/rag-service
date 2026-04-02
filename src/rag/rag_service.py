from __future__ import annotations

import ast
import asyncio
import logging
import re
from difflib import SequenceMatcher
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import quote

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.clients.config import ARTICLE_LINK_BASE_URL
from src.clients.llm_client import LLMClient
from src.rag.schemas import ChunkResult, RAGRequest, SearchRequest
from src.rag.search_service import SearchService

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a rigorous research assistant. Answer based on the provided references only.\n"
    "Always include: (1) title, authors, year, abstract, (2) key conclusions, (3) key formulas, (4) article links.\n"
    "If evidence is insufficient, state uncertainty explicitly."
)

YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
CONCLUSION_HINTS = (
    "conclusion",
    "conclusions",
    "summary",
    "discussion",
    "final remarks",
    "closing remarks",
    "in conclusion",
    "we conclude",
    "overall",
    "结论",
    "总结",
    "结语",
    "小结",
    "归纳",
    "讨论",
    "综上",
    "总之",
    "本文",
)


class RAGService:
    def __init__(self, db: AsyncSession, llm: Optional[LLMClient] = None):
        self.db = db
        self.llm = llm or LLMClient()

    async def generate_stream(self, request: RAGRequest) -> AsyncGenerator[str, None]:
        bundle = await self._prepare_bundle(request)
        if bundle.get("empty"):
            yield "抱歉，未检索到相关资料，无法回答该问题。"
            return

        answer = self._render_strict_answer(
            paper_metadata=bundle.get("paper_metadata", []),
            formulas=bundle.get("formulas", []),
            conclusions=bundle.get("conclusions", []),
            article_links=bundle.get("article_links", []),
        )
        for line in answer.splitlines(keepends=True):
            yield line

    async def generate(self, request: RAGRequest) -> dict:
        bundle = await self._prepare_bundle(request)
        if bundle.get("empty"):
            return {
                "answer": "抱歉，未检索到相关资料，无法回答该问题。",
                "sources": [],
                "paper_metadata": [],
                "formulas": [],
                "conclusions": [],
                "article_links": [],
                "search_strategy": bundle.get("search_strategy", "none"),
                "total_chunks_found": 0,
            }

        chunks: List[ChunkResult] = bundle.get("chunks", [])
        formulas: List[Dict[str, Any]] = bundle.get("formulas", [])
        conclusions: List[Dict[str, Any]] = bundle.get("conclusions", [])
        article_links: List[Dict[str, str]] = bundle.get("article_links", [])
        paper_metadata: List[Dict[str, Any]] = bundle.get("paper_metadata", [])

        answer = self._render_strict_answer(
            paper_metadata=paper_metadata,
            formulas=formulas,
            conclusions=conclusions,
            article_links=article_links,
        )
        papers = self._build_papers_result(
            paper_metadata=paper_metadata,
            formulas=formulas,
            conclusions=conclusions,
            article_links=article_links,
        )

        return {
            "answer": answer,
            "papers": papers,
            "total_papers_found": len(papers),
            "sources": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": self._clean_title(c.title),
                    "authors": self._format_authors(c.authors),
                    "publish_year": c.publish_year,
                    "abstract": c.abstract,
                    "source_url": self._to_public_link(c.source_url or ""),
                    "page": c.page,
                    "section_path": c.section_path,
                    "relevance_score": c.relevance_score,
                    "content_preview": c.content[:200],
                }
                for c in chunks
            ],
            "paper_metadata": paper_metadata,
            "formulas": formulas,
            "conclusions": conclusions,
            "article_links": article_links,
            "search_strategy": bundle.get("search_strategy", ""),
            "total_chunks_found": len(chunks),
        }


    def _build_papers_result(
        self,
        *,
        paper_metadata: List[Dict[str, Any]],
        formulas: List[Dict[str, Any]],
        conclusions: List[Dict[str, Any]],
        article_links: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        formula_map: Dict[str, List[Dict[str, Any]]] = {}
        for f in formulas:
            formula_map.setdefault(str(f.get("doc_id", "")), []).append(f)

        conclusion_map: Dict[str, List[Dict[str, Any]]] = {}
        for c in conclusions:
            conclusion_map.setdefault(str(c.get("doc_id", "")), []).append(c)

        link_map: Dict[str, str] = {}
        for l in article_links:
            doc_id = str(l.get("doc_id", ""))
            if doc_id and doc_id not in link_map:
                link_map[doc_id] = str(l.get("source_url", ""))

        papers: List[Dict[str, Any]] = []
        for m in paper_metadata:
            doc_id = str(m.get("doc_id", ""))
            papers.append(
                {
                    "doc_id": doc_id,
                    "title": self._clean_title(m.get("title")) or "未知",
                    "authors": self._format_authors(m.get("authors")),
                    "publish_year": m.get("publish_year"),
                    "abstract": self._sanitize_extracted_text(m.get("abstract")) or "暂无",
                    "article_link": link_map.get(doc_id) or str(m.get("source_url") or ""),
                    "formulas": formula_map.get(doc_id, []),
                    "conclusions": conclusion_map.get(doc_id, []),
                }
            )
        return papers

    async def _prepare_bundle(self, request: RAGRequest) -> Dict[str, Any]:
        if request.context:
            return {
                "context": request.context,
                "chunks": [],
                "paper_metadata": [],
                "formulas": [],
                "conclusions": [],
                "article_links": [],
                "search_strategy": "pre_built_context",
                "empty": False,
            }

        search_request = SearchRequest(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            search_mode=request.search_mode,
            use_reranker=request.use_reranker,
            filters=request.filters,
        )
        search_response = await SearchService(self.db).search(search_request)
        chunks = search_response.results
        if not chunks:
            return {"empty": True, "search_strategy": search_response.search_strategy}

        doc_ids = sorted({c.doc_id for c in chunks})
        formulas: List[Dict[str, Any]] = []
        conclusions: List[Dict[str, Any]] = []

        if request.include_formulas:
            formulas = await self._fetch_formulas(doc_ids, per_doc_limit=request.formula_top_k)
            formulas = self._dedup_formulas_by_expr(formulas)
            formulas = self._attach_formula_latex(formulas)

        if request.include_conclusions:
            conclusions = await self._fetch_conclusions(doc_ids, per_doc_limit=request.conclusion_top_k)
            conclusions = self._denoise_conclusions(conclusions)
            if not conclusions:
                conclusions = await self._fetch_conclusions_by_section(doc_ids, per_doc_limit=request.conclusion_top_k)
                conclusions = self._denoise_conclusions(conclusions)
            if not conclusions:
                conclusions = await self._fetch_tail_chunks_as_conclusions(doc_ids, per_doc_limit=request.conclusion_top_k)
                conclusions = [
                    c
                    for c in conclusions
                    if self._looks_like_conclusion_text(
                        str(c.get("content") or ""),
                        self._normalize_section_text(c.get("section_path")),
                    )
                ]
                conclusions = self._denoise_conclusions(conclusions)
            if not conclusions:
                # Final fallback: keep tail chunks that are not bibliography-like.
                tail_candidates = await self._fetch_tail_chunks_as_conclusions(doc_ids, per_doc_limit=request.conclusion_top_k)
                tail_candidates = [
                    c
                    for c in tail_candidates
                    if self._is_high_confidence_conclusion(
                        str(c.get("content") or ""),
                        c.get("section_path"),
                    )
                ]
                conclusions = self._denoise_conclusions(tail_candidates)
            if not conclusions:
                # Weak fallback: allow plausible wrap-up text while still excluding references/noise.
                tail_candidates = await self._fetch_tail_chunks_as_conclusions(doc_ids, per_doc_limit=request.conclusion_top_k + 2)
                tail_candidates = [
                    c
                    for c in tail_candidates
                    if self._is_plausible_conclusion_text(
                        str(c.get("content") or ""),
                        c.get("section_path"),
                    )
                ]
                conclusions = self._denoise_conclusions(tail_candidates)
            if not conclusions:
                fallback = []
                for c in chunks:
                    sec = self._normalize_section_text(c.section_path)
                    if any(k in sec for k in ("结论", "总结", "结语", "小结", "归纳", "展望", "conclusion", "summary", "discussion")):
                        fallback.append({
                            "doc_id": c.doc_id,
                            "chunk_id": c.chunk_id,
                            "page": c.page,
                            "section_path": c.section_path,
                            "content": c.content,
                        })
                if fallback:
                    conclusions = self._denoise_conclusions(fallback)
            if not conclusions:
                # Original-only last fallback: pick raw tail chunks (non-bibliography) from source text.
                conclusions = await self._fetch_raw_tail_conclusions(doc_ids, per_doc_limit=1)

        article_links = self._collect_article_links(chunks) if request.include_article_links else []
        paper_metadata = await self._collect_paper_metadata(chunks)
        if request.include_conclusions:
            conclusions = self._collapse_conclusions_by_doc(conclusions)
        context = self._build_context(
            chunks,
            paper_metadata=paper_metadata,
            formulas=formulas,
            conclusions=conclusions,
            article_links=article_links,
        )

        return {
            "context": context,
            "chunks": chunks,
            "paper_metadata": paper_metadata,
            "formulas": formulas,
            "conclusions": conclusions,
            "article_links": article_links,
            "search_strategy": search_response.search_strategy,
            "empty": False,
        }

    async def _fetch_formulas(self, doc_ids: List[str], per_doc_limit: int) -> List[Dict[str, Any]]:
        if not doc_ids:
            return []
        try:
            sql = (
                text(
                    """
                    SELECT doc_id, id, name_zh, expr, page
                    FROM (
                        SELECT
                            f.doc_id,
                            f.id,
                            f.name_zh,
                            f.expr,
                            f.page,
                            row_number() OVER (PARTITION BY f.doc_id ORDER BY f.id) AS rn
                        FROM formulas f
                        WHERE f.doc_id IN :doc_ids
                    ) t
                    WHERE t.rn <= :per_doc_limit
                    ORDER BY t.doc_id, t.rn
                    """
                ).bindparams(bindparam("doc_ids", expanding=True))
            )
            rows = (await self.db.execute(sql, {"doc_ids": doc_ids, "per_doc_limit": per_doc_limit})).mappings().all()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("fetch formulas failed, fallback to no formulas: %s", exc)
            return []

    async def _fetch_conclusions(self, doc_ids: List[str], per_doc_limit: int) -> List[Dict[str, Any]]:
        if not doc_ids:
            return []
        try:
            sql = (
                text(
                    """
                    SELECT doc_id, chunk_id, page, section_path, content
                    FROM (
                        SELECT
                            c.doc_id,
                            c.chunk_id,
                            c.page,
                            c.section_path,
                            c.content,
                            CASE
                                WHEN c.section_path::text ILIKE '%conclusion%'
                                  OR c.section_path::text ILIKE '%conclusions%'
                                  OR c.section_path::text ILIKE '%summary%'
                                  OR c.section_path::text ILIKE '%discussion%'
                                  OR c.section_path::text ILIKE '%结论%'
                                  OR c.section_path::text ILIKE '%总结%'
                                  OR c.section_path::text ILIKE '%讨论%'
                                  OR c.section_path::text ILIKE '%结语%'
                                  OR c.section_path::text ILIKE '%小结%'
                                  OR c.section_path::text ILIKE '%归纳%'
                                THEN 2
                                WHEN c.content ILIKE '%in conclusion%'
                                  OR c.content ILIKE '%we conclude%'
                                  OR c.content ILIKE '%overall%'
                                  OR c.content ILIKE '%conclusion%'
                                  OR c.content ILIKE '%conclusions%'
                                  OR c.content ILIKE '%summary%'
                                  OR c.content ILIKE '%结论%'
                                  OR c.content ILIKE '%总结%'
                                  OR c.content ILIKE '%结语%'
                                  OR c.content ILIKE '%小结%'
                                  OR c.content ILIKE '%归纳%'
                                  OR c.content ILIKE '%综上%'
                                  OR c.content ILIKE '%总之%'
                                THEN 1
                                ELSE 0
                            END AS score,
                            row_number() OVER (
                                PARTITION BY c.doc_id
                                ORDER BY
                                    CASE
                                        WHEN c.section_path::text ILIKE '%conclusion%'
                                          OR c.section_path::text ILIKE '%conclusions%'
                                          OR c.section_path::text ILIKE '%summary%'
                                          OR c.section_path::text ILIKE '%discussion%'
                                          OR c.section_path::text ILIKE '%结论%'
                                          OR c.section_path::text ILIKE '%总结%'
                                          OR c.section_path::text ILIKE '%讨论%'
                                          OR c.section_path::text ILIKE '%结语%'
                                          OR c.section_path::text ILIKE '%小结%'
                                          OR c.section_path::text ILIKE '%归纳%'
                                        THEN 0 ELSE 1
                                    END,
                                    c.chunk_index NULLS LAST,
                                    c.id
                            ) AS rn
                        FROM chunks c
                        WHERE c.doc_id IN :doc_ids
                          AND (
                              c.content ILIKE '%conclusion%'
                              OR c.content ILIKE '%conclusions%'
                              OR c.content ILIKE '%summary%'
                              OR c.content ILIKE '%discussion%'
                              OR c.content ILIKE '%in conclusion%'
                              OR c.content ILIKE '%we conclude%'
                              OR c.content ILIKE '%overall%'
                              OR c.content ILIKE '%结论%'
                              OR c.content ILIKE '%总结%'
                              OR c.content ILIKE '%讨论%'
                              OR c.content ILIKE '%结语%'
                              OR c.content ILIKE '%小结%'
                              OR c.content ILIKE '%归纳%'
                              OR c.content ILIKE '%综上%'
                              OR c.content ILIKE '%总之%'
                              OR c.section_path::text ILIKE '%conclusion%'
                              OR c.section_path::text ILIKE '%conclusions%'
                              OR c.section_path::text ILIKE '%summary%'
                              OR c.section_path::text ILIKE '%discussion%'
                              OR c.section_path::text ILIKE '%结论%'
                              OR c.section_path::text ILIKE '%总结%'
                              OR c.section_path::text ILIKE '%讨论%'
                              OR c.section_path::text ILIKE '%结语%'
                              OR c.section_path::text ILIKE '%小结%'
                              OR c.section_path::text ILIKE '%归纳%'
                          )
                          AND c.content NOT ILIKE '%references%'
                          AND c.content NOT ILIKE '%bibliography%'
                          AND c.section_path::text NOT ILIKE '%references%'
                          AND c.section_path::text NOT ILIKE '%bibliography%'
                    ) t
                    WHERE t.score > 0
                      AND t.rn <= :per_doc_limit
                    ORDER BY t.doc_id, t.rn
                    """
                ).bindparams(bindparam("doc_ids", expanding=True))
            )
            rows = (await self.db.execute(sql, {"doc_ids": doc_ids, "per_doc_limit": per_doc_limit})).mappings().all()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("fetch conclusions failed, fallback to no conclusions: %s", exc)
            return []

    async def _fetch_conclusions_by_section(self, doc_ids: List[str], per_doc_limit: int) -> List[Dict[str, Any]]:
        if not doc_ids:
            return []
        try:
            sql = (
                text(
                    """
                    SELECT doc_id, chunk_id, page, section_path, content
                    FROM (
                        SELECT
                            c.doc_id,
                            c.chunk_id,
                            c.page,
                            c.section_path,
                            c.content,
                            row_number() OVER (
                                PARTITION BY c.doc_id
                                ORDER BY c.chunk_index NULLS LAST, c.id
                            ) AS rn
                        FROM chunks c
                        WHERE c.doc_id IN :doc_ids
                          AND c.content IS NOT NULL
                          AND length(c.content) >= 20
                          AND (
                              c.section_path::text ILIKE '%conclusion%'
                              OR c.section_path::text ILIKE '%conclusions%'
                              OR c.section_path::text ILIKE '%summary%'
                              OR c.section_path::text ILIKE '%discussion%'
                              OR c.section_path::text ILIKE '%结论%'
                              OR c.section_path::text ILIKE '%总结%'
                              OR c.section_path::text ILIKE '%讨论%'
                              OR c.section_path::text ILIKE '%结语%'
                              OR c.section_path::text ILIKE '%小结%'
                              OR c.section_path::text ILIKE '%归纳%'
                              OR c.section_path::text ILIKE '%展望%'
                          )
                          AND c.section_path::text NOT ILIKE '%references%'
                          AND c.section_path::text NOT ILIKE '%bibliography%'
                    ) t
                    WHERE t.rn <= :per_doc_limit
                    ORDER BY t.doc_id, t.rn
                    """
                ).bindparams(bindparam("doc_ids", expanding=True))
            )
            rows = (await self.db.execute(sql, {"doc_ids": doc_ids, "per_doc_limit": per_doc_limit})).mappings().all()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("fetch conclusions by section failed: %s", exc)
            return []

    async def _fetch_tail_chunks_as_conclusions(self, doc_ids: List[str], per_doc_limit: int) -> List[Dict[str, Any]]:
        if not doc_ids:
            return []
        try:
            sql = (
                text(
                    """
                    SELECT doc_id, chunk_id, page, section_path, content
                    FROM (
                        SELECT
                            c.doc_id,
                            c.chunk_id,
                            c.page,
                            c.section_path,
                            c.content,
                            row_number() OVER (
                                PARTITION BY c.doc_id
                                ORDER BY c.chunk_index DESC NULLS LAST, c.id DESC
                            ) AS rn
                        FROM chunks c
                        WHERE c.doc_id IN :doc_ids
                          AND c.content IS NOT NULL
                          AND length(c.content) >= 40
                          AND c.content NOT ILIKE '%references%'
                          AND c.content NOT ILIKE '%bibliography%'
                          AND c.section_path::text NOT ILIKE '%references%'
                          AND c.section_path::text NOT ILIKE '%bibliography%'
                    ) t
                    WHERE t.rn <= :per_doc_limit
                    ORDER BY t.doc_id, t.rn
                    """
                ).bindparams(bindparam("doc_ids", expanding=True))
            )
            rows = (await self.db.execute(sql, {"doc_ids": doc_ids, "per_doc_limit": per_doc_limit})).mappings().all()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("fetch tail chunks as conclusions failed: %s", exc)
            return []

    async def _fetch_raw_tail_conclusions(self, doc_ids: List[str], per_doc_limit: int) -> List[Dict[str, Any]]:
        if not doc_ids:
            return []
        try:
            sql = (
                text(
                    """
                    SELECT doc_id, chunk_id, page, section_path, content
                    FROM (
                        SELECT
                            c.doc_id,
                            c.chunk_id,
                            c.page,
                            c.section_path,
                            c.content,
                            row_number() OVER (
                                PARTITION BY c.doc_id
                                ORDER BY c.chunk_index DESC NULLS LAST, c.id DESC
                            ) AS rn
                        FROM chunks c
                        WHERE c.doc_id IN :doc_ids
                          AND c.content IS NOT NULL
                          AND length(c.content) >= 60
                    ) t
                    WHERE t.rn <= :per_doc_limit
                    ORDER BY t.doc_id, t.rn
                    """
                ).bindparams(bindparam("doc_ids", expanding=True))
            )
            rows = (await self.db.execute(sql, {"doc_ids": doc_ids, "per_doc_limit": per_doc_limit})).mappings().all()
            result: List[Dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                content = str(item.get("content") or "")
                section_path = item.get("section_path")
                # Keep only original tail text that is not bibliography/noise.
                if self._is_noisy_conclusion(content, section_path):
                    continue
                item["source"] = "tail_raw_original"
                item["content"] = self._trim_to_complete_sentence(self._sanitize_extracted_text(content), hard_limit=900)
                if len(str(item["content"])) < 40:
                    continue
                result.append(item)
            return result
        except Exception as exc:
            logger.warning("fetch raw tail conclusions failed: %s", exc)
            return []

    async def _collect_paper_metadata(self, chunks: List[ChunkResult]) -> List[Dict[str, Any]]:
        by_doc: Dict[str, Dict[str, Any]] = {}
        for c in chunks:
            if c.doc_id in by_doc:
                continue
            by_doc[c.doc_id] = {
                "doc_id": c.doc_id,
                "title": self._clean_title(c.title),
                "authors": self._format_authors(c.authors),
                "publish_year": c.publish_year,
                "abstract": self._sanitize_extracted_text(c.abstract),
                "source_url": self._to_public_link(c.source_url or ""),
            }

        await self._backfill_publish_years(by_doc)
        return list(by_doc.values())

    async def _backfill_publish_years(self, by_doc: Dict[str, Dict[str, Any]]) -> None:
        missing_ids = [doc_id for doc_id, m in by_doc.items() if not m.get("publish_year")]
        if not missing_ids:
            return

        try:
            sql = (
                text(
                    """
                    SELECT c.doc_id, c.section_path::text AS section_path_text, c.content
                    FROM chunks c
                    WHERE c.doc_id IN :doc_ids
                      AND c.content ~ '(19|20)[0-9]{2}'
                    ORDER BY c.doc_id, c.chunk_index NULLS LAST, c.id
                    """
                ).bindparams(bindparam("doc_ids", expanding=True))
            )
            rows = (await self.db.execute(sql, {"doc_ids": missing_ids})).mappings().all()

            best: Dict[str, tuple[int, int]] = {}
            for row in rows:
                doc_id = row["doc_id"]
                content = str(row.get("content") or "")
                section = str(row.get("section_path_text") or "")
                for year in self._extract_year_candidates(content):
                    score = self._score_year_signal(content, section, year)
                    current = best.get(doc_id)
                    if current is None or score > current[0] or (score == current[0] and year > current[1]):
                        best[doc_id] = (score, year)

            for doc_id, (_, year) in best.items():
                if by_doc[doc_id].get("publish_year") is None:
                    by_doc[doc_id]["publish_year"] = year
        except Exception as exc:
            logger.warning("publish_year backfill failed: %s", exc)

    def _extract_year_candidates(self, text_content: str) -> List[int]:
        now = datetime.utcnow().year
        years = []
        for m in YEAR_PATTERN.findall(text_content or ""):
            year = int(m)
            if 1900 <= year <= now + 1:
                years.append(year)
        return years

    def _score_year_signal(self, content: str, section: str, year: int) -> int:
        txt = (content or "").lower()
        sec = (section or "").lower()
        score = 0

        if "published online" in txt or "published" in txt:
            score += 8
        if "accepted" in txt:
            score += 6
        if "received" in txt:
            score += 4
        if "copyright" in txt or "©" in content:
            score += 3
        if "data availability" in sec:
            score += 2
        if "references" in sec or "bibliography" in sec:
            score -= 5

        if year >= 2018:
            score += 2
        elif year >= 2000:
            score += 1

        return score

    def _clean_title(self, title: Optional[str]) -> Optional[str]:
        if not title:
            return title
        cleaned = title.strip()
        cleaned = re.sub(r"^\s*OPEN\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    def _format_authors(self, authors: Any) -> str:
        if authors is None:
            return "未知"
        if isinstance(authors, list):
            return ", ".join(str(x) for x in authors if str(x).strip()) or "未知"
        if isinstance(authors, str):
            s = authors.strip()
            if not s:
                return "未知"
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return ", ".join(str(x) for x in parsed if str(x).strip()) or "未知"
                except Exception:
                    pass
            return s
        return str(authors)

    def _to_public_link(self, source_url: str) -> str:
        url = (source_url or "").strip()
        if not url:
            return ""
        if url.startswith("local://") and ARTICLE_LINK_BASE_URL:
            filename = url[len("local://") :].lstrip("/")
            return f"{ARTICLE_LINK_BASE_URL}/{quote(filename)}"
        return url

    def _collect_article_links(self, chunks: List[ChunkResult]) -> List[Dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        links: List[Dict[str, str]] = []
        for c in chunks:
            url = self._to_public_link(c.source_url or "")
            if not url:
                continue
            key = (c.doc_id, url)
            if key in seen:
                continue
            seen.add(key)
            links.append({"doc_id": c.doc_id, "title": self._clean_title(c.title) or "", "source_url": url})
        return links

    def _compact_text(self, text_content: Optional[str], limit: int = 400) -> str:
        txt = re.sub(r"\s+", " ", (text_content or "")).strip()
        if len(txt) <= limit:
            return txt
        return txt[: limit - 3] + "..."

    def _sanitize_extracted_text(self, text_content: Optional[str]) -> str:
        txt = str(text_content or "")
        txt = re.sub(r"<[^>]+>", "", txt)
        txt = txt.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
        txt = txt.replace("&amp;", "&")
        txt = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def _trim_to_complete_sentence(self, text_content: Optional[str], hard_limit: int = 1200) -> str:
        txt = self._sanitize_extracted_text(text_content)
        if len(txt) <= hard_limit:
            return txt

        cut = txt[:hard_limit]
        sentence_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"), cut.rfind("\u3002"), cut.rfind("\uff01"), cut.rfind("\uff1f"))
        if sentence_end >= max(80, hard_limit // 2):
            return cut[: sentence_end + 1].strip()
        return cut.strip()

    def _normalize_expr(self, expr: str) -> str:
        normalized = (expr or "").lower().strip()
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    def _dedup_formulas_by_expr(self, formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        result: List[Dict[str, Any]] = []
        for f in formulas:
            doc_id = str(f.get("doc_id", ""))
            expr_norm = self._normalize_expr(str(f.get("expr") or ""))
            if not expr_norm:
                continue
            key = (doc_id, expr_norm)
            if key in seen:
                continue
            seen.add(key)
            result.append(f)
        return result

    def _escape_latex_text(self, text_content: str) -> str:
        s = text_content or ""
        s = s.replace("\\", r"\textbackslash{}")
        s = s.replace("_", r"\_")
        s = s.replace("{", r"\{").replace("}", r"\}")
        return s

    def _to_latex_expr(self, expr: str) -> str:
        e = (expr or "").strip()
        norm = re.sub(r"\s+", "", e.lower())

        if norm.startswith("r2=") or ("y_bar" in norm and "widehat_y_i" in norm):
            return r"R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}"

        if norm.startswith("mse=") and "abs(" in norm:
            return r"\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i-\hat{y}_i\right|"

        if norm.startswith("mse="):
            return r"\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2"

        if norm.startswith("mae="):
            return r"\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i-\hat{y}_i\right|"

        if norm.startswith("l_mae="):
            return r"L_{\mathrm{MAE}} = \frac{1}{N}\sum_{i=1}^{N}\left|y_i-\hat{y}_i\right|"

        if norm.startswith("l_mse="):
            return r"L_{\mathrm{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2"

        if norm.startswith("l_cs="):
            return r"L_{\mathrm{CS}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)\right]"

        if "alpha_ij" in norm and "sum_kexp" in norm:
            return r"\alpha_{ij} = \frac{\exp(W_a[h_i;h_j])}{\sum_k \exp(W_a[h_i;h_k])}"

        if norm.startswith("w_ij=-log(1+exp(theta_ij))"):
            return r"w_{ij} = -\log\left(1+\exp(\theta_{ij})\right)"

        if norm.startswith("x=b_1*t+b_2*t**(1/2)+x_0"):
            return r"X = B_1 t + B_2 t^{1/2} + X_0"

        if norm.startswith("x=x_0+dt*t**n"):
            return r"X = X_0 + D_t t^n"

        if "relu" in norm and "h_(i+1)" in norm:
            return r"H_{i+1} = \mathrm{ReLU}(W_{i+1}\cdot H_i + b_{i+1})"

        # Fallback: keep readable as plain text in LaTeX block.
        return r"\text{" + self._escape_latex_text(e) + r"}"

    def _attach_formula_latex(self, formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for f in formulas:
            f2 = dict(f)
            f2["expr_latex"] = self._to_latex_expr(str(f2.get("expr") or ""))
            enriched.append(f2)
        return enriched

    def _is_noisy_conclusion(self, content: str, section_path: Any) -> bool:
        c = (content or "").strip().lower()
        sec = self._normalize_section_text(section_path)

        noise_prefixes = (
            "supplementary materials",
            "figure ",
            "fig. ",
            "table ",
            "author contributions",
            "funding",
            "conflicts of interest",
            "availability of data",
        )
        if c.startswith(noise_prefixes):
            return True

        if "supplementary materials" in c:
            return True
        if "figure s" in c or "table s" in c:
            return True
        if self._looks_like_reference_block(c):
            return True
        if "references" in sec or "bibliography" in sec:
            return True
        if re.search(r"(目录|目\s*录|图表清单|参考文献|第[一二三四五六七八九十0-9]+章|第[一二三四五六七八九十0-9]+节)", c):
            return True
        if "keywords:" in c or "关键词" in c:
            return True
        if "....." in c or "……" in c:
            return True
        # Very long text can still be a valid conclusion section in thesis-like documents.
        if len(c) > 12000:
            return True
        if len(c) < 40:
            return not self._looks_like_conclusion_text(c, sec)
        return False

    def _normalize_section_text(self, section_path: Any) -> str:
        if isinstance(section_path, list):
            raw = " ".join(str(x) for x in section_path if str(x).strip())
        else:
            raw = str(section_path or "")
        normalized = raw.lower()
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    def _looks_like_conclusion_text(self, content: str, section_text: str) -> bool:
        c = (content or "").lower()
        sec = (section_text or "").lower()
        if self._looks_like_reference_block(c):
            return False
        return any(k in c or k in sec for k in CONCLUSION_HINTS)

    def _is_high_confidence_conclusion(self, content: str, section_path: Any) -> bool:
        sec = self._normalize_section_text(section_path)
        c = str(content or "").lower().strip()
        if not c:
            return False
        if self._looks_like_reference_block(c):
            return False
        # Section signal is strongest and should dominate.
        if any(k in sec for k in ("conclusion", "summary", "discussion", "结论", "总结", "结语", "小结", "归纳")):
            return True
        # Content signal should include explicit wrap-up cue.
        return any(k in c for k in ("in conclusion", "we conclude", "overall", "结论", "总结", "综上", "总之"))

    def _is_plausible_conclusion_text(self, content: str, section_path: Any) -> bool:
        c = str(content or "").strip()
        cl = c.lower()
        sec = self._normalize_section_text(section_path)
        if not c:
            return False
        if self._is_noisy_conclusion(c, section_path):
            return False
        if self._looks_like_reference_block(cl):
            return False
        if len(c) < 40:
            return False
        if any(k in sec for k in ("conclusion", "summary", "discussion", "结论", "总结", "结语", "小结", "归纳", "展望")):
            return True

        # For thesis-like Chinese text, accept "result/ending" semantics.
        weak_cues = ("最后", "综上", "总之", "结果表明", "验证了", "得到", "形成", "可见", "提出", "建立了", "优化")
        cue_hits = sum(1 for k in weak_cues if k in c)
        if cue_hits >= 2:
            return True

        # English wrap-up style without explicit "conclusion" section.
        if any(k in cl for k in ("this study", "this paper", "results show", "we found", "in summary")):
            return True

        return False

    def _extract_conclusion_from_abstract(self, abstract: Optional[str]) -> str:
        txt = self._sanitize_extracted_text(abstract)
        if not txt or len(txt) < 40:
            return ""

        parts = re.split(r"(?<=[。！？.!?])\s+", txt)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return ""

        cue_words = ("最后", "综上", "总之", "可见", "结果表明", "验证了", "得到", "最终", "提出")
        for p in reversed(parts):
            if any(k in p for k in cue_words):
                return self._trim_to_complete_sentence(p, hard_limit=280)

        candidate = parts[-1]
        if len(candidate) < 20 and len(parts) >= 2:
            candidate = parts[-2] + " " + candidate
        return self._trim_to_complete_sentence(candidate, hard_limit=280)

    def _fill_missing_conclusions_from_abstract(
        self,
        *,
        conclusions: List[Dict[str, Any]],
        paper_metadata: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result = list(conclusions)
        existing_doc_ids = {str(c.get("doc_id") or "") for c in result}
        for m in paper_metadata:
            doc_id = str(m.get("doc_id") or "")
            if not doc_id or doc_id in existing_doc_ids:
                continue
            fallback = self._extract_conclusion_from_abstract(m.get("abstract"))
            if not fallback:
                continue
            result.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": "",
                    "page": None,
                    "section_path": ["abstract_fallback"],
                    "content": fallback,
                    "source": "abstract_fallback",
                }
            )
            existing_doc_ids.add(doc_id)
        return result

    def _collapse_conclusions_by_doc(self, conclusions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for c in conclusions:
            by_doc.setdefault(str(c.get("doc_id") or ""), []).append(c)

        collapsed: List[Dict[str, Any]] = []
        for doc_id, items in by_doc.items():
            if not doc_id or not items:
                continue

            # Prefer real conclusion chunks over abstract fallback.
            primary = [x for x in items if str(x.get("source") or "") != "abstract_fallback"]
            candidates = primary or items

            merged = ""
            for item in candidates:
                txt = self._sanitize_extracted_text(str(item.get("content") or ""))
                if not txt:
                    continue
                if not merged:
                    merged = txt
                elif self._is_near_duplicate_text(merged, txt):
                    continue
                else:
                    merged = self._merge_overlap_text(merged, txt, min_overlap=18)

            merged = self._polish_conclusion_text(self._trim_to_complete_sentence(merged, hard_limit=1200))
            if not merged:
                continue

            first = dict(candidates[0])
            first["content"] = merged
            if len(candidates) > 1:
                first["source"] = "merged"
            collapsed.append(first)

        return collapsed

    def _looks_like_reference_block(self, text_content: str) -> bool:
        txt = str(text_content or "").strip().lower()
        if not txt:
            return False

        if re.search(r"(参考文献|bibliography|references)\s*[:：]?", txt):
            return True

        # If text clearly carries conclusion cues, avoid over-filtering.
        if any(k in txt for k in ("结论", "总结", "结语", "小结", "综上", "总之", "in conclusion", "we conclude")):
            return False

        # Citation-heavy chunks are likely bibliography tails, not conclusions.
        bracket_citations = len(re.findall(r"\[\s*\d{1,4}\s*\]", txt))
        citation_tags = len(re.findall(r"\[[jmdcpr]\]", txt))
        year_hits = len(re.findall(r"\b(19\d{2}|20\d{2})\b", txt))
        source_hits = len(re.findall(r"\b(j\.|m\.|doi|vol\.|no\.|pp\.|springer|elsevier|ieee|acm)\b", txt))
        cn_source_hits = len(re.findall(r"(出版社|学报|大学|硕士学位论文|博士学位论文|第\d+卷|第\d+期)", txt))
        if bracket_citations >= 3 and (year_hits >= 3 or (source_hits + cn_source_hits) >= 3):
            return True
        if citation_tags >= 3 and (year_hits >= 2 or bracket_citations >= 2):
            return True

        semicolon_items = len(re.findall(r"[;；]\s*", txt))
        if semicolon_items >= 8 and year_hits >= 4:
            return True

        return False

    def _denoise_conclusions(self, conclusions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Keep source text, but remove noisy prefixes and overlapping chunk heads.
        by_doc: Dict[str, List[Dict[str, Any]]] = {}

        for c in conclusions:
            content = str(c.get("content") or "")
            section_path = c.get("section_path")
            if self._is_noisy_conclusion(content, section_path):
                continue

            cleaned = self._sanitize_extracted_text(content)
            cleaned = re.sub(r"^\s*[\.\,\;\:\-]+\s*", "", cleaned)
            cleaned = re.sub(r"^\s*[a-zA-Z]\s+(?=[a-z])", "", cleaned)
            cleaned = re.sub(r"^\s*\d+\.\s*", "", cleaned)
            cleaned = re.sub(r"^\s*(conclusions?|results and discussion)\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^\s*(?:onclusions?|lusions?)\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = self._trim_to_complete_sentence(cleaned, hard_limit=1800)
            if len(cleaned) < 20:
                continue
            if len(cleaned) < 40 and not self._looks_like_conclusion_text(cleaned, self._normalize_section_text(section_path)):
                continue

            doc_id = str(c.get("doc_id") or "")
            bucket = by_doc.setdefault(doc_id, [])
            if bucket:
                prev = str(bucket[-1].get("content") or "")
                cleaned = self._strip_overlap_prefix(prev, cleaned)
                if not cleaned:
                    continue
                if self._is_near_duplicate_text(prev, cleaned):
                    continue

            c2 = dict(c)
            c2["content"] = cleaned
            bucket.append(c2)

        flattened: List[Dict[str, Any]] = []
        for items in by_doc.values():
            flattened.extend(items)
        return flattened

    def _strip_overlap_prefix(self, prev_text: str, next_text: str, min_overlap: int = 30) -> str:
        prev = (prev_text or "").strip()
        nxt = (next_text or "").strip()
        if not prev or not nxt:
            return nxt

        pl = prev.lower()
        nl = nxt.lower()
        max_k = min(len(prev), len(nxt), 280)
        for k in range(max_k, min_overlap - 1, -1):
            if pl[-k:] == nl[:k]:
                return nxt[k:].lstrip()

        return nxt

    def _normalize_compare_text(self, text_content: str) -> str:
        txt = re.sub(r"\s+", " ", (text_content or "")).strip().lower()
        txt = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", txt)
        return txt

    def _is_near_duplicate_text(self, a: str, b: str) -> bool:
        na = self._normalize_compare_text(a)
        nb = self._normalize_compare_text(b)
        if not na or not nb:
            return False
        if na in nb or nb in na:
            return True
        return SequenceMatcher(None, na, nb).ratio() >= 0.90

    def _merge_overlap_text(self, prev_text: str, next_text: str, min_overlap: int = 24) -> str:
        a = (prev_text or "").strip()
        b = (next_text or "").strip()
        if not a:
            return b
        if not b:
            return a

        al = a.lower()
        bl = b.lower()
        max_k = min(len(a), len(b), 220)
        for k in range(max_k, min_overlap - 1, -1):
            if al[-k:] == bl[:k]:
                return (a + b[k:]).strip()

        return f"{a} {b}".strip()

    def _polish_conclusion_text(self, text_content: str) -> str:
        txt = re.sub(r"\s+", " ", (text_content or "")).strip()
        txt = re.sub(r"\b([A-Za-z]{2,})\s+\1\b", r"\1", txt, flags=re.IGNORECASE)
        if txt and txt[0].islower():
            txt = txt[0].upper() + txt[1:]
        txt = re.sub(
            r"([.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            txt,
        )
        return txt

    def _render_strict_answer(
        self,
        *,
        paper_metadata: List[Dict[str, Any]],
        formulas: List[Dict[str, Any]],
        conclusions: List[Dict[str, Any]],
        article_links: List[Dict[str, str]],
    ) -> str:
        formula_map: Dict[str, List[Dict[str, Any]]] = {}
        for f in formulas:
            formula_map.setdefault(str(f.get("doc_id", "")), []).append(f)

        conclusion_map: Dict[str, List[Dict[str, Any]]] = {}
        for c in conclusions:
            conclusion_map.setdefault(str(c.get("doc_id", "")), []).append(c)

        link_map: Dict[str, str] = {}
        for l in article_links:
            doc_id = str(l.get("doc_id", ""))
            if doc_id and doc_id not in link_map:
                link_map[doc_id] = str(l.get("source_url", ""))

        lines: List[str] = []
        lines.append("以下按固定模板输出：题目、作者、时间、摘要、公式、结论、文章链接。")
        lines.append("")

        for idx, m in enumerate(paper_metadata, 1):
            doc_id = str(m.get("doc_id", ""))
            title = self._clean_title(m.get("title")) or "未知"
            authors = self._format_authors(m.get("authors"))
            year = m.get("publish_year") if m.get("publish_year") is not None else "未知"
            abstract = self._sanitize_extracted_text(m.get("abstract")) or "暂无"
            link = link_map.get(doc_id) or str(m.get("source_url") or "暂无")

            lines.append(f"【论文{idx}】")
            lines.append(f"题目：{title}")
            lines.append(f"作者：{authors}")
            lines.append(f"时间：{year}")
            lines.append(f"摘要：{abstract}")

            lines.append("公式：")
            doc_formulas = formula_map.get(doc_id, [])
            if not doc_formulas:
                lines.append("1. 暂无")
            else:
                for i, f in enumerate(doc_formulas, 1):
                    name = f.get("name_zh") or f.get("id") or "未命名公式"
                    expr = self._compact_text(str(f.get("expr") or ""), 300)
                    latex = str(f.get("expr_latex") or "")
                    lines.append(f"{i}. {name}: {expr}")
                    if latex:
                        lines.append(f"   LaTeX: $$ {latex} $$")

            lines.append("结论：")
            doc_conclusions = conclusion_map.get(doc_id, [])
            if not doc_conclusions:
                lines.append("1. 暂无")
            else:
                for i, c in enumerate(doc_conclusions, 1):
                    content = self._sanitize_extracted_text(str(c.get("content") or ""))
                    src = str(c.get("source") or "")
                    prefix = "[摘要推断] " if src == "abstract_fallback" else ""
                    lines.append(f"{i}. {prefix}{content}")

            lines.append(f"文章链接：{link}")
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _build_context(
        chunks: List[ChunkResult],
        paper_metadata: Optional[List[Dict[str, Any]]] = None,
        formulas: Optional[List[Dict[str, Any]]] = None,
        conclusions: Optional[List[Dict[str, Any]]] = None,
        article_links: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        parts: List[str] = []

        parts.append("--- References: Retrieved Chunks ---")
        for i, chunk in enumerate(chunks, 1):
            header_parts = [f"[C{i}]"]
            if chunk.title:
                header_parts.append(f"title: {chunk.title}")
            if chunk.page is not None:
                header_parts.append(f"page: {chunk.page}")
            if chunk.section_path:
                path = " > ".join(chunk.section_path) if isinstance(chunk.section_path, list) else str(chunk.section_path)
                header_parts.append(f"section: {path}")
            parts.append(f"{' | '.join(header_parts)}\n{chunk.content}")

        if paper_metadata:
            parts.append("\n--- References: Paper Metadata ---")
            for i, m in enumerate(paper_metadata, 1):
                parts.append(
                    f"[M{i}] doc_id: {m.get('doc_id', '')} | title: {m.get('title', '')}\n"
                    f"authors: {m.get('authors', '')}\n"
                    f"year: {m.get('publish_year', '')}\n"
                    f"abstract: {m.get('abstract', '')}\n"
                    f"link: {m.get('source_url', '')}"
                )

        if formulas:
            parts.append("\n--- References: Formulas ---")
            for i, f in enumerate(formulas, 1):
                formula_id = f.get("id", "")
                name_zh = f.get("name_zh", "")
                expr = f.get("expr", "")
                latex = f.get("expr_latex", "")
                doc_id = f.get("doc_id", "")
                page = f.get("page")
                page_hint = f" | page: {page}" if page is not None else ""
                line = f"[F{i}] doc_id: {doc_id}{page_hint} | id: {formula_id} | name: {name_zh}\n{expr}"
                if latex:
                    line += f"\nlatex: {latex}"
                parts.append(line)

        if conclusions:
            parts.append("\n--- References: Conclusions ---")
            for i, c in enumerate(conclusions, 1):
                doc_id = c.get("doc_id", "")
                page = c.get("page")
                content = str(c.get("content", "")).strip()
                page_hint = f" | page: {page}" if page is not None else ""
                parts.append(f"[K{i}] doc_id: {doc_id}{page_hint}\n{content}")

        if article_links:
            parts.append("\n--- References: Article Links ---")
            for i, link in enumerate(article_links, 1):
                parts.append(
                    f"[L{i}] doc_id: {link.get('doc_id', '')} | title: {link.get('title', '')}\n"
                    f"{link.get('source_url', '')}"
                )

        parts.append("\n--- End References ---")
        return "\n\n".join(parts)

    async def _stream_llm(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        stream = self.llm.completion_stream(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        done = object()

        def _next_token(it: Any):
            try:
                return next(it)
            except StopIteration:
                return done

        loop = asyncio.get_event_loop()
        gen_iter = iter(stream)
        while True:
            token = await loop.run_in_executor(None, _next_token, gen_iter)
            if token is done:
                break
            yield token
