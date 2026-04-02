from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.clients.config import RRF_K, TEST_MODE
from src.clients.embedding_client import get_embeddings
from src.clients.reranker_client import RerankerClient, RerankerUnavailableError
from src.rag.models import SearchLog
from src.rag.schemas import ChunkResult, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def search(self, request: SearchRequest) -> SearchResponse:
        mode = request.search_mode

        if mode == "embedding":
            return await self._embedding_search(request)
        if mode == "bm25":
            return await self._bm25_search(request)
        if mode == "hybrid":
            return await self._hybrid_search(request)
        if mode == "adaptive":
            return await self._adaptive_search(request)
        raise ValueError(f"unsupported search mode: {mode}")

    async def _embedding_search(self, request: SearchRequest) -> SearchResponse:
        start = time.time()

        emb_start = time.time()
        query_embedding = await self._encode_query(request.query)
        emb_time = (time.time() - emb_start) * 1000

        vec_start = time.time()
        raw_results = await self._vector_search(
            query_embedding=query_embedding,
            limit=request.top_k,
            min_score=request.min_score,
            filters=request.filters,
        )
        vec_time = (time.time() - vec_start) * 1000

        results = [self._build_chunk_result(row, vector_distance=dist) for row, dist in raw_results]

        total_time = (time.time() - start) * 1000
        await self._log_search(
            request,
            "embedding",
            total_time,
            len(results),
            embedding_time=emb_time,
            vector_search_time=vec_time,
        )

        return SearchResponse(
            results=results,
            total_found=len(results),
            processing_time_ms=total_time,
            search_strategy="embedding_only",
            query_embedding_time_ms=emb_time,
            vector_search_time_ms=vec_time,
        )

    async def _vector_search(
        self,
        query_embedding: List[float],
        limit: int,
        min_score: float = 0.0,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[dict, float]]:
        where_clauses = []
        params: Dict[str, object] = {
            "query_embedding": str(query_embedding),
            "limit": limit,
        }

        if min_score > 0:
            where_clauses.append("(e.embedding <=> :query_embedding) <= :max_distance")
            params["max_distance"] = 1.0 - min_score

        if filters:
            if "doc_id" in filters:
                where_clauses.append("d.doc_id = :filter_doc_id")
                params["filter_doc_id"] = filters["doc_id"]
            if "source_type" in filters:
                where_clauses.append("d.source_type = :filter_source_type")
                params["filter_source_type"] = filters["source_type"]
            if "publish_year_from" in filters:
                where_clauses.append("d.publish_year >= :filter_year_from")
                params["filter_year_from"] = filters["publish_year_from"]
            if "publish_year_to" in filters:
                where_clauses.append("d.publish_year <= :filter_year_to")
                params["filter_year_to"] = filters["publish_year_to"]
            if "keywords" in filters:
                where_clauses.append("d.keywords && :filter_keywords")
                params["filter_keywords"] = filters["keywords"]

        where_sql = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""

        sql = text(
            f"""
            SELECT
                c.chunk_id,
                c.content,
                c.page,
                c.section_path,
                c.chunk_index,
                d.doc_id,
                d.title,
                d.authors,
                d.source_type,
                d.publish_year,
                d.source_url,
                d.abstract,
                (e.embedding <=> :query_embedding) AS distance
            FROM embeddings e
            JOIN chunks c ON e.chunk_id = c.chunk_id
            JOIN documents d ON e.doc_id = d.doc_id
            WHERE 1=1 {where_sql}
            ORDER BY e.embedding <=> :query_embedding
            LIMIT :limit
            """
        )

        result = await self.db.execute(sql, params)
        rows = result.mappings().all()
        return [(dict(row), row["distance"]) for row in rows]

    async def _bm25_search(self, request: SearchRequest) -> SearchResponse:
        start = time.time()

        bm25_start = time.time()
        raw_results = await self._text_search(
            query=request.query,
            limit=request.top_k,
            filters=request.filters,
        )
        bm25_time = (time.time() - bm25_start) * 1000

        results = [self._build_chunk_result(row, bm25_score=row["bm25_rank"]) for row in raw_results]

        total_time = (time.time() - start) * 1000
        await self._log_search(request, "bm25", total_time, len(results), bm25_time=bm25_time)

        return SearchResponse(
            results=results,
            total_found=len(results),
            processing_time_ms=total_time,
            search_strategy="bm25_only",
            bm25_search_time_ms=bm25_time,
        )

    async def _text_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict] = None,
    ) -> List[dict]:
        where_clauses = []
        params: Dict[str, object] = {
            "query": query,
            "fuzzy_query": f"%{query}%",
            "limit": limit,
        }

        if filters:
            if "doc_id" in filters:
                where_clauses.append("d.doc_id = :filter_doc_id")
                params["filter_doc_id"] = filters["doc_id"]
            if "source_type" in filters:
                where_clauses.append("d.source_type = :filter_source_type")
                params["filter_source_type"] = filters["source_type"]
            if "publish_year_from" in filters:
                where_clauses.append("d.publish_year >= :filter_year_from")
                params["filter_year_from"] = filters["publish_year_from"]
            if "publish_year_to" in filters:
                where_clauses.append("d.publish_year <= :filter_year_to")
                params["filter_year_to"] = filters["publish_year_to"]

        extra_where = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""

        sql = text(
            f"""
            SELECT
                c.chunk_id,
                c.content,
                c.page,
                c.section_path,
                c.chunk_index,
                d.doc_id,
                d.title,
                d.authors,
                d.source_type,
                d.publish_year,
                d.source_url,
                d.abstract,
                ts_rank(
                    to_tsvector('english', c.content),
                    plainto_tsquery('english', :query)
                ) AS bm25_rank
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE (
                to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)
                OR c.content ILIKE :fuzzy_query
            )
            {extra_where}
            ORDER BY bm25_rank DESC
            LIMIT :limit
            """
        )

        result = await self.db.execute(sql, params)
        rows = result.mappings().all()
        return [dict(row) for row in rows]

    async def _hybrid_search(self, request: SearchRequest) -> SearchResponse:
        start = time.time()

        emb_start = time.time()
        query_embedding = await self._encode_query(request.query)
        emb_time = (time.time() - emb_start) * 1000

        vec_start = time.time()
        vector_results = await self._vector_search(
            query_embedding=query_embedding,
            limit=request.rerank_top_k,
            min_score=request.min_score,
            filters=request.filters,
        )
        vec_time = (time.time() - vec_start) * 1000

        bm25_start = time.time()
        bm25_results = await self._text_search(
            query=request.query,
            limit=request.rerank_top_k,
            filters=request.filters,
        )
        bm25_time = (time.time() - bm25_start) * 1000

        merged = self._rrf_merge(vector_results, bm25_results, k=RRF_K)

        if not merged:
            return SearchResponse(
                results=[],
                total_found=0,
                processing_time_ms=(time.time() - start) * 1000,
                search_strategy="hybrid",
                query_embedding_time_ms=emb_time,
                vector_search_time_ms=vec_time,
                bm25_search_time_ms=bm25_time,
            )

        rerank_time = None
        if request.use_reranker:
            rerank_start = time.time()
            merged = await self._rerank_chunks(request.query, merged)
            rerank_time = (time.time() - rerank_start) * 1000

        final = merged[:request.top_k]
        results = [
            self._build_chunk_result(
                row,
                vector_distance=row.get("_vector_distance"),
                bm25_score=row.get("_bm25_score"),
                rerank_score=row.get("_rerank_score"),
            )
            for row in final
        ]

        total_time = (time.time() - start) * 1000
        await self._log_search(
            request,
            "hybrid",
            total_time,
            len(results),
            embedding_time=emb_time,
            vector_search_time=vec_time,
            bm25_time=bm25_time,
            rerank_time=rerank_time,
        )

        return SearchResponse(
            results=results,
            total_found=len(results),
            processing_time_ms=total_time,
            search_strategy="hybrid",
            query_embedding_time_ms=emb_time,
            vector_search_time_ms=vec_time,
            bm25_search_time_ms=bm25_time,
            rerank_time_ms=rerank_time,
        )

    def _rrf_merge(
        self,
        vector_results: List[Tuple[dict, float]],
        bm25_results: List[dict],
        k: int = 60,
    ) -> List[dict]:
        rrf_scores: Dict[str, float] = {}
        row_map: Dict[str, dict] = {}

        for rank, (row, distance) in enumerate(vector_results, start=1):
            cid = row["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in row_map:
                row_map[cid] = {**row, "_vector_distance": distance}
            else:
                row_map[cid]["_vector_distance"] = distance

        for rank, row in enumerate(bm25_results, start=1):
            cid = row["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in row_map:
                row_map[cid] = {**row, "_bm25_score": row.get("bm25_rank", 0.0)}
            else:
                row_map[cid]["_bm25_score"] = row.get("bm25_rank", 0.0)

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        result = []
        for cid in sorted_ids:
            row = row_map[cid]
            row["_rrf_score"] = rrf_scores[cid]
            result.append(row)

        return result

    async def _rerank_chunks(self, query: str, candidates: List[dict]) -> List[dict]:
        rerank_texts = []
        for row in candidates:
            parts = []
            title = row.get("title")
            if title:
                parts.append(f"[title] {title}")

            section_path = row.get("section_path")
            if section_path:
                if isinstance(section_path, list):
                    path_str = " > ".join(section_path)
                else:
                    path_str = str(section_path)
                parts.append(f"[section] {path_str}")

            parts.append(row["content"])
            rerank_texts.append("\n".join(parts))

        try:
            async with RerankerClient() as client:
                result = await client.rerank(query, rerank_texts)

            scores = result.get("scores", [])
            for idx, score in enumerate(scores):
                if idx < len(candidates):
                    candidates[idx]["_rerank_score"] = score

            candidates.sort(key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
        except RerankerUnavailableError as exc:
            logger.warning("reranker unavailable, fallback to RRF ordering: %s", exc)

        return candidates

    async def _adaptive_search(self, request: SearchRequest) -> SearchResponse:
        import re

        query = request.query.strip()
        word_count = len(query.split())

        has_exact_pattern = bool(
            re.search(
                r"\b\d{4}\.\d{4,5}\b|eq\s*\(\d+\)|table\s+\d+|figure\s+\d+",
                query,
                re.IGNORECASE,
            )
        )

        if has_exact_pattern:
            return await self._bm25_search(request)
        if word_count <= 3:
            return await self._embedding_search(request)

        if word_count >= 10:
            request.use_reranker = True
        return await self._hybrid_search(request)

    async def _encode_query(self, query: str) -> List[float]:
        if TEST_MODE:
            import random

            random.seed(hashlib.md5(query.encode()).hexdigest())
            vec = [random.gauss(0, 1) for _ in range(1024)]
            norm = sum(x * x for x in vec) ** 0.5
            logger.info("[TEST_MODE] use synthetic embedding vector")
            return [x / norm for x in vec]

        embeddings = await asyncio.to_thread(get_embeddings, [query], 16)
        return embeddings[0]

    def _build_chunk_result(
        self,
        row: dict,
        vector_distance: Optional[float] = None,
        bm25_score: Optional[float] = None,
        rerank_score: Optional[float] = None,
    ) -> ChunkResult:
        relevance = 0.0
        if rerank_score is not None:
            relevance = rerank_score
        elif vector_distance is not None:
            relevance = max(0.0, 1.0 - vector_distance)
        elif bm25_score is not None:
            relevance = bm25_score

        return ChunkResult(
            chunk_id=row["chunk_id"],
            content=row["content"],
            page=row.get("page"),
            section_path=row.get("section_path"),
            chunk_index=row.get("chunk_index"),
            doc_id=row["doc_id"],
            title=row.get("title"),
            authors=row.get("authors"),
            source_type=row.get("source_type"),
            publish_year=row.get("publish_year"),
            source_url=row.get("source_url"),
            abstract=row.get("abstract"),
            relevance_score=relevance,
            vector_distance=vector_distance,
            bm25_score=bm25_score,
            rerank_score=rerank_score,
        )

    async def _log_search(
        self,
        request: SearchRequest,
        search_type: str,
        processing_time: float,
        results_count: int,
        embedding_time: Optional[float] = None,
        vector_search_time: Optional[float] = None,
        bm25_time: Optional[float] = None,
        rerank_time: Optional[float] = None,
    ):
        try:
            log = SearchLog(
                query_text=request.query,
                query_hash=hashlib.md5(request.query.encode()).hexdigest(),
                search_type=search_type,
                processing_time_ms=processing_time,
                results_count=results_count,
                top_k=request.top_k,
                embedding_time_ms=embedding_time,
                vector_search_time_ms=vector_search_time,
                rerank_time_ms=rerank_time,
            )
            self.db.add(log)
            await self.db.commit()
        except Exception as exc:
            logger.warning("search log insert failed: %s", exc)
            await self.db.rollback()
