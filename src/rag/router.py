from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.rag.database import get_db
from src.rag.rag_service import RAGService
from src.rag.schemas import RAGRequest, SearchRequest, SearchResponse
from src.rag.search_service import SearchService


router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, db: AsyncSession = Depends(get_db)):
    try:
        service = SearchService(db)
        return await service.search(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"search service error: {exc}") from exc

# ── 新增：RAG 问答接口（SSE 流式） ──

@router.post("/chat")
async def chat_stream(request: RAGRequest, db: AsyncSession = Depends(get_db)):
    """
    流式 RAG 问答接口。

    调用示例：
        POST /api/v1/chat
        {
            "query": "磁性齿轮的工作原理是什么？",
            "search_mode": "hybrid",
            "top_k": 5,
            "system_prompt": "你是飞行器领域专家..."  // 可选，不传用默认
        }

    返回 SSE 流：
        data: {"token": "磁"}
        data: {"token": "性"}
        data: {"token": "齿轮"}
        ...
        data: {"done": true}
    """

    async def event_generator():
        try:
            service = RAGService(db)
            async for token in service.generate_stream(request):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 新增：RAG 问答接口（非流式，方便调试） ──

@router.post("/chat/sync")
async def chat_sync(request: RAGRequest, db: AsyncSession = Depends(get_db)):
    """非流式版本，一次性返回完整回答 + 引用来源，方便测试。"""
    try:
        service = RAGService(db)
        return await service.generate(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"rag service error: {exc}") from exc

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
