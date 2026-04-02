from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.clients.config import API_HOST, API_PORT, PDF_SERVE_DIR
from src.rag.database import engine
from src.rag.router import router as rag_router
from src.service.pdf_chunks_embedding_extract_service import (
    DEFAULT_INPUT,
    InputRecord,
    run_pdf_chunks_embeddiing_extract,
)


class PdfChunksEmbeddingExtractRequest(BaseModel):
    filename: str = Field(default=DEFAULT_INPUT.filename, description="PDF filename")
    preview_url: str = Field(default=DEFAULT_INPUT.preview_url, description="PDF URL")
    module: str = Field(default=DEFAULT_INPUT.module, description="Source module")

    write_pdf_db: bool = Field(default=True, description="Write document metadata into documents table")
    write_chunks_db: bool = Field(default=True, description="Write chunks into chunks table")
    write_embeddings_db: bool = Field(default=True, description="Write embeddings into embeddings table")
    run_extract_if_aircraft: bool = Field(default=True, description="Run extract stage when source type is aircraft")
    embedding_batch_size: int = Field(default=32, ge=1, le=512, description="Embedding API batch size")
    embedding_limit: int = Field(default=1000, ge=1, le=200000, description="Fetch limit per embedding loop")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("KnowledgeRetrievalEngine API started")
    print(f"API docs: http://{API_HOST}:{API_PORT}/docs")
    print(f"Health: http://{API_HOST}:{API_PORT}/api/v1/health")
    print(f"File route: http://{API_HOST}:{API_PORT}/files/<pdf_filename>")
    print(f"File dir: {Path(PDF_SERVE_DIR).resolve()}")
    print("=" * 50)
    yield
    await engine.dispose()
    print("KnowledgeRetrievalEngine API stopped")


app = FastAPI(
    title="KnowledgeRetrievalEngine API",
    description="Unified API for RAG search and retrieval services",
    version="1.0.0",
    lifespan=lifespan,
)

serve_dir = Path(PDF_SERVE_DIR).resolve()
serve_dir.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(serve_dir)), name="files")

app.include_router(rag_router)


@app.post("/api/v1/pdf-chunks-embedding-extract", tags=["pipeline"])
async def pdf_chunks_embedding_extract(request: PdfChunksEmbeddingExtractRequest):
    try:
        input_record = InputRecord(
            filename=request.filename,
            preview_url=request.preview_url,
            module=request.module,
        )
        result = await run_in_threadpool(
            run_pdf_chunks_embeddiing_extract,
            input_record,
            request.write_pdf_db,
            request.write_chunks_db,
            request.write_embeddings_db,
            request.run_extract_if_aircraft,
            request.embedding_batch_size,
            request.embedding_limit,
        )
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pipeline service error: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
