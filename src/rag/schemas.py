from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="user query")
    top_k: int = Field(default=10, ge=1, le=100, description="result size")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="minimum relevance")

    rerank_top_k: int = Field(default=50, ge=1, le=200, description="candidate size for reranking")
    use_reranker: bool = Field(default=True, description="enable reranker")

    filters: Optional[Dict[str, Any]] = Field(default=None, description="metadata filters")
    search_mode: str = Field(
        default="hybrid",
        description="search mode: embedding / bm25 / hybrid / adaptive",
    )


class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    page: Optional[int] = None
    section_path: Optional[List[str]] = None
    chunk_index: Optional[int] = None

    doc_id: str
    title: Optional[str] = None
    authors: Optional[str] = None
    source_type: Optional[str] = None
    publish_year: Optional[int] = None
    source_url: Optional[str] = None
    abstract: Optional[str] = None

    relevance_score: float = 0.0
    vector_distance: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[ChunkResult] = Field(default_factory=list)
    total_found: int = 0

    processing_time_ms: float = 0.0
    search_strategy: str = ""
    query_embedding_time_ms: Optional[float] = None
    vector_search_time_ms: Optional[float] = None
    bm25_search_time_ms: Optional[float] = None
    rerank_time_ms: Optional[float] = None


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, description="user query")

    top_k: int = Field(default=5, ge=1, le=50, description="retrieved chunk size")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    search_mode: str = Field(default="hybrid")
    use_reranker: bool = Field(default=True)
    filters: Optional[Dict[str, Any]] = Field(default=None)
    context: Optional[str] = Field(
        default=None,
        description="If provided, skip retrieval and directly use this context",
    )

    include_formulas: bool = Field(default=True, description="include formulas in context and response")
    include_conclusions: bool = Field(default=True, description="include conclusion chunks in context and response")
    include_article_links: bool = Field(default=True, description="include source links in context and response")
    formula_top_k: int = Field(default=6, ge=1, le=30, description="max formulas per document")
    conclusion_top_k: int = Field(default=3, ge=1, le=20, description="max conclusion chunks per document")

    system_prompt: Optional[str] = Field(default=None, description="custom system prompt")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=1, le=16384)
