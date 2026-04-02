from __future__ import annotations

import logging
from typing import Dict, List, Optional

import aiohttp

from src.clients.config import RERANKER_BASE_URL, RERANKER_TIMEOUT

logger = logging.getLogger(__name__)

MAX_DOCUMENTS = 1000


class RerankerUnavailableError(Exception):
    """Raised when reranker API call fails or returns invalid data."""


class RerankerClient:
    """Async client for Cross-Encoder reranking service."""

    def __init__(self, base_url: str = RERANKER_BASE_URL, timeout: int = RERANKER_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, *exc):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> Dict:
        if not documents:
            return {"scores": []}

        if len(documents) > MAX_DOCUMENTS:
            logger.warning("candidate size %s exceeds max %s, truncating", len(documents), MAX_DOCUMENTS)
            documents = documents[:MAX_DOCUMENTS]

        payload: Dict[str, object] = {
            "query": query,
            "documents": documents,
        }
        if top_k is not None:
            payload["top_k"] = top_k

        if self._session is None:
            raise RerankerUnavailableError("Reranker client session is not initialized")

        try:
            async with self._session.post(f"{self.base_url}/rerank", json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RerankerUnavailableError(
                        f"Reranker service returned status={resp.status}: {error_text}"
                    )
                return await resp.json()
        except Exception as exc:
            if isinstance(exc, RerankerUnavailableError):
                raise
            raise RerankerUnavailableError(f"Reranker service request failed: {exc}") from exc

    async def health_check(self) -> bool:
        if self._session is None:
            return False
        try:
            async with self._session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False
