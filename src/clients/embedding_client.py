from __future__ import annotations

import logging
import os
from typing import List

import requests

logger = logging.getLogger(__name__)


class EmbeddingAPIError(Exception):
    """Raised when embedding API call fails or returns invalid data."""


EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", "http://www.science42.vip:40291")
EMBEDDING_API_ENDPOINT = f"{EMBEDDING_API_BASE_URL.rstrip('/')}/embed"


def get_embeddings(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """Call embedding API and return vectors for input texts."""
    if not texts:
        return []
    if len(texts) > 1000:
        raise ValueError("texts size cannot exceed 1000")

    payload = {
        "texts": texts,
        "batch_size": batch_size,
    }

    try:
        response = requests.post(EMBEDDING_API_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("Embedding API network/request error: %s", exc)
        raise EmbeddingAPIError(f"Embedding API request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        logger.exception("Embedding API response is not valid JSON")
        raise EmbeddingAPIError("Embedding API response is not valid JSON") from exc

    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list):
        raise EmbeddingAPIError("Embedding API response missing 'embeddings'")

    if len(embeddings) != len(texts):
        raise EmbeddingAPIError(
            f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
        )

    dimensions = data.get("dimensions")
    if dimensions is not None:
        for idx, vector in enumerate(embeddings):
            if not isinstance(vector, list) or len(vector) != dimensions:
                raise EmbeddingAPIError(
                    f"Invalid vector size at index {idx}: got {len(vector) if isinstance(vector, list) else 'N/A'}, expected {dimensions}"
                )

    return embeddings
