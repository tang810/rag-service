from __future__ import annotations

import hashlib


def generate_doc_id(source_url: str) -> str:
    """Generate a stable unique doc_id from source URL using SHA256."""
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()
