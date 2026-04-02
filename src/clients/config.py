import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration."""

    HOST: str = os.getenv("DB_HOST", "localhost")
    PORT: int = int(os.getenv("DB_PORT", "5432"))
    USER: str = os.getenv("DB_USER", "postgres")
    PASSWORD: str = os.getenv("DB_PASSWORD", "ximukeji2026")
    DATABASE: str = os.getenv("DB_NAME", "knowledge_rag_materials")

    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 0
    POOL_TIMEOUT: int = 30

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://"
            f"{self.USER}:{self.PASSWORD}"
            f"@{self.HOST}:{self.PORT}/{self.DATABASE}"
        )


db_config = DatabaseConfig()

DATABASE_URL = db_config.url

EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL",
    os.getenv("EMBEDDING_API_BASE_URL", "http://localhost:40291"),
)
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "30"))

RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "http://localhost:40292")
RERANKER_TIMEOUT = int(os.getenv("RERANKER_TIMEOUT", "30"))

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
DEFAULT_RERANK_TOP_K = int(os.getenv("DEFAULT_RERANK_TOP_K", "50"))
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.0"))
DEFAULT_SEARCH_MODE = os.getenv("DEFAULT_SEARCH_MODE", "hybrid")

RRF_K = int(os.getenv("RRF_K", "60"))
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "1469"))

# Optional public URL prefix for converting local://<filename> into clickable links.
ARTICLE_LINK_BASE_URL = os.getenv("ARTICLE_LINK_BASE_URL", "").rstrip("/")

# Directory served by /files for local://<filename> links.
PDF_SERVE_DIR = os.getenv(
    "PDF_SERVE_DIR",
    str((Path(__file__).resolve().parents[2] / "src" / "mypdf").resolve()),
)
