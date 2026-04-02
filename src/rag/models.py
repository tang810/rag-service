from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, TIMESTAMP, Column, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Text, unique=True, nullable=False, index=True)

    title = Column(Text)
    authors = Column(Text)
    keywords = Column(ARRAY(Text))

    journal_conference = Column(Text)
    source_url = Column(Text)
    source_type = Column(Text)

    publish_year = Column(Integer)
    abstract = Column(Text)
    doc_type = Column(Text)
    create_time = Column(TIMESTAMP, default=datetime.now)

    chunks = relationship("Chunk", back_populates="document", lazy="selectin")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Text, unique=True, nullable=False, index=True)
    doc_id = Column(Text, ForeignKey("documents.doc_id"), nullable=False)

    content = Column(Text, nullable=False)

    page = Column(Integer)
    section_path = Column(JSONB)
    chunk_index = Column(Integer)

    create_time = Column(TIMESTAMP, default=datetime.now)

    document = relationship("Document", back_populates="chunks")
    embedding_row = relationship("Embedding", back_populates="chunk", uselist=False)


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Text, ForeignKey("chunks.chunk_id"), nullable=False, unique=True)
    doc_id = Column(Text, ForeignKey("documents.doc_id"), nullable=False)

    embedding = Column(Vector(1024), nullable=False)

    create_time = Column(TIMESTAMP, default=datetime.now)

    chunk = relationship("Chunk", back_populates="embedding_row")


class SearchLog(Base):
    __tablename__ = "search_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    query_hash = Column(Text, index=True)
    search_type = Column(Text)

    processing_time_ms = Column(Integer)
    results_count = Column(Integer)
    top_k = Column(Integer)

    embedding_time_ms = Column(Integer)
    vector_search_time_ms = Column(Integer)
    rerank_time_ms = Column(Integer)

    create_time = Column(TIMESTAMP, default=datetime.now)
