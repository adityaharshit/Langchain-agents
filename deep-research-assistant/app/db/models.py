"""
SQLAlchemy models for Deep Research Assistant with pgvector support.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, ForeignKey, 
    UniqueConstraint, Index, JSON as JSONB
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Document(Base):
    """Document storage table for scraped web content."""
    __tablename__ = "documents"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    publish_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    fetch_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    raw_html: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")
    source_trust_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    license: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    inserted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, url='{self.url}', title='{self.title[:50]}...')>"


class Chunk(Base):
    """Chunk storage table with vector embeddings for semantic search."""
    __tablename__ = "chunks"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_meta: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1536), nullable=True)  # OpenAI text-embedding-3-small
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes for efficient vector search
    __table_args__ = (
        Index("chunks_embedding_idx", "embedding", postgresql_using="hnsw", 
              postgresql_with={"m": 16, "ef_construction": 64}, 
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("chunks_document_id_idx", "document_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, document_id={self.document_id}, tokens={self.token_count})>"


class Query(Base):
    """Query tracking table for user research requests."""
    __tablename__ = "queries"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    final_result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    provenance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    processing_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Relationships
    tasks = relationship("Task", back_populates="query", cascade="all, delete-orphan")
    agent_logs = relationship("AgentLog", back_populates="query", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Query(id={self.id}, text='{self.query_text[:50]}...', created_at={self.created_at})>"


class Task(Base):
    """Task tracking table for individual agent operations."""
    __tablename__ = "tasks"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(Integer, ForeignKey("queries.id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")  # pending, running, completed, failed
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    query = relationship("Query", back_populates="tasks")
    
    # Indexes
    __table_args__ = (
        Index("tasks_query_id_idx", "query_id"),
        Index("tasks_status_idx", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<Task(id={self.id}, tool='{self.tool_name}', status='{self.status}')>"


class AgentLog(Base):
    """Agent logging table for detailed execution tracking and provenance."""
    __tablename__ = "agents_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(Integer, ForeignKey("queries.id"), nullable=False)
    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    step_name: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    log_level: Mapped[str] = mapped_column(String, nullable=False, default="INFO")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    
    # Relationships
    query = relationship("Query", back_populates="agent_logs")
    
    # Indexes
    __table_args__ = (
        Index("agent_logs_query_id_idx", "query_id"),
        Index("agent_logs_agent_name_idx", "agent_name"),
        Index("agent_logs_created_at_idx", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<AgentLog(id={self.id}, agent='{self.agent_name}', step='{self.step_name}')>"


# Additional indexes for performance optimization
def create_additional_indexes():
    """Create additional indexes for query performance."""
    return [
        # Full-text search index for documents
        Index("documents_title_text_idx", Document.title, Document.cleaned_text, 
              postgresql_using="gin", postgresql_ops={"title": "gin_trgm_ops", "cleaned_text": "gin_trgm_ops"}),
        
        # Composite indexes for common queries
        Index("chunks_document_token_idx", Chunk.document_id, Chunk.token_count),
        Index("queries_created_user_idx", Query.created_at, Query.user_id),
        Index("tasks_query_status_idx", Task.query_id, Task.status),
    ]