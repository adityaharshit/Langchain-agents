"""
Pydantic request/response schemas for Deep Research Assistant API.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl


# Request Schemas
class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    query: str = Field(..., description="Research query to process", min_length=1, max_length=2000)
    stream: bool = Field(True, description="Whether to stream progress events")
    max_results: int = Field(8, description="Maximum number of results to retrieve", ge=1, le=50)
    search_method: Literal["semantic", "keyword", "hybrid"] = Field("semantic", description="Search method to use")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold", ge=0.0, le=1.0)


class TaskStatusRequest(BaseModel):
    """Request schema for task status updates."""
    status: Literal["pending", "running", "completed", "failed"]
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# Response Schemas
class ProgressEvent(BaseModel):
    """Progress event schema for SSE streaming."""
    event: Literal["progress", "partial", "final", "error"]
    step: str
    payload: Dict[str, Any]
    timestamp: str
    task_id: str


class ProvenanceRecord(BaseModel):
    """Provenance record linking claims to sources."""
    claim: str
    chunk_id: int
    document_title: str
    document_url: HttpUrl
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    citation_number: int


class SearchResult(BaseModel):
    """Search result schema."""
    chunk_id: int
    document_id: int
    chunk_text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    document_title: str
    document_url: HttpUrl
    chunk_meta: Dict[str, Any]
    token_count: int


class ResearchResult(BaseModel):
    """Final research result schema."""
    answer_markdown: str
    citations: List[str]
    provenance: List[ProvenanceRecord]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    total_sources: int
    search_method: str


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    task_id: str
    status: Literal["accepted", "processing", "completed", "failed"]
    message: str
    result: Optional[ResearchResult] = None


class TaskStatusResponse(BaseModel):
    """Response schema for task status endpoint."""
    task_id: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[ResearchResult] = None
    error_message: Optional[str] = None
    progress_events: List[ProgressEvent] = []


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: Literal["healthy", "unhealthy"]
    timestamp: datetime
    version: str
    components: Dict[str, Dict[str, Any]]


class StatsResponse(BaseModel):
    """System statistics response schema."""
    database_stats: Dict[str, Any]
    processing_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    uptime_seconds: float


# Agent Schemas
class AgentToolBinding(BaseModel):
    """Schema for agent-tool binding configuration."""
    agent_name: str
    allowed_tools: List[str]
    tool_config: Dict[str, Any] = {}


class MCPToolInfo(BaseModel):
    """MCP tool information schema."""
    name: str
    description: str
    allowed_agents: List[str]
    parameters: Dict[str, Any]
    tags: List[str] = []


class AgentExecutionLog(BaseModel):
    """Agent execution log entry schema."""
    agent_name: str
    tool_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime


# Document and Chunk Schemas
class DocumentMetadata(BaseModel):
    """Document metadata schema."""
    title: str
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    description: Optional[str] = None
    keywords: List[str] = []
    language: str = "en"
    source_trust_score: float = Field(0.5, ge=0.0, le=1.0)
    license: Optional[str] = None


class ChunkMetadata(BaseModel):
    """Chunk metadata schema."""
    section: str
    heading: str
    position: int
    boundary_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    has_overlap: bool = False
    original_tokens: Optional[int] = None


class DocumentInfo(BaseModel):
    """Document information schema."""
    id: int
    url: HttpUrl
    title: str
    language: str
    source_trust_score: float
    chunk_count: int
    total_tokens: int
    inserted_at: datetime


class ChunkInfo(BaseModel):
    """Chunk information schema."""
    id: int
    document_id: int
    chunk_text: str
    token_count: int
    chunk_meta: ChunkMetadata
    similarity_score: Optional[float] = None


# Error Schemas
class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[str] = None


class ValidationError(BaseModel):
    """Validation error schema."""
    field: str
    message: str
    invalid_value: Any


# Configuration Schemas
class SystemConfig(BaseModel):
    """System configuration schema."""
    database_url: str
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    max_concurrent_scrapes: int = 5
    scrape_rate_limit: float = 1.0
    retrieval_k: int = 8
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.75


# Batch Processing Schemas
class BatchProcessingRequest(BaseModel):
    """Batch processing request schema."""
    urls: List[HttpUrl] = Field(..., max_items=100)
    batch_size: int = Field(10, ge=1, le=50)
    store_in_db: bool = True
    language_filter: bool = True


class BatchProcessingResponse(BaseModel):
    """Batch processing response schema."""
    batch_id: str
    total_urls: int
    status: Literal["queued", "processing", "completed", "failed"]
    processed_count: int = 0
    success_count: int = 0
    error_count: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


# Analysis Schemas
class AnalysisRequest(BaseModel):
    """Analysis request schema."""
    query: str
    document_ids: Optional[List[int]] = None
    analysis_types: List[Literal["comparative", "trend", "causal", "statistical"]] = ["comparative"]
    max_documents: int = Field(20, ge=1, le=100)


class AnalysisResult(BaseModel):
    """Analysis result schema."""
    analysis_type: str
    findings: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[Dict[str, Any]]
    methodology: str
    limitations: List[str] = []


class ComprehensiveAnalysisResponse(BaseModel):
    """Comprehensive analysis response schema."""
    query: str
    analyses: List[AnalysisResult]
    synthesis: str
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    sources_analyzed: int