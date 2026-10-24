"""Data models for mcp-notion-bridge."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import time


class NotionPage(BaseModel):
    """Represents a Notion page extracted via MCP or direct API."""
    id: str
    title: str = "Untitled"
    url: Optional[str] = None
    last_edited_time: Optional[str] = None
    content_markdown: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A single atomic text chunk with provenance and ranking scores."""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Literal["notion_mcp", "vector_db", "lexical_bm25", "hybrid"] = "notion_mcp"
    score: float = 0.0
    rank: Optional[int] = None


class RetrievalResult(BaseModel):
    """Encapsulates the output of a hybrid MCP retrieval operation."""
    query: str
    is_fresh: bool = False
    fused_results: List[DocumentChunk] = Field(default_factory=list)
    live_chunks_count: int = 0
    dense_chunks_count: int = 0
    sparse_chunks_count: int = 0
    latency_ms: float = 0.0
    used_cache: bool = False


class SyncStatus(BaseModel):
    """Operational status report for Notion workspace sync."""
    pages_scanned: int = 0
    pages_synced: int = 0
    chunks_generated: int = 0
    duration_ms: float = 0.0
    status: Literal["success", "partial", "failed"] = "success"
    message: Optional[str] = None
