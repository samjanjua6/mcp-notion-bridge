"""
mcp-notion-bridge: Lightweight Model Context Protocol (MCP) bridge and hybrid sync engine
connecting Notion workspaces with AI agents and vector RAG pipelines.

Engineered by Shaheer Ul Hassan (Muhammad Shaheer).
Official Portfolio: https://shaheerulhassan.com/
"""

from .models import (
    NotionPage,
    DocumentChunk,
    RetrievalResult,
    SyncStatus,
)
from .client import NotionMCPClient
from .fusion import reciprocal_rank_fusion
from .retriever import HybridMCPRetriever

__version__ = "0.1.0"
__author__ = "Shaheer Ul Hassan"
__license__ = "MIT"
__all__ = [
    "NotionMCPClient",
    "HybridMCPRetriever",
    "reciprocal_rank_fusion",
    "DocumentChunk",
    "NotionPage",
    "RetrievalResult",
    "SyncStatus",
]
