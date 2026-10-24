"""Reciprocal Rank Fusion (RRF) implementation for hybrid retrieval."""

from typing import List, Dict, Optional
from .models import DocumentChunk


def reciprocal_rank_fusion(
    ranked_lists: List[List[DocumentChunk]],
    k: int = 60,
    top_n: Optional[int] = None,
    priority_sources: Optional[List[str]] = None,
    priority_boost: float = 1.2
) -> List[DocumentChunk]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{m in M} [ weight(m) / (k + rank_m(d)) ]

    Parameters:
        ranked_lists: A list of ranked DocumentChunk lists (e.g. [dense_results, sparse_results, live_mcp_results]).
        k: Smoothing constant preventing high-rank saturation (standard default = 60).
        top_n: Number of top documents to return. If None, returns all unique documents.
        priority_sources: List of source names (e.g. ['notion_mcp']) that receive a priority score multiplier.
        priority_boost: Multiplier applied to priority sources (default = 1.2).

    Returns:
        A list of DocumentChunk sorted in descending order of fused RRF score.
    """
    if not ranked_lists:
        return []

    scores: Dict[str, float] = {}
    chunk_map: Dict[str, DocumentChunk] = {}
    priority_set = set(priority_sources or [])

    for r_list in ranked_lists:
        for rank, chunk in enumerate(r_list):
            doc_id = chunk.id
            if doc_id not in chunk_map:
                chunk_map[doc_id] = chunk

            # Base RRF calculation: 1.0 / (k + rank + 1) where rank is 0-indexed
            rank_score = 1.0 / (k + rank + 1)

            # Apply priority boost for live volatile context (e.g. live Notion MCP)
            if chunk.source in priority_set:
                rank_score *= priority_boost

            scores[doc_id] = scores.get(doc_id, 0.0) + rank_score

    # Sort descending by fused score
    sorted_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)

    if top_n is not None:
        sorted_doc_ids = sorted_doc_ids[:top_n]

    fused_results: List[DocumentChunk] = []
    for final_rank, doc_id in enumerate(sorted_doc_ids, start=1):
        original = chunk_map[doc_id]
        fused_chunk = DocumentChunk(
            id=original.id,
            text=original.text,
            metadata=original.metadata,
            source=original.source,
            score=round(scores[doc_id], 6),
            rank=final_rank
        )
        fused_results.append(fused_chunk)

    return fused_results
