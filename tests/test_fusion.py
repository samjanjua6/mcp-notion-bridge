import pytest
from mcp_notion_bridge.models import DocumentChunk
from mcp_notion_bridge.fusion import reciprocal_rank_fusion


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_single_list():
    chunks = [
        DocumentChunk(id="c1", text="Chunk 1", source="vector_db"),
        DocumentChunk(id="c2", text="Chunk 2", source="vector_db"),
    ]
    fused = reciprocal_rank_fusion([chunks], k=60)
    assert len(fused) == 2
    assert fused[0].id == "c1"
    assert fused[1].id == "c2"
    assert fused[0].score > fused[1].score
    assert fused[0].rank == 1
    assert fused[1].rank == 2


def test_rrf_deduplication_and_scoring():
    # c1 appears in both lists, c2 only in list 1, c3 only in list 2
    list_1 = [
        DocumentChunk(id="c1", text="Doc 1", source="vector_db"),
        DocumentChunk(id="c2", text="Doc 2", source="vector_db"),
    ]
    list_2 = [
        DocumentChunk(id="c1", text="Doc 1", source="lexical_bm25"),
        DocumentChunk(id="c3", text="Doc 3", source="lexical_bm25"),
    ]

    fused = reciprocal_rank_fusion([list_1, list_2], k=60)
    # c1 should be rank 1 because it accumulated scores from both lists
    assert fused[0].id == "c1"
    assert len(fused) == 3

    # Exact RRF check for c1: (1 / (60 + 0 + 1)) + (1 / (60 + 0 + 1)) = 2 / 61 = 0.032787
    expected_c1 = round((1.0 / 61.0) + (1.0 / 61.0), 6)
    assert fused[0].score == expected_c1


def test_rrf_top_n_truncation():
    chunks = [DocumentChunk(id=f"c{i}", text=f"Doc {i}") for i in range(10)]
    fused = reciprocal_rank_fusion([chunks], k=60, top_n=3)
    assert len(fused) == 3
    assert [f.id for f in fused] == ["c0", "c1", "c2"]


def test_rrf_priority_boost():
    mcp_chunk = DocumentChunk(id="mcp_1", text="Live MCP doc", source="notion_mcp")
    vector_chunk = DocumentChunk(id="vec_1", text="Vector doc", source="vector_db")

    # When both are rank 0 in their respective lists, mcp_1 gets priority boost
    fused = reciprocal_rank_fusion(
        [[mcp_chunk], [vector_chunk]],
        k=60,
        priority_sources=["notion_mcp"],
        priority_boost=1.5
    )
    assert fused[0].id == "mcp_1"
    assert fused[0].score > fused[1].score
