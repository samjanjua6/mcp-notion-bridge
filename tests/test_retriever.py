import pytest
from mcp_notion_bridge.retriever import HybridMCPRetriever
from mcp_notion_bridge.models import DocumentChunk


@pytest.mark.asyncio
async def test_hybrid_retriever_flow():
    retriever = HybridMCPRetriever()
    result = await retriever.retrieve("library return policy", top_k=3, use_live_mcp=True)

    assert result.query == "library return policy"
    assert len(result.fused_results) <= 3
    assert result.latency_ms >= 0.0
    assert result.is_fresh is True
    assert result.used_cache is False

    # Check caching
    cached_result = await retriever.retrieve("library return policy", top_k=3, use_cache=True)
    assert cached_result.used_cache is True


@pytest.mark.asyncio
async def test_custom_vector_and_lexical_callbacks():
    def custom_vector(q, k):
        return [DocumentChunk(id="custom-vec-1", text="Custom dense hit", source="vector_db")]

    def custom_lexical(q, k):
        return [DocumentChunk(id="custom-lex-1", text="Custom sparse hit", source="lexical_bm25")]

    retriever = HybridMCPRetriever(
        vector_search_fn=custom_vector,
        lexical_search_fn=custom_lexical
    )

    result = await retriever.retrieve("test query", top_k=2, use_live_mcp=False)
    ids = [c.id for c in result.fused_results]
    assert "custom-vec-1" in ids
    assert "custom-lex-1" in ids
