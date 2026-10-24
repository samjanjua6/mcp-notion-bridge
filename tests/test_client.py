import pytest
from mcp_notion_bridge.client import NotionMCPClient
from mcp_notion_bridge.models import NotionPage


@pytest.mark.asyncio
async def test_mock_search():
    client = NotionMCPClient()
    pages = await client.search_pages("refund policy", limit=3)
    assert len(pages) > 0
    assert "Refund Policy" in pages[0].title
    assert "refund policy" in pages[0].content_markdown


@pytest.mark.asyncio
async def test_chunk_page():
    client = NotionMCPClient()
    page = NotionPage(
        id="test-p1",
        title="Terms of Service",
        content_markdown="Word " * 600
    )
    chunks = client.chunk_page(page, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[0].source == "notion_mcp"
    assert "test-p1#chunk-1" in chunks[0].id
    assert chunks[0].metadata["title"] == "Terms of Service"


@pytest.mark.asyncio
async def test_empty_page_chunking():
    client = NotionMCPClient()
    page = NotionPage(id="empty", content_markdown="")
    chunks = client.chunk_page(page)
    assert chunks == []
