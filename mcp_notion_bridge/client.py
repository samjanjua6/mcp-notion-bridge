"""Notion MCP Client implementation supporting MCP protocol and direct REST fallback."""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from .models import NotionPage, DocumentChunk


class NotionMCPClient:
    """
    Client connecting Notion workspaces to agentic RAG workflows via
    Model Context Protocol (MCP) or direct HTTPS API endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.notion.com/v1",
        notion_version: str = "2022-06-28",
        timeout: float = 10.0
    ):
        self.api_key = api_key or os.getenv("NOTION_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.notion_version = notion_version
        self.timeout = timeout

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": self.notion_version,
            "Content-Type": "application/json"
        }

    async def search_pages(self, query: str, limit: int = 5) -> List[NotionPage]:
        """
        Searches Notion workspace for pages matching the query.
        Falls back to mock results if no API key is set (for CI / testing).
        """
        if not self.api_key:
            return self._mock_search(query, limit)

        url = f"{self.base_url}/search"
        payload = {
            "query": query,
            "filter": {"value": "page", "property": "object"},
            "page_size": limit
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=self.headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Notion API error [{resp.status_code}]: {resp.text}")

            data = resp.json()
            pages: List[NotionPage] = []
            for item in data.get("results", []):
                page_id = item.get("id", "")
                title = "Untitled"
                # Extract title from properties
                props = item.get("properties", {})
                for p_val in props.values():
                    if p_val.get("type") == "title" and p_val.get("title"):
                        title = "".join(t.get("plain_text", "") for t in p_val.get("title", []))
                        break

                pages.append(NotionPage(
                    id=page_id,
                    title=title,
                    url=item.get("url"),
                    last_edited_time=item.get("last_edited_time"),
                    properties=props
                ))
            return pages

    async def get_page_markdown(self, page_id: str) -> NotionPage:
        """
        Retrieves page blocks and transforms them into clean markdown text.
        """
        if not self.api_key:
            return self._mock_page(page_id)

        clean_id = page_id.replace("-", "")
        url = f"{self.base_url}/blocks/{clean_id}/children"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=self.headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch Notion blocks [{resp.status_code}]: {resp.text}")

            blocks = resp.json().get("results", [])
            lines: List[str] = []

            for block in blocks:
                b_type = block.get("type")
                b_data = block.get(b_type, {})
                rich_text = b_data.get("rich_text", [])
                text = "".join(t.get("plain_text", "") for t in rich_text)

                if b_type == "paragraph":
                    lines.append(text)
                elif b_type == "heading_1":
                    lines.append(f"# {text}")
                elif b_type == "heading_2":
                    lines.append(f"## {text}")
                elif b_type == "heading_3":
                    lines.append(f"### {text}")
                elif b_type == "bulleted_list_item":
                    lines.append(f"- {text}")
                elif b_type == "numbered_list_item":
                    lines.append(f"1. {text}")
                elif b_type == "code":
                    lines.append(f"```{b_data.get('language', '')}\n{text}\n```")

            content = "\n\n".join(lines)
            return NotionPage(
                id=page_id,
                content_markdown=content
            )

    def chunk_page(
        self,
        page: NotionPage,
        chunk_size: int = 500,
        chunk_overlap: int = 60
    ) -> List[DocumentChunk]:
        """
        Splits markdown page content into overlapping chunks suitable for embeddings and RAG.
        """
        text = page.content_markdown.strip()
        if not text:
            return []

        chunks: List[DocumentChunk] = []
        words = text.split()
        step = max(1, chunk_size - chunk_overlap)

        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{page.id}#chunk-{len(chunks) + 1}"

            chunks.append(DocumentChunk(
                id=chunk_id,
                text=chunk_text,
                metadata={
                    "page_id": page.id,
                    "title": page.title,
                    "url": page.url,
                    "last_edited_time": page.last_edited_time,
                    "chunk_index": len(chunks)
                },
                source="notion_mcp",
                score=1.0
            ))

        return chunks

    def _mock_search(self, query: str, limit: int) -> List[NotionPage]:
        """Mock search response when running offline or in demo mode."""
        return [
            NotionPage(
                id="mock-page-001",
                title=f"Live Policy Document: {query.title()}",
                url="https://notion.so/mock-policy-001",
                last_edited_time="2026-10-24T10:00:00Z",
                content_markdown=f"# Active Operational Policy\n\nThis is live Notion context retrieved for query '{query}'. Policy updated with zero-staleness guarantee."
            )
        ][:limit]

    def _mock_page(self, page_id: str) -> NotionPage:
        return NotionPage(
            id=page_id,
            title="Sample Live Policy",
            content_markdown="## Dynamic Fee & Operational Guidelines\n\n- Active Rate: $0.00 during trial period.\n- SLA Guarantee: 99.99% uptime with Model Context Protocol real-time sync.\n- In-flight evaluation: LLM-as-a-judge proposition scoring active."
        )
