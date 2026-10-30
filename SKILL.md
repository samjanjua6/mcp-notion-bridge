---
name: notion-agent-bridge
description: Connect Claude Code, Cursor, and AI agents directly to live Notion workspaces using the mcp-notion-bridge protocol. Solves stale embeddings with sub-millisecond Reciprocal Rank Fusion (RRF), live workspace inspection, and zero-hallucination document synthesis.
version: 0.1.0
author: Shaheer Ul Hassan
license: MIT
homepage: https://shaheerulhassan.com/blog/solving-stale-embeddings-hybrid-rag-notion-mcp
repository: https://github.com/samjanjua6/mcp-notion-bridge
pypi: https://pypi.org/project/mcp-notion-bridge/
tags:
  - notion
  - mcp
  - model-context-protocol
  - claude-code
  - cursor
  - rag
  - reciprocal-rank-fusion
  - hybrid-retrieval
---

# Notion Agent Bridge (`mcp-notion-bridge`)

An autonomous agent skill enabling **Claude Code**, **Cursor**, and **AI Coding Agents** to query, inspect, and retrieve live organizational knowledge directly from Notion workspaces.

Standard RAG architectures suffer from the **Stale Embeddings Crisis**: vector embeddings are days or weeks old, leading agents to write code or plan sprints against outdated PRDs, deprecated API endpoints, or obsolete tickets. This skill bridges the agent directly to live Notion pages using **Reciprocal Rank Fusion (RRF)** ($k=60$), prioritizing volatile, up-to-the-minute documents over stale static vectors.

---

## ⚡ When to Activate This Skill

The agent MUST invoke this skill when:
1. **Engineering Specs & PRDs**: The user asks to implement a feature, check requirements, or verify architectural decisions based on documentation stored in Notion.
2. **Sprint Planning & Ticket Auditing**: The user asks about task status, deadlines, ownership, or roadmap priorities located in Notion databases.
3. **Internal Documentation & Style Guides**: The user asks about company coding guidelines, onboarding protocols, or deployment steps.
4. **Live Notion Page Inspection**: The user provides a Notion URL, Page ID, or query string and asks for a summary or codebase alignment.
5. **Pre-Implementation Verification**: Before executing large multi-file edits in Cursor or Claude Code, the agent should query Notion to confirm no recent deprecations were published.

---

## 🛠️ Prerequisites & Installation

### 1. Install the CLI & Engine
```bash
pip install mcp-notion-bridge
```

### 2. Set Your Notion Integration Token
Obtain an internal integration token from [notion.so/my-integrations](https://www.notion.so/my-integrations) and share the relevant pages or parent databases with your integration:
```bash
export NOTION_API_KEY="secret_your_notion_integration_token_here"
```
*(On Windows PowerShell: `$env:NOTION_API_KEY="secret_your_notion_integration_token_here"`)*

---

## 💻 Available Commands for Agents

### 1. Inspect Live Pages (`inspect`)
Search the workspace by keyword and inspect matching page titles, last edited timestamps, and structural block counts:
```bash
mcp-notion-bridge inspect "<search-query>" --limit 5
```
*Use this when the agent needs to locate the exact page ID or verify the most recently updated document.*

### 2. Execute Hybrid Multi-Pass Retrieval (`retrieve`)
Run full Reciprocal Rank Fusion across live Notion MCP context, local embeddings, and lexical BM25 matching:
```bash
mcp-notion-bridge retrieve "<search-query>" --top-k 5
```
*Returns ranked, deduped document chunks scored via RRF with exact millisecond latency breakdown.*

### 3. In-Memory RRF Micro-Benchmark (`benchmark`)
Verify local sub-millisecond fusion throughput on the host machine:
```bash
mcp-notion-bridge benchmark
```

---

## 🤖 Python Programmatic API (For Agentic State Graphs)

Agents writing automated scripts or LangGraph state nodes can integrate the bridge directly:

```python
import asyncio
from mcp_notion_bridge import HybridMCPRetriever, NotionMCPClient

async def sync_workspace_context(query: str):
    # Initialize hybrid retriever with 1.25x MCP priority boost
    retriever = HybridMCPRetriever(mcp_priority_boost=1.25, k_constant=60)
    
    # Execute fused retrieval across live Notion + dense vectors
    result = await retriever.retrieve(query, top_k=5, use_live_mcp=True)
    
    print(f"Retrieval Latency: {result.latency_ms} ms")
    print(f"Is Fresh:          {result.is_fresh}")
    
    for chunk in result.fused_results:
        print(f"[Rank {chunk.rank}] (Score: {chunk.score:.4f}) Source: {chunk.source}")
        print(f"{chunk.text}\n---")

if __name__ == "__main__":
    asyncio.run(sync_workspace_context("Authentication service architecture"))
```

---

## ⚙️ Direct MCP Configuration (Claude Desktop & Cursor)

To install `mcp-notion-bridge` as a native background MCP service:

### In Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "notion-bridge": {
      "command": "python",
      "args": ["-m", "mcp_notion_bridge.cli", "retrieve", "system architecture"],
      "env": {
        "NOTION_API_KEY": "secret_your_notion_token_here"
      }
    }
  }
}
```

### In Cursor (`.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "notion-bridge": {
      "command": "mcp-notion-bridge",
      "args": ["retrieve", "internal documentation"],
      "env": {
        "NOTION_API_KEY": "secret_your_notion_token_here"
      }
    }
  }
}
```

---

## 📐 Agent Operating Rules & Anti-Hallucination Guidelines

When generating answers using context from `mcp-notion-bridge`, the AI agent MUST adhere to these four operational rules:

1. **Strict Provenance & In-Text Citations**:
   Every claim synthesized from Notion must cite the source:
   > *"According to the Q4 Architecture Roadmap (updated 2026-10-24), JWT tokens are deprecated in favor of PASETO v4."*
2. **Freshness Precedence Over Vector Memory**:
   If static vector search contradicts the live Notion block fetched via MCP, the agent **must prioritize the Notion block** and explicitly flag the discrepancy to the developer:
   > *"Note: Local vector store specified PostgreSQL 15, but the live Notion Engineering Guide was updated yesterday to PostgreSQL 16. Using PostgreSQL 16."*
3. **No Hallucinated Page IDs or Secrets**:
   Never invent Notion page IDs or assume workspace contents if `inspect` returns 0 results. If no documentation is found, reply:
   > *"No matching documentation found in the connected Notion workspace for query: '<query>'. Would you like me to inspect another keyword?"*
4. **Markdown Preservation**:
   Preserve headings, callouts, and code block formatting extracted from Notion's native rich-text blocks.
