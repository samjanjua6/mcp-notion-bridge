# mcp-notion-bridge

[![PyPI version](https://img.shields.io/pypi/v/mcp-notion-bridge.svg?color=blue)](https://pypi.org/project/mcp-notion-bridge/)
[![Python versions](https://img.shields.io/pypi/pyversions/mcp-notion-bridge.svg)](https://pypi.org/project/mcp-notion-bridge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/mcp-notion-bridge.svg)](https://pypistats.org/packages/mcp-notion-bridge)

> **Lightweight Model Context Protocol (MCP) bridge and hybrid sync engine connecting Notion workspaces with AI agents and vector RAG pipelines.**

Stop serving obsolete data from static vector indices. `mcp-notion-bridge` implements an event-driven retrieval layer adhering to Anthropic's open **Model Context Protocol (MCP)**. It dynamically intercepts volatile documentation updates from live Notion workspaces and fuses them with local vector stores (ChromaDB, Pgvector) and sparse BM25 search using **Reciprocal Rank Fusion (RRF)**.

---

## ⚡ The Problem: The Stale Embeddings Crisis

Real-world enterprise documentation lives in collaborative workspaces like Notion. When teams update discount policies, compliance clauses, or fee structures:
* **The vector index is instantly obsolete**: Embeddings do not auto-sync without expensive batch re-indexing.
* **Batch re-embedding is slow and costly**: Processing millions of tokens on every minor edit causes write-locks, high API bills, and synchronization lag.
* **LLMs confidently hallucinate outdated terms**: RAG pipelines retrieve stale chunks with high cosine similarity.

```
[Traditional RAG]
[User Query] ──> [Static Vector DB (Stale Chunks!)] ──> [LLM] ──> ❌ Outdated Policy Answer
```

---

## 🛡️ The Architecture: Live MCP Hybrid Retrieval

```
                             ┌──────────────────────────────────────────────┐
                             │       Live Notion Workspace (via MCP)        │
                             └──────────────────────┬───────────────────────┘
                                                    │
                                  [Live Volatile Chunks (Priority)]
                                                    │
[User Query] ──> [HybridMCPRetriever] ──────────────┼─────────────────────────┐
                        │                           │                         │
                        ▼                           ▼                         ▼
            [Live Notion MCP Stream]    [Dense ChromaDB Vectors]    [Sparse BM25 Lexical]
                        │                           │                         │
                        └───────────────────┬───────┴─────────────────────────┘
                                            ▼
                           [Reciprocal Rank Fusion (RRF k=60)]
                                            │
                                            ▼
                        [Zero-Staleness Context + Provenance]
                                            │
                                            ▼
                               [AI Agent / LangGraph Graph]
```

`mcp-notion-bridge` solves this with a multi-pass hybrid engine:
1. **Live MCP Interception**: Checks active Notion pages over MCP before querying static indices.
2. **Dense Semantic Retrieval**: Queries local ChromaDB or pgvector embeddings.
3. **Sparse Lexical Retrieval**: Executes BM25 keyword matching for exact IDs, SKUs, and codes.
4. **Reciprocal Rank Fusion (RRF)**: Mathematically fuses candidate rankings with zero arbitrary hyperparameter weighting.

---

## 📦 Installation

```bash
pip install mcp-notion-bridge
```

### Optional Extras (for ChromaDB and Rank-BM25):
```bash
pip install "mcp-notion-bridge[rag]"
```

Zero heavy dependencies. Pure Python with Pydantic v2 and HTTPX.

---

## 🚀 Quickstart in 4 Lines

```python
import asyncio
from mcp_notion_bridge import HybridMCPRetriever

async def main():
    retriever = HybridMCPRetriever()
    result = await retriever.retrieve("What is the active cancellation policy?", top_k=3)
    
    print(f"Latency: {result.latency_ms} ms | Live Freshness: {result.is_fresh}")
    for doc in result.fused_results:
        print(f"[{doc.source.upper()}] (Score: {doc.score:.4f}) {doc.text[:100]}...")

asyncio.run(main())
```

---

## 🧩 LangGraph & Agentic Workflows

Integrate `mcp-notion-bridge` directly into your LangGraph state graph as an autonomous retrieval node:

```python
from mcp_notion_bridge import HybridMCPRetriever
from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    context_chunks: List[str]
    is_live_notion: bool

retriever = HybridMCPRetriever()

async def retrieve_context_node(state: AgentState):
    result = await retriever.retrieve(state["query"], top_k=4)
    return {
        "context_chunks": [c.text for c in result.fused_results],
        "is_live_notion": result.is_fresh
    }
```

---

## 📐 Mathematical Formulation: Reciprocal Rank Fusion

Candidate chunks from heterogeneous sources (live Notion MCP $M$, dense vector search $D$, and sparse BM25 $S$) are merged using Reciprocal Rank Fusion:

$$\text{RRF}(d) = \sum_{m \in \{M, D, S\}} \frac{w_m}{k + r_m(d)}$$

Where:
* $k = 60$ (standard smoothing constant preventing rank saturation).
* $r_m(d)$ is the 1-based ordinal rank of document $d$ in retrieval system $m$.
* $w_m = 1.25$ for live MCP streams (ensuring verified volatile changes override older static embeddings).

---

## 🖥️ Command-Line Interface (CLI)

`mcp-notion-bridge` includes a built-in terminal CLI:

```bash
# Inspect live Notion workspace via MCP
mcp-notion-bridge inspect "refund policy" --limit 3

# Run full hybrid retrieval with RRF
mcp-notion-bridge retrieve "overdue library fees" --top-k 5

# Benchmark in-flight RRF latency
mcp-notion-bridge benchmark
```

---

## 🧪 Running Unit Tests

```bash
git clone https://github.com/samjanjua6/mcp-notion-bridge.git
cd mcp-notion-bridge
pip install -e ".[dev]"
pytest
```

---

## 👤 Author & Research Citation

Engineered by **[Shaheer Ul Hassan (Muhammad Shaheer)](https://shaheerulhassan.com/)**  
*Agentic AI & Systems Engineer | Islamabad, Pakistan*  
* Portfolio: [shaheerulhassan.com](https://shaheerulhassan.com/)  
* Technical Case Study: [Solving the Stale Embeddings Crisis with MCP](https://shaheerulhassan.com/blog/solving-stale-embeddings-hybrid-rag-notion-mcp)  
* LinkedIn: [Muhammad Shaheer Ul Hassan](https://www.linkedin.com/in/muhammad-shaheer-ul-hassan-b3b7213b3/)  
* PyPI Suite: [`rag-critic`](https://pypi.org/project/rag-critic/) | [`deeptrace-ela`](https://pypi.org/project/deeptrace-ela/)

---

## 📄 License

MIT License. Open-source for academic, enterprise, and personal development.
