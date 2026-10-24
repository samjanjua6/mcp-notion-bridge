"""Command-line interface for mcp-notion-bridge."""

import sys
import asyncio
import argparse
from .client import NotionMCPClient
from .retriever import HybridMCPRetriever
from .fusion import reciprocal_rank_fusion
from .models import DocumentChunk

__version__ = "0.1.0"


def main():
    parser = argparse.ArgumentParser(
        prog="mcp-notion-bridge",
        description="Model Context Protocol (MCP) hybrid sync engine connecting Notion workspaces to AI agents & RAG."
    )
    parser.add_argument("--version", action="version", version=f"mcp-notion-bridge v{__version__} by Shaheer Ul Hassan")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Search and inspect live Notion pages via MCP")
    inspect_parser.add_argument("query", type=str, help="Search query string")
    inspect_parser.add_argument("--limit", type=int, default=3, help="Max pages to inspect (default: 3)")

    # Command: retrieve
    retrieve_parser = subparsers.add_parser("retrieve", help="Execute hybrid MCP + Vector + BM25 RRF retrieval")
    retrieve_parser.add_argument("query", type=str, help="Retrieval query string")
    retrieve_parser.add_argument("--top-k", type=int, default=5, help="Number of fused results (default: 5)")
    retrieve_parser.add_argument("--no-mcp", action="store_true", help="Bypass live MCP check")

    # Command: benchmark
    subparsers.add_parser("benchmark", help="Run in-flight RRF latency benchmark")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "inspect":
        asyncio.run(run_inspect(args.query, args.limit))
    elif args.command == "retrieve":
        asyncio.run(run_retrieve(args.query, args.top_k, not args.no_mcp))
    elif args.command == "benchmark":
        asyncio.run(run_benchmark())


async def run_inspect(query: str, limit: int):
    print(f"\n[mcp-notion-bridge] Inspecting live workspace for query: '{query}'")
    client = NotionMCPClient()
    pages = await client.search_pages(query, limit=limit)
    print(f"Found {len(pages)} matching page(s):\n")

    for i, p in enumerate(pages, start=1):
        print(f"[{i}] {p.title}")
        print(f"    Page ID: {p.id}")
        if p.url:
            print(f"    URL:     {p.url}")
        if p.last_edited_time:
            print(f"    Updated: {p.last_edited_time}")
        chunks = client.chunk_page(p, chunk_size=200, chunk_overlap=30)
        print(f"    Chunks:  {len(chunks)} generated\n")


async def run_retrieve(query: str, top_k: int, use_mcp: bool):
    print(f"\n[mcp-notion-bridge] Running Hybrid MCP Retrieval (top_k={top_k}, live_mcp={use_mcp})")
    retriever = HybridMCPRetriever()
    result = await retriever.retrieve(query, top_k=top_k, use_live_mcp=use_mcp)

    print(f"Latency:        {result.latency_ms} ms")
    print(f"Live Freshness: {'ACTIVE (Live Notion)' if result.is_fresh else 'STATIC (Indexed)'}")
    print(f"Source Counts:  MCP={result.live_chunks_count} | Vector={result.dense_chunks_count} | BM25={result.sparse_chunks_count}\n")
    print("--- Top Fused Candidates (RRF k=60) ---")
    for chunk in result.fused_results:
        print(f"Rank {chunk.rank} | Score: {chunk.score:.5f} | Source: {chunk.source.upper()}")
        print(f"Text: {chunk.text[:120]}...\n")


async def run_benchmark():
    print("\n[mcp-notion-bridge] Executing Reciprocal Rank Fusion Benchmark (1,000 iterations)...")
    import time
    # Create 3 ranked lists of 10 chunks each
    list_a = [DocumentChunk(id=f"doc-{i}", text="Sample text", source="notion_mcp") for i in range(10)]
    list_b = [DocumentChunk(id=f"doc-{i}", text="Sample text", source="vector_db") for i in range(5, 15)]
    list_c = [DocumentChunk(id=f"doc-{i}", text="Sample text", source="lexical_bm25") for i in range(3, 12)]

    t0 = time.perf_counter()
    for _ in range(1000):
        reciprocal_rank_fusion([list_a, list_b, list_c], k=60, top_n=5)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"Benchmark Complete: 1,000 RRF fusions executed in {elapsed:.2f} ms")
    print(f"Average Fusion Latency: {elapsed / 1000:.4f} ms per query (Microsecond-grade)")


if __name__ == "__main__":
    main()
