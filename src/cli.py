"""
cli.py - CLI entry point for QMD-VL.

Commands:
- qmd-vl serve          Start MCP server
- qmd-vl index <path>   Index a file or directory
- qmd-vl search <query> Search with hybrid pipeline
- qmd-vl status         Show server status
- qmd-vl warmup         Warm up all models

Optional --profile flag for per-stage timing.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src import db
from src.models import warm_up, get_registry, status as model_status
from src.server import run_server as run_mcp_server
from src.search import hybrid_query
from src.store import index_document, insert_image, search_fts, search_vec


def _profile(func, *args, **kwargs):
    """Run function and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def cmd_serve(args):
    """Start MCP server."""
    print("Starting QMD-VL MCP server...")
    run_mcp_server()


def cmd_index(args):
    """Index a file or directory."""
    path = Path(args.path).resolve()
    
    if not path.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize database
    db.initialize_database(args.store)
    
    # Warm up models
    registry = get_registry()
    
    if path.is_file():
        files = [path]
    else:
        # Index all files in directory
        files = list(path.rglob("*"))
        files = [f for f in files if f.is_file()]
    
    indexed = 0
    errors = 0
    
    for file_path in files:
        try:
            # Determine if image or text
            ext = file_path.suffix.lower()
            image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
            
            if ext in image_exts:
                # Index image
                hash_val = insert_image(
                    path=str(file_path),
                    collection=args.collection,
                    embed_func=registry.embed_image,
                )
                print(f"Indexed image: {file_path} ({hash_val[:8]}...)")
            else:
                # Index text
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                hash_val = index_document(
                    path=str(file_path.relative_to(path.parent if path.is_file() else path)),
                    text=text,
                    collection=args.collection,
                    model="Qwen3-VL-Embedding-2B",
                    embed_func=registry.embed_text,
                )
                print(f"Indexed: {file_path} ({hash_val[:8]}...)")
            
            indexed += 1
            
        except Exception as e:
            print(f"Error indexing {file_path}: {e}", file=sys.stderr)
            errors += 1
    
    print(f"\nIndexed {indexed} files, {errors} errors")


def cmd_search(args):
    """Search with hybrid pipeline."""
    query = args.query
    
    if not query:
        print("Error: Query is required", file=sys.stderr)
        sys.exit(1)
    
    # Initialize database
    db.initialize_database(args.store)
    
    # Warm up models if not already loaded
    registry = get_registry()
    
    if args.profile:
        print("Running with profiling...\n")
        
        # Profile each stage
        # Stage 1: BM25 probe + expansion
        stage1_start = time.perf_counter()
        from src.store import search_fts
        from src.expand import expand_query
        
        fts_results, fts_time = _profile(
            search_fts, query, limit=20, collection=args.collection
        )
        print(f"  BM25 probe: {fts_time:.1f}ms")
        
        expansions, expand_time = _profile(
            expand_query, query, fts_results
        )
        print(f"  Query expansion: {expand_time:.1f}ms")
        
        stage1_time = (time.perf_counter() - stage1_start) * 1000
        print(f"Stage 1 total: {stage1_time:.1f}ms")
        
        # Stage 2: Parallel searches (embedded in hybrid_query)
        # We can't easily break this out, so we time the whole thing
        results, total_time = _profile(
            hybrid_query,
            query=query,
            limit=args.limit,
            collection=args.collection,
        )
        
        print(f"\nStage 2 (search + fusion): {total_time - stage1_time:.1f}ms")
        print(f"Total search time: {total_time:.1f}ms")
        
    else:
        # Normal search
        results = hybrid_query(
            query=query,
            limit=args.limit,
            collection=args.collection,
        )
    
    # Output results
    print(f"\nFound {len(results)} results for: {query}\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}")
        print(f"   Path: {r.filepath}")
        print(f"   Score: {r.score:.4f} (rerank: {r.rerank_score:.4f}, rrf_rank: {r.rrf_rank})")
        print(f"   Source: {r.source}")
        if r.body:
            snippet = r.body[:200].replace("\n", " ")
            print(f"   Snippet: {snippet}...")
        print()


def cmd_status(args):
    """Show server status."""
    # Initialize database
    db.initialize_database(args.store)
    
    # Get model status
    models = model_status()
    
    # Get database info
    db_info = {}
    if db.embeddings_table is not None:
        try:
            db_info["embeddings_count"] = db.embeddings_table.count_rows()
        except:
            db_info["embeddings_count"] = 0
    if db.documents_table is not None:
        try:
            db_info["documents_count"] = db.documents_table.count_rows()
        except:
            db_info["documents_count"] = 0
    
    print("QMD-VL Status")
    print("=" * 40)
    print("\nModels:")
    print(f"  Embedder loaded: {models['embedder_loaded']}")
    print(f"  Reranker loaded: {models['reranker_loaded']}")
    print(f"  Expander loaded: {models['expander_loaded']}")
    print(f"  Device: {models['device']}")
    print(f"  Dtype: {models['dtype']}")
    if models.get('memory_allocated_gb'):
        print(f"  Memory allocated: {models['memory_allocated_gb']:.2f} GB")
    
    print("\nDatabase:")
    print(f"  Embeddings: {db_info.get('embeddings_count', 'N/A')}")
    print(f"  Documents: {db_info.get('documents_count', 'N/A')}")


def cmd_warmup(args):
    """Warm up all models."""
    print("Warming up QMD-VL models...")
    warm_up()
    print("All models ready.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="qmd-vl",
        description="QMD Vision-Language Memory Search",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # serve
    serve_parser = subparsers.add_parser("serve", help="Start MCP server")
    
    # index
    index_parser = subparsers.add_parser("index", help="Index a file or directory")
    index_parser.add_argument("path", help="Path to file or directory")
    index_parser.add_argument(
        "--collection", "-c",
        default="default",
        help="Collection name (default: default)"
    )
    index_parser.add_argument(
        "--store", "-s",
        help="Store directory (default: ~/.qmd)"
    )
    
    # search
    search_parser = subparsers.add_parser("search", help="Search with hybrid pipeline")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Maximum results (default: 10)"
    )
    search_parser.add_argument(
        "--collection", "-c",
        help="Filter by collection"
    )
    search_parser.add_argument(
        "--store", "-s",
        help="Store directory (default: ~/.qmd)"
    )
    search_parser.add_argument(
        "--profile", "-p",
        action="store_true",
        help="Show per-stage timing"
    )
    
    # status
    status_parser = subparsers.add_parser("status", help="Show server status")
    status_parser.add_argument(
        "--store", "-s",
        help="Store directory (default: ~/.qmd)"
    )
    
    # warmup
    warmup_parser = subparsers.add_parser("warmup", help="Warm up all models")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch
    commands = {
        "serve": cmd_serve,
        "index": cmd_index,
        "search": cmd_search,
        "status": cmd_status,
        "warmup": cmd_warmup,
    }
    
    func = commands.get(args.command)
    if func:
        func(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()