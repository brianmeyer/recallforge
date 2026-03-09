"""
CLI for RecallForge.

Commands:
  recallforge serve       Start MCP server
  recallforge index       Index a file or directory
  recallforge search      Run a search query
  recallforge status      Show system status
"""

import argparse
import json
import os
import sys

from . import __version__, RECALLFORGE_BACKEND, RECALLFORGE_MODE, RECALLFORGE_STORAGE


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="recallforge",
        description="RecallForge - Cross-Modal Vision-Language Search Engine",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"RecallForge {__version__}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start MCP server")
    serve_parser.add_argument(
        "--mode", "-m",
        choices=["embed", "hybrid", "full"],
        default=None,
        help="Search mode (default: from RECALLFORGE_MODE env)",
    )
    serve_parser.add_argument(
        "--backend", "-b",
        choices=["torch", "mlx", "auto"],
        default=None,
        help="Model backend (default: from RECALLFORGE_BACKEND env)",
    )
    serve_parser.add_argument(
        "--store-path",
        default=None,
        help="Path to storage directory (default: ~/.recallforge)",
    )
    serve_parser.add_argument(
        "--quantize",
        choices=["bf16", "4bit"],
        default=None,
        help="MLX quantization (default: bf16)",
    )
    
    # index command
    index_parser = subparsers.add_parser("index", help="Index files")
    index_parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to index",
    )
    index_parser.add_argument(
        "--collection", "-c",
        default="default",
        help="Collection name (default: default)",
    )
    index_parser.add_argument(
        "--store-path",
        default=None,
        help="Path to storage directory",
    )
    
    # search command
    search_parser = subparsers.add_parser("search", help="Search indexed content")
    search_parser.add_argument(
        "query",
        help="Search query",
    )
    search_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Max results (default: 10)",
    )
    search_parser.add_argument(
        "--collection", "-c",
        default=None,
        help="Filter by collection",
    )
    search_parser.add_argument(
        "--content-type",
        choices=["text", "image"],
        default=None,
        help="Filter by content type",
    )
    search_parser.add_argument(
        "--mode", "-m",
        choices=["embed", "hybrid", "full"],
        default=None,
        help="Search mode",
    )
    search_parser.add_argument(
        "--store-path",
        default=None,
        help="Path to storage directory",
    )
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument(
        "--store-path",
        default=None,
        help="Path to storage directory",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Handle commands
    if args.command == "serve":
        return cmd_serve(args)
    elif args.command == "index":
        return cmd_index(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 1


def cmd_serve(args):
    """Start the MCP server."""
    # Set environment from CLI args
    if args.mode:
        os.environ["RECALLFORGE_MODE"] = args.mode
    if args.backend:
        os.environ["RECALLFORGE_BACKEND"] = args.backend
    if args.quantize:
        os.environ["RECALLFORGE_MLX_QUANTIZE"] = args.quantize
    if args.store_path:
        os.environ["RECALLFORGE_STORE_PATH"] = args.store_path
    
    from .server import run_server
    run_server()
    return 0


def cmd_index(args):
    """Index files."""
    from . import get_backend, get_storage
    
    store_path = args.store_path or os.environ.get("RECALLFORGE_STORE_PATH")
    storage = get_storage(store_path)
    backend = get_backend()
    
    # Warm up embedder only
    backend._load_embedder()
    
    indexed = 0
    for path in args.paths:
        if os.path.isfile(path):
            if _is_image_file(path):
                print(f"Indexing image: {path}")
                storage.index_image(
                    path=path,
                    collection=args.collection,
                    embed_func=backend.embed_image,
                )
            else:
                print(f"Indexing file: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                storage.index_document(
                    path=os.path.basename(path),
                    text=text,
                    collection=args.collection,
                    model="Qwen3-VL-Embedding-2B",
                    embed_func=backend.embed_text,
                )
            indexed += 1
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.startswith('.') or f.startswith('_'):
                        continue
                    fp = os.path.join(root, f)
                    try:
                        if _is_image_file(fp):
                            print(f"Indexing image: {fp}")
                            storage.index_image(
                                path=fp,
                                collection=args.collection,
                                embed_func=backend.embed_image,
                            )
                        else:
                            print(f"Indexing file: {fp}")
                            with open(fp, 'r', encoding='utf-8') as file:
                                text = file.read()
                            storage.index_document(
                                path=fp,
                                text=text,
                                collection=args.collection,
                                model="Qwen3-VL-Embedding-2B",
                                embed_func=backend.embed_text,
                            )
                        indexed += 1
                    except Exception as e:
                        print(f"  Error indexing {fp}: {e}")
    
    print(f"\nIndexed {indexed} items")
    return 0


def _is_image_file(path: str) -> bool:
    """Check if file is an image."""
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}


def cmd_search(args):
    """Run a search query."""
    from . import get_backend, get_storage
    
    if args.mode:
        os.environ["RECALLFORGE_MODE"] = args.mode
    
    store_path = args.store_path or os.environ.get("RECALLFORGE_STORE_PATH")
    storage = get_storage(store_path)
    backend = get_backend()
    
    from .search import HybridSearcher
    
    searcher = HybridSearcher(
        backend=backend,
        storage=storage,
        limit=args.limit,
        collection=args.collection,
        content_type=args.content_type,
    )
    
    results = searcher.search(args.query)
    
    print(f"\nResults for: '{args.query}'\n")
    for i, r in enumerate(results):
        print(f"{i+1}. [{r.score:.3f}] {r.title}")
        print(f"   {r.display_path}")
        print(f"   RRF rank: {r.rrf_rank}, Rerank: {r.rerank_score:.3f}")
        if r.body:
            snippet = r.body[:200].replace('\n', ' ')
            print(f"   {snippet}...")
        print()
    
    return 0


def cmd_status(args):
    """Show system status."""
    from . import get_backend, get_storage
    
    store_path = args.store_path or os.environ.get("RECALLFORGE_STORE_PATH")
    storage = get_storage(store_path)
    backend = get_backend()
    
    info = backend.get_info()
    
    print("RecallForge Status")
    print("=" * 40)
    print(f"Version:        {__version__}")
    print(f"Backend:        {info.name}")
    print(f"Device:         {info.device}")
    print(f"Dtype:          {info.dtype}")
    print(f"Mode:           {info.embedder_loaded and 'loaded' or 'not loaded'}")
    if info.quantization:
        print(f"Quantization:   {info.quantization}")
    print()
    print("Models:")
    print(f"  Embedder:     {'✓' if info.embedder_loaded else '✗'}")
    print(f"  Reranker:     {'✓' if info.reranker_loaded else '✗'}")
    print(f"  Expander:     {'✓' if info.expander_loaded else '✗'}")
    print(f"  Memory:       {info.memory_allocated_gb:.1f} GB")
    print()
    print("Storage:")
    print(f"  Embeddings:   {storage.count_embeddings()}")
    print(f"  Documents:    {storage.count_documents()}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())