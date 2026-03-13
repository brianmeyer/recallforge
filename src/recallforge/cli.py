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
from .documents import is_document_file
from .video import is_video_file


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
        nargs="?",
        help="Search query",
    )
    search_parser.add_argument(
        "--image",
        dest="image_path",
        default=None,
        help="Image query path (mutually exclusive with text query)",
    )
    search_parser.add_argument(
        "--video",
        dest="video_path",
        default=None,
        help="Video query path (mutually exclusive with text query and --image)",
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
        choices=["text", "image", "video"],
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
    
    # watch command
    watch_parser = subparsers.add_parser("watch", help="Watch folder daemon commands")
    watch_subparsers = watch_parser.add_subparsers(dest="watch_command", help="Watch commands")
    
    # watch start
    watch_start_parser = watch_subparsers.add_parser("start", help="Start watching a folder")
    watch_start_parser.add_argument(
        "folder",
        help="Folder path to watch",
    )
    watch_start_parser.add_argument(
        "--collection", "-c",
        default="default",
        help="Collection name (default: default)",
    )
    watch_start_parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Watch recursively (default: true). Use --no-recursive to disable.",
    )
    watch_start_parser.add_argument(
        "--include",
        action="append",
        help="Include glob pattern (can be specified multiple times)",
    )
    watch_start_parser.add_argument(
        "--exclude",
        action="append",
        help="Exclude glob pattern (can be specified multiple times)",
    )
    watch_start_parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Debounce seconds (default: 2.0)",
    )
    watch_start_parser.add_argument(
        "--store-path",
        default=None,
        help="Path to storage directory",
    )
    
    # watch stop
    watch_stop_parser = watch_subparsers.add_parser("stop", help="Stop watching a folder")
    watch_stop_parser.add_argument(
        "watch_id",
        nargs="?",
        default=None,
        help="Watch ID to stop (or 'all' to stop all)",
    )
    watch_stop_parser.add_argument(
        "--store-path",
        default=None,
        help="Path to storage directory",
    )
    
    # watch list
    watch_list_parser = watch_subparsers.add_parser("list", help="List active watches")
    watch_list_parser.add_argument(
        "--store-path",
        default=None,
        help="Path to storage directory",
    )
    
    # watch status
    watch_status_parser = watch_subparsers.add_parser("status", help="Show watch status")
    watch_status_parser.add_argument(
        "watch_id",
        nargs="?",
        default=None,
        help="Watch ID to check status for",
    )
    watch_status_parser.add_argument(
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
    elif args.command == "watch":
        return cmd_watch(args)
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
            elif _is_video_file(path):
                print(f"Indexing video: {path}")
                storage.index_video(
                    path=path,
                    collection=args.collection,
                    embed_text_func=backend.embed_text,
                    embed_image_func=backend.embed_image,
                    embed_video_func=getattr(backend, "embed_video", None),
                )
            elif _is_document_file(path):
                print(f"Indexing document: {path}")
                storage.index_document_file(
                    path=path,
                    collection=args.collection,
                    embed_func=backend.embed_text,
                    model="Qwen3-VL-Embedding-2B",
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
                dirs[:] = [
                    dirname for dirname in sorted(dirs)
                    if dirname not in {".git", "node_modules", "__pycache__", ".venv", "venv"}
                    and not dirname.startswith(".")
                ]
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
                        elif _is_video_file(fp):
                            print(f"Indexing video: {fp}")
                            storage.index_video(
                                path=fp,
                                collection=args.collection,
                                embed_text_func=backend.embed_text,
                                embed_image_func=backend.embed_image,
                                embed_video_func=getattr(backend, "embed_video", None),
                            )
                        elif _is_document_file(fp):
                            print(f"Indexing document: {fp}")
                            storage.index_document_file(
                                path=fp,
                                collection=args.collection,
                                embed_func=backend.embed_text,
                                model="Qwen3-VL-Embedding-2B",
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


def _is_video_file(path: str) -> bool:
    """Check if file is a video."""
    return is_video_file(path)


def _is_document_file(path: str) -> bool:
    """Check if file is a supported office document."""
    return is_document_file(path)


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

    query_text = args.query.strip() if isinstance(args.query, str) else ""
    image_path = args.image_path.strip() if isinstance(args.image_path, str) else ""
    video_path = args.video_path.strip() if isinstance(args.video_path, str) else ""
    provided = [bool(query_text), bool(image_path), bool(video_path)]
    if sum(provided) != 1:
        print("Provide exactly one of: query, --image, or --video", file=sys.stderr)
        return 2

    if image_path:
        results = searcher.search_image(image_path)
        print(f"\nResults for image: '{image_path}'\n")
    elif video_path:
        results = searcher.search_video(video_path)
        print(f"\nResults for video: '{video_path}'\n")
    else:
        results = searcher.search(query_text)
        print(f"\nResults for: '{query_text}'\n")

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



def cmd_watch(args):
    """Manage watch-folder daemon."""
    from . import get_backend, get_storage
    from .watch_folder import WatchConfig, get_daemon

    store_path = args.store_path or os.environ.get("RECALLFORGE_STORE_PATH")
    storage = get_storage(store_path)
    backend = get_backend()
    daemon = get_daemon(storage, backend, store_path)

    if args.watch_command == "start":
        includes = args.include or [
            "**/*.md", "**/*.txt",
            "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.webp",
            "**/*.mp4", "**/*.mov", "**/*.m4v", "**/*.avi", "**/*.mkv", "**/*.webm",
        ]
        excludes = args.exclude or ["**/.git/**", "**/node_modules/**"]
        config = WatchConfig(
            folder_path=args.folder,
            collection=args.collection,
            recursive=args.recursive,
            include_globs=includes,
            exclude_globs=excludes,
            debounce_seconds=args.debounce,
        )
        watch_id = daemon.start_watch(config)
        print(json.dumps({"watch_id": watch_id, "status": "running", "folder": args.folder}, indent=2))
        return 0

    if args.watch_command == "stop":
        if args.watch_id == "all":
            stopped = daemon.stop_all()
            print(json.dumps({"stopped": stopped}, indent=2))
            return 0
        if not args.watch_id:
            print("watch stop requires watch_id or 'all'", file=sys.stderr)
            return 2
        ok = daemon.stop_watch(args.watch_id)
        print(json.dumps({"watch_id": args.watch_id, "stopped": bool(ok)}, indent=2))
        return 0 if ok else 1

    if args.watch_command == "list":
        print(json.dumps(daemon.list_watches(), indent=2))
        return 0

    if args.watch_command == "status":
        if args.watch_id:
            status = daemon.get_watch_status(args.watch_id)
            print(json.dumps(status or {"error": "not_found", "watch_id": args.watch_id}, indent=2))
            return 0 if status else 1
        print(json.dumps(daemon.list_watches(), indent=2))
        return 0

    print("watch requires subcommand: start|stop|list|status", file=sys.stderr)
    return 2

if __name__ == "__main__":
    sys.exit(main())
