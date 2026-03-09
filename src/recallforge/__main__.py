"""
Entry point for running RecallForge as a module.

Usage:
    python -m recallforge serve
    python -m recallforge search "query"
    python -m recallforge index /path/to/files
"""

from .cli import main

if __name__ == "__main__":
    main()