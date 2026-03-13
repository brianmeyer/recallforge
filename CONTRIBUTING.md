# Contributing to RecallForge

Thanks for your interest in contributing to RecallForge, a cross-modal vision-language search engine. This document will help you get started.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/brianmeyer/recallforge.git
   cd recallforge
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install development dependencies**
   
   For standard systems:
   ```bash
   pip install -e '.[dev,torch]'
   ```
   
   For Apple Silicon (M1/M2/M3):
   ```bash
   pip install -e '.[dev,mlx]'
   ```

4. **Run tests to verify setup**
   ```bash
   pytest -x -m 'not live'
   ```

## Development Guidelines

### Project Structure

RecallForge uses a `src/` layout. Key directories:

- `src/recallforge/` - Main package code
- `src/recallforge/storage/` - Pluggable storage backends (LanceDB, etc.)
- `src/recallforge/backends/` - ML backends (MLX, PyTorch)
- `tests/` - Test suite mirroring source structure

### Code Organization

- New features belong in appropriate submodules under `src/recallforge/`
- Storage backends go in `src/recallforge/storage/`
- ML/compute backends go in `src/recallforge/backends/`
- Tests mirror the source structure in `tests/`

## Pull Request Process

### Branching

1. Branch from `master`
2. Use descriptive branch names:
   - `feat/description` - New features
   - `fix/description` - Bug fixes
   - `docs/description` - Documentation changes
   - `refactor/description` - Code refactoring

### Before Submitting

1. **Write tests** for new features
2. **Run the test suite**:
   ```bash
   pytest -x -m 'not live'
   ```
3. **Keep PRs focused** - One issue per PR
4. **Ensure CI passes** on your branch

### Review Process

- PRs require at least one approval
- Address all review comments
- Squash commits before merge (if requested)

## Code Style

### Type Hints

All public functions must have type hints:

```python
def search(query: str, limit: int = 10) -> list[SearchResult]:
    ...
```

### Docstrings

All public classes and methods must have docstrings:

```python
class VectorStore:
    """Persistent storage for embedding vectors."""

    def add(self, items: list[Item]) -> int:
        """Add items to the store.
        
        Args:
            items: List of items to add.
            
        Returns:
            Number of items successfully added.
        """
        ...
```

### Error Handling

Never use bare `except Exception` blocks. Log errors instead of swallowing them:

```python
# Bad
try:
    process(data)
except Exception:
    pass

# Good
import logging

logger = logging.getLogger(__name__)

try:
    process(data)
except Exception as e:
    logger.error(f"Failed to process data: {e}")
    raise
```

## Testing

### Running Tests

- **Unit tests only** (no model downloads):
  ```bash
  pytest -x
  ```

- **All tests including live model tests**:
  ```bash
  pytest -m live
  ```

- **All tests**:
  ```bash
  pytest
  ```

### Marking Tests

Mark model-dependent tests with `@pytest.mark.live`:

```python
import pytest

@pytest.mark.live
def test_embedding_generation():
    """Test requires downloading/running ML models."""
    ...
```

### Test Structure

- Tests live in `tests/` mirroring `src/recallforge/` structure
- One test file per source module
- Use fixtures for common setup

## Architecture Notes

### Multi-Backend Support

RecallForge supports multiple ML backends:

- **MLX** - Apple Silicon optimized
- **PyTorch** - Cross-platform

Backends are abstracted behind a common interface. New backends should implement the same interface.

### Pluggable Storage

Storage is pluggable with LanceDB as the default:

- Implement the storage interface to add new backends
- Storage backends handle persistence and indexing
- Vector operations are backend-agnostic

### MCP Server Integration

RecallForge includes MCP (Model Context Protocol) server integration for AI agent tools:

- Exposes search capabilities to MCP-compatible agents
- Server implementation in `src/recallforge/server.py`

## Questions?

Open an issue on GitHub for:
- Bug reports
- Feature requests
- Questions about architecture or contribution approach

---

Thank you for contributing to RecallForge!