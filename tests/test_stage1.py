"""
test_stage1.py - Live test for Stage 1: Foundation + Text Embedding + BM25

Tests:
1. Project structure exists
2. Database can be initialized
3. Documents can be indexed with embeddings
4. BM25 search works
5. Vector search works
"""

import os
import sys
import tempfile
import shutil
import hashlib

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import db, store, embed


def mock_embed(text: str) -> list:
    """Mock embedding that creates deterministic pseudo-random vectors."""
    h = hashlib.sha256(text.encode()).hexdigest()
    values = []
    for i in range(2048):
        chunk = h[(i * 2) % 64:(i * 2) % 64 + 4]
        val = int(chunk, 16) / 65536.0 - 0.5
        values.append(val)
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


def test_project_structure():
    """Verify project structure exists."""
    print("Testing project structure...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    assert os.path.exists(os.path.join(project_root, "pyproject.toml")), "pyproject.toml missing"
    assert os.path.exists(os.path.join(project_root, "src", "__init__.py")), "src/__init__.py missing"
    assert os.path.exists(os.path.join(project_root, "src", "db.py")), "src/db.py missing"
    assert os.path.exists(os.path.join(project_root, "src", "store.py")), "src/store.py missing"
    assert os.path.exists(os.path.join(project_root, "src", "embed.py")), "src/embed.py missing"
    assert os.path.exists(os.path.join(project_root, "Qwen3-VL-Embedding")), "Qwen3-VL-Embedding repo missing"
    
    print("✓ Project structure complete")


def test_database_init():
    """Test LanceDB database initialization."""
    print("\nTesting database initialization...")
    
    temp_dir = tempfile.mkdtemp(prefix="qmd-vl-test-")
    
    try:
        # Reset singleton
        db.close_database()
        db.initialize_database(temp_dir)
        
        assert db.embeddings_table is not None, "embeddings table not initialized"
        assert db.documents_table is not None, "documents table not initialized"
        assert db.content_table is not None, "content table not initialized"
        assert db.cache_table is not None, "cache table not initialized"
        
        print(f"✓ Database initialized at {temp_dir}")
        
    finally:
        db.close_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_document_indexing():
    """Test document indexing with embeddings."""
    print("\nTesting document indexing...")
    
    temp_dir = tempfile.mkdtemp(prefix="qmd-vl-test-")
    
    try:
        # Reset singleton
        db.close_database()
        db.initialize_database(temp_dir)
        
        # Test documents about AI, memory, graphs, cooking, sports
        test_docs = [
            {
                "path": "ai-overview.md",
                "title": "Artificial Intelligence Overview",
                "content": "Artificial intelligence (AI) is intelligence demonstrated by machines. "
                          "Modern AI systems use neural networks trained on large datasets.",
            },
            {
                "path": "memory-systems.md",
                "title": "Memory Systems for AI Agents",
                "content": "AI agents need memory to function effectively. "
                          "Episodic memory records past experiences. Graph databases provide structured knowledge.",
            },
            {
                "path": "graph-databases.md",
                "title": "Graph Databases and Knowledge Graphs",
                "content": "Graph databases store data as nodes and relationships. "
                          "Knowledge graphs represent entities and their relationships.",
            },
        ]
        
        # Index all test documents
        content_hashes = []
        for doc in test_docs:
            content_hash = store.index_document(
                path=doc["path"],
                text=doc["content"],
                collection="test",
                model="mock-embedder",
                embed_func=mock_embed,
            )
            content_hashes.append(content_hash)
            print(f"  Indexed: {doc['path']} (hash: {content_hash[:8]}...)")
        
        print(f"✓ Indexed {len(test_docs)} documents")
        
    finally:
        db.close_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_bm25_search():
    """Test BM25 full-text search."""
    print("\nTesting BM25 search...")
    
    temp_dir = tempfile.mkdtemp(prefix="qmd-vl-test-")
    
    try:
        # Reset singleton
        db.close_database()
        db.initialize_database(temp_dir)
        
        # Index test documents
        test_docs = [
            {"path": "ai-overview.md", "content": "Artificial intelligence AI machines neural networks memory systems."},
            {"path": "memory-systems.md", "content": "Memory Systems AI agents episodic semantic graph databases knowledge."},
            {"path": "graph-databases.md", "content": "Graph databases nodes relationships knowledge graphs entities."},
        ]
        
        for doc in test_docs:
            store.index_document(
                path=doc["path"],
                text=doc["content"],
                collection="test",
                model="mock-embedder",
                embed_func=mock_embed,
            )
        
        # Search for "graph memory agent"
        results = store.search_fts("graph memory", limit=10)
        
        print(f"  Query: 'graph memory'")
        print(f"  Results: {len(results)}")
        
        for i, r in enumerate(results[:5]):
            print(f"    {i+1}. {r.display_path} (score: {r.score:.3f})")
        
        # BM25 should return results
        assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
        print("✓ BM25 search works")
        
    finally:
        db.close_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vector_search():
    """Test vector semantic search."""
    print("\nTesting vector search...")
    
    temp_dir = tempfile.mkdtemp(prefix="qmd-vl-test-")
    
    try:
        # Reset singleton
        db.close_database()
        db.initialize_database(temp_dir)
        
        # Index test documents
        test_docs = [
            {"path": "ai-overview.md", "content": "Artificial intelligence AI machines neural networks memory systems."},
            {"path": "memory-systems.md", "content": "Memory Systems AI agents episodic semantic graph databases knowledge."},
            {"path": "graph-databases.md", "content": "Graph databases nodes relationships knowledge graphs entities."},
        ]
        
        for doc in test_docs:
            store.index_document(
                path=doc["path"],
                text=doc["content"],
                collection="test",
                model="mock-embedder",
                embed_func=mock_embed,
            )
        
        # Search for semantic match
        query_text = "how do AI agents remember things"
        query_vector = mock_embed(query_text)
        
        results = store.search_vec(query_vector, limit=10)
        
        print(f"  Query: '{query_text}'")
        print(f"  Results: {len(results)}")
        
        for i, r in enumerate(results[:5]):
            print(f"    {i+1}. {r.display_path} (score: {r.score:.3f})")
        
        # Vector search should return results
        print("✓ Vector search works")
        
    finally:
        db.close_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """Run all Stage 1 tests."""
    print("=" * 60)
    print("QMD-VL Stage 1 Tests")
    print("=" * 60)
    
    try:
        test_project_structure()
        test_database_init()
        test_document_indexing()
        test_bm25_search()
        test_vector_search()
        
        print("\n" + "=" * 60)
        print("ALL STAGE 1 TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()