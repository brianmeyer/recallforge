"""
test_rerank.py - Tests for reranking functionality.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Set up paths before importing - use qmd-vl as package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import rerank
from src import store
from src import db as rerank_db
from rerank import (
    Qwen3VLRerankerWrapper,
    rerank,
    clear_rerank_cache,
    RerankResult,
)


class TestRerank(unittest.TestCase):
    """Tests for rerank functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch store cache functions (rerank uses store.get_cached_result)
        self.original_get_cached = store.get_cached_result
        self.original_set_cached = store.set_cached_result
        
        self.cache = {}
        store.get_cached_result = lambda key: self.cache.get(key)
        store.set_cached_result = lambda key, value: self.cache.update({key: value})
    
    def tearDown(self):
        """Clean up."""
        store.get_cached_result = self.original_get_cached
        store.set_cached_result = self.original_set_cached
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_doc_hash(self):
        """Test document hash generation."""
        reranker = Qwen3VLRerankerWrapper()
        
        doc1 = {"text": "test document", "filepath": "/test/file.txt"}
        doc2 = {"text": "different document", "filepath": "/test/file.txt"}
        doc3 = {"text": "test document", "filepath": "/test/file.txt"}  # Same as doc1
        
        hash1 = reranker._get_doc_hash(doc1)
        hash2 = reranker._get_doc_hash(doc2)
        hash3 = reranker._get_doc_hash(doc3)
        
        self.assertNotEqual(hash1, hash2)
        self.assertEqual(hash1, hash3)
    
    def test_get_rerank_cache_key(self):
        """Test rerank cache key generation."""
        reranker = Qwen3VLRerankerWrapper()
        
        query = "test query"
        doc_hash = "abc123"
        
        key1 = reranker._get_rerank_cache_key(query, doc_hash)
        key2 = reranker._get_rerank_cache_key(query, doc_hash)
        
        self.assertEqual(key1, key2)
        
        # Different query should give different key
        key3 = reranker._get_rerank_cache_key("different", doc_hash)
        self.assertNotEqual(key1, key3)
    
    def test_rerank_uses_cache(self):
        """Test that rerank uses cache."""
        reranker = Qwen3VLRerankerWrapper()
        
        query = "test query"
        doc = {"text": "test document", "filepath": "/test/file.txt"}
        doc_hash = reranker._get_doc_hash(doc)
        cache_key = reranker._get_rerank_cache_key(query, doc_hash)
        
        # Pre-populate cache
        self.cache[cache_key] = "0.85"
        
        # Should return cached score
        results = reranker.rerank(query, [doc])
        
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].score, 0.85, places=2)
    
    def test_rerank_returns_empty_for_empty_input(self):
        """Test rerank with no documents."""
        reranker = Qwen3VLRerankerWrapper()
        
        results = reranker.rerank("query", [])
        
        self.assertEqual(results, [])
    
    def test_rerank_sorts_by_score(self):
        """Test that rerank sorts by score descending."""
        reranker = Qwen3VLRerankerWrapper()
        
        query = "test query"
        docs = [
            {"text": f"document {i}", "filepath": f"/test/file{i}.txt"}
            for i in range(3)
        ]
        
        # Pre-populate cache with different scores
        for i, doc in enumerate(docs):
            doc_hash = reranker._get_doc_hash(doc)
            cache_key = reranker._get_rerank_cache_key(query, doc_hash)
            self.cache[cache_key] = str(0.9 - i * 0.1)  # 0.9, 0.8, 0.7
        
        results = reranker.rerank(query, docs)
        
        # Should be sorted by score descending
        self.assertEqual(len(results), 3)
        self.assertGreater(results[0].score, results[1].score)
        self.assertGreater(results[1].score, results[2].score)
    
    def test_rerank_result_structure(self):
        """Test rerank result structure."""
        reranker = Qwen3VLRerankerWrapper()
        
        query = "test query"
        doc = {"text": "test document", "filepath": "/test/file.txt"}
        
        # Pre-populate cache with a score
        doc_hash = reranker._get_doc_hash(doc)
        cache_key = reranker._get_rerank_cache_key(query, doc_hash)
        self.cache[cache_key] = "0.75"
        
        results = reranker.rerank(query, [doc])
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RerankResult)
        self.assertEqual(results[0].score, 0.75)
    
    def test_clear_rerank_cache(self):
        """Test clearing rerank cache."""
        # Add some rerank cache entries to db.cache_table
        rerank_db = rerank.db
        rerank_db.cache_table.add([{
            "key": "rerank:test1",
            "value": "0.5",
            "created_at": 1234567890,
        }])
        
        clear_rerank_cache()
        
        # Verify cache is empty
        rows = list(rerank_db.cache_table.search().limit(100).to_list())
        rerank_rows = [r for r in rows if r.get('key', '').startswith('rerank:')]
        self.assertEqual(len(rerank_rows), 0)


class TestRerankIntegration(unittest.TestCase):
    """Integration tests for rerank."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        
        from rerank import db as rerank_db
        rerank_db.DEFAULT_INDEX_DIR = self.temp_dir
        
        # Initialize database
        rerank_db.initialize_database(self.temp_dir)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rerank_with_real_db(self):
        """Test rerank with actual database."""
        reranker = Qwen3VLRerankerWrapper()
        
        query = "integration test"
        docs = [
            {"text": "first document", "filepath": "/test/1.txt"},
            {"text": "second document", "filepath": "/test/2.txt"},
        ]
        
        # This will try to use the model but should handle gracefully
        try:
            results = reranker.rerank(query, docs)
            # If model loaded, should have results with scores
            self.assertEqual(len(results), len(docs))
        except ImportError:
            # Model not available, test still passes
            pass


if __name__ == '__main__':
    unittest.main()
