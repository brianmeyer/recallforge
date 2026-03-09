"""
test_expand.py - Tests for query expansion functionality.
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

import expand
from src import store
from src import db as expand_db
from expand import (
    expand_query,
    _get_expand_cache_key,
    clear_expand_cache,
    strong_signal_detected,
    ExpandedQuery,
)


class TestExpandQuery(unittest.TestCase):
    """Tests for expand_query function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for database
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch store cache functions (expand uses store.get_cached_result)
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
    
    def test_get_expand_cache_key(self):
        """Test cache key generation is deterministic."""
        query = "test query"
        key1 = _get_expand_cache_key(query)
        key2 = _get_expand_cache_key(query)
        self.assertEqual(key1, key2)
        
        # Different queries should have different keys
        key3 = _get_expand_cache_key("different query")
        self.assertNotEqual(key1, key3)
    
    def test_strong_signal_detected_no_results(self):
        """Test no strong signal with no results (expansion needed)."""
        self.assertFalse(strong_signal_detected([]))
    
    def test_strong_signal_detected_single_result(self):
        """Test no strong signal with single result (not enough evidence)."""
        mock_result = MagicMock()
        mock_result.score = 0.9
        self.assertFalse(strong_signal_detected([mock_result]))
    
    def test_strong_signal_detected_strong(self):
        """Test strong signal detection with clear winner."""
        r1 = MagicMock()
        r1.score = 0.95
        r2 = MagicMock()
        r2.score = 0.3
        self.assertTrue(strong_signal_detected([r1, r2]))
    
    def test_strong_signal_detected_weak(self):
        """Test no strong signal when scores are close."""
        r1 = MagicMock()
        r1.score = 0.6
        r2 = MagicMock()
        r2.score = 0.55
        self.assertFalse(strong_signal_detected([r1, r2]))
    
    def test_expand_query_uses_cache(self):
        """Test that expand_query uses cache."""
        query = "test query"
        
        # Pre-populate cache with a mock result
        cache_key = _get_expand_cache_key(query)
        mock_result = json.dumps([
            {"original": query, "type": "lex", "text": "keywords"},
        ])
        store.set_cached_result(cache_key, mock_result)
        
        results = expand_query(query)
        
        # Should get from cache
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].type, "lex")
    
    def test_expand_query_caches_result(self):
        """Test that expand_query caches results."""
        query = "another test"
        
        # Mock model response
        with patch('expand._load_expand_model'), \
             patch('expand._generate_expansions') as mock_gen:
            
            mock_gen.return_value = [
                ExpandedQuery(original=query, type="lex", text="expansion text")
            ]
            
            # Clear cache first
            store.cache = {}
            
            results = expand_query(query)
            
            # Check cache was populated
            key = _get_expand_cache_key(query)
            cached = store.get_cached_result(key)
            self.assertIsNotNone(cached)
            
            # Parse and verify
            import json
            cached_data = json.loads(cached)
            self.assertEqual(len(cached_data), 1)
            self.assertEqual(cached_data[0]['type'], 'lex')
    
    def test_expand_query_fallback_on_error(self):
        """Test fallback to fast expansion on model error."""
        query = "error test"
        
        with patch('expand._load_expand_model'), \
             patch('expand._generate_expansions') as mock_gen:
            
            mock_gen.side_effect = Exception("Model error")
            
            # Should still return something (fast fallback)
            results = expand_query(query)
            
            # Should have at least a vec variant
            types = [r.type for r in results]
            self.assertIn('vec', types)


class TestExpandIntegration(unittest.TestCase):
    """Integration tests requiring database."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        
        expand_db = expand.db
        expand_db.DEFAULT_INDEX_DIR = self.temp_dir
        
        # Initialize database
        expand_db.initialize_database(self.temp_dir)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_clear_expand_cache(self):
        """Test clearing expand cache."""
        expand_db = expand.db
        
        # Add some cache entries
        expand_db.cache_table.add([{
            "key": "expand:test1",
            "value": '{"type": "lex", "text": "test"}',
            "created_at": 1234567890,
        }])
        
        # Clear cache
        clear_expand_cache()
        
        # Verify cache is empty
        rows = list(expand_db.cache_table.search().limit(100).to_list())
        expand_rows = [r for r in rows if r.get('key', '').startswith('expand:')]
        self.assertEqual(len(expand_rows), 0)


if __name__ == '__main__':
    unittest.main()
