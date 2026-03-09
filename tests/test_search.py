"""
test_search.py - Tests for hybrid search pipeline.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# Set up paths before importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src import store
from src import db as src_db
from src.search import (
    HybridSearcher,
    hybrid_query,
    clear_expand_and_rerank_caches,
    HybridResult,
)
from src.store import SearchResult


class TestHybridSearcher(unittest.TestCase):
    """Tests for HybridSearcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Reset the database singleton
        src_db.close_database()
        src_db.initialize_database(self.temp_dir)
        
        self.searcher = HybridSearcher(limit=10, collection="test")
        
        # Mock results
        self.mock_fts_results = [
            SearchResult(
                filepath="qmd://test/file1.txt",
                display_path="test/file1.txt",
                title="Doc 1",
                context=None,
                hash="hash1",
                docid="hash1",
                collection="test",
                modified_at="",
                body_length=100,
                score=0.95,
                source="fts",
                body="Document 1 content",
            ),
            SearchResult(
                filepath="qmd://test/file2.txt",
                display_path="test/file2.txt",
                title="Doc 2",
                context=None,
                hash="hash2",
                docid="hash2",
                collection="test",
                modified_at="",
                body_length=200,
                score=0.85,
                source="fts",
                body="Document 2 content",
            ),
        ]
        
        self.mock_vec_results = [
            SearchResult(
                filepath="qmd://test/file2.txt",
                display_path="test/file2.txt",
                title="Doc 2",
                context=None,
                hash="hash2",
                docid="hash2",
                collection="test",
                modified_at="",
                body_length=200,
                score=0.90,
                source="vec",
                body="Document 2 content",
            ),
        ]
    
    def tearDown(self):
        """Clean up."""
        src_db.close_database()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_bm25_probe(self):
        """Test BM25 probe returns results."""
        with patch('src.search.search_fts', return_value=self.mock_fts_results):
            query = "test query"
            results = self.searcher._bm25_probe(query)
            
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].score, 0.95)
    
    def test_vector_search(self):
        """Test vector search returns results."""
        import numpy as np
        mock_vector = np.random.rand(2048).astype(np.float32)
        
        with patch('src.search.embed_text', return_value=mock_vector), \
             patch('src.search.search_vec', return_value=self.mock_vec_results):
            query = "test query"
            results = self.searcher._vector_search(query)
            
            self.assertEqual(len(results), 1)
    
    def test_reciprocal_rank_fusion(self):
        """Test RRF fusion combines results correctly."""
        all_results = {
            'original_fts': self.mock_fts_results,
            'original_vec': self.mock_vec_results,
        }
        
        fused = self.searcher._reciprocal_rank_fusion(all_results)
        
        # Should have unique results (2 unique filepaths)
        self.assertGreater(len(fused), 0)
        
        # All results should have non-zero RRF score
        for r in fused:
            self.assertGreater(r.score, 0)
    
    def test_select_best_chunk(self):
        """Test best chunk selection."""
        result = self.mock_fts_results[0]
        
        chunk = self.searcher._select_best_chunk(result)
        
        self.assertIn('text', chunk)
        self.assertIn('filepath', chunk)
    
    def test_rerank_candidates(self):
        """Test candidate reranking."""
        candidates = self.mock_fts_results[:1]
        
        with patch('src.search.rerank') as mock_rerank:
            mock_rerank.return_value = [
                MagicMock(document={'filepath': 'qmd://test/file1.txt'}, score=0.8)
            ]
            
            rerank_scores = self.searcher._rerank_candidates(candidates, "test query")
            
            # Should return scores for each candidate
            self.assertIsInstance(rerank_scores, dict)
    
    def test_blend_scores(self):
        """Test score blending."""
        rrf_results = self.mock_fts_results
        
        rerank_scores = {
            "qmd://test/file1.txt": 0.8,
            "qmd://test/file2.txt": 0.7,
        }
        
        blended = self.searcher._blend_scores(rrf_results, rerank_scores)
        
        self.assertEqual(len(blended), 2)
        self.assertIsInstance(blended[0], HybridResult)
        
        # Top result should have higher blended score
        self.assertGreater(blended[0].score, blended[1].score)
    
    def test_search_full_pipeline(self):
        """Test full search pipeline."""
        query = "full pipeline test"
        
        import numpy as np
        mock_vector = np.random.rand(2048).astype(np.float32)
        
        with patch('src.search.search_fts', return_value=self.mock_fts_results), \
             patch('src.search.search_vec', return_value=self.mock_vec_results), \
             patch('src.search.embed_text', return_value=mock_vector), \
             patch('src.search.rerank') as mock_rerank, \
             patch('src.search.expand_query', return_value=[]):
            
            mock_rerank.return_value = [
                MagicMock(document={'filepath': 'qmd://test/file1.txt'}, score=0.8),
                MagicMock(document={'filepath': 'qmd://test/file2.txt'}, score=0.7),
            ]
            
            results = self.searcher.search(query)
            
            # Should return HybridResults
            self.assertGreater(len(results), 0)
            self.assertIsInstance(results[0], HybridResult)
            self.assertIn('rrf_rank', dir(results[0]))
            self.assertIn('rerank_score', dir(results[0]))


class TestHybridQuery(unittest.TestCase):
    """Tests for hybrid_query convenience function."""
    
    def test_hybrid_query_returns_results(self):
        """Test hybrid_query returns search results."""
        with patch('src.search.HybridSearcher') as MockSearcher:
            mock_instance = MagicMock()
            mock_instance.search.return_value = [
                HybridResult(
                    filepath="qmd://test/file.txt",
                    display_path="test/file.txt",
                    title="Doc",
                    context=None,
                    hash="hash",
                    docid="hash",
                    collection="test",
                    modified_at="",
                    body_length=100,
                    body="content",
                    score=0.9,
                    rrf_rank=1,
                    rerank_score=0.85,
                    source="fts",
                )
            ]
            MockSearcher.return_value = mock_instance
            
            from src.search import hybrid_query
            results = hybrid_query("test query", limit=5)
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].score, 0.9)


class TestCacheClearing(unittest.TestCase):
    """Tests for cache clearing functions."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Reset the database singleton
        src_db.close_database()
        src_db.initialize_database(self.temp_dir)
    
    def tearDown(self):
        """Clean up."""
        src_db.close_database()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_clear_expand_and_rerank_caches(self):
        """Test clearing both caches."""
        # Add some cache entries
        src_db.cache_table.add([
            {"key": "expand:test1", "value": '{"type": "lex"}', "created_at": 1234567890},
            {"key": "rerank:test2", "value": "0.5", "created_at": 1234567890},
        ])
        
        clear_expand_and_rerank_caches()
        
        # Verify caches are empty
        rows = list(src_db.cache_table.search().limit(100).to_list())
        expand_rows = [r for r in rows if r.get('key', '').startswith('expand:')]
        rerank_rows = [r for r in rows if r.get('key', '').startswith('rerank:')]
        self.assertEqual(len(expand_rows), 0)
        self.assertEqual(len(rerank_rows), 0)


if __name__ == '__main__':
    unittest.main()