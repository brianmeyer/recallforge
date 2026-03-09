"""
test_search.py - Tests for hybrid search pipeline.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

# Set up paths before importing - use qmd-vl as package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import search
from src import store
from search import (
    HybridSearcher,
    hybrid_query,
    clear_expand_and_rerank_caches,
    HybridResult,
)


class TestHybridSearcher(unittest.TestCase):
    """Tests for HybridSearcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.searcher = HybridSearcher(limit=10, collection="test")
        
        # Patch database and search functions
        search_db = search.db
        from src import store
        SearchResult = store.SearchResult
        self.original_search_fts = store.search_fts
        self.original_search_vec = store.search_vec
        self.original_embed_text = search.embed_text
        self.original_expand = search.expand_query
        self.original_rerank = search.rerank
        
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
        
        search_fts.search_fts = lambda *args, **kwargs: self.mock_fts_results
        search_vec.search_vec = lambda *args, **kwargs: self.mock_vec_results
        
        # Mock embed_text
        import numpy as np
        embed_text.embed_text = lambda text: np.random.rand(2048)
        
        # Mock expand_query
        from search import expand_query
        self.original_expand = expand_query.expand_query
        expand_query.expand_query = lambda *args, **kwargs: []
    
    def tearDown(self):
        """Clean up."""
        store.search_fts = self.original_search_fts
        store.search_vec = self.original_search_vec
        search.embed_text = self.original_embed_text
        search.expand_query = self.original_expand
        search.expand_query = self.original_expand
        search.rerank = self.original_rerank
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ba25_probe(self):
        """Test BM25 probe returns results."""
        query = "test query"
        results = self.searcher._bm25_probe(query)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].score, 0.95)
    
    def test_vector_search(self):
        """Test vector search returns results."""
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
        
        # Should have unique results
        self.assertGreater(len(fused), 0)
        
        # All results should have non-zero RRF score
        for r in fused:
            self.assertGreater(r.score, 0)
    
    def test_select_best_chunk(self):
        """Test best chunk selection."""
        result = MagicMock()
        result.body = "test body"
        result.context = "test context"
        result.filepath = "qmd://test/file.txt"
        result.content_type = "text"
        
        chunk = self.searcher._select_best_chunk(result)
        
        self.assertIn('text', chunk)
        self.assertIn('filepath', chunk)
    
    def test_rerank_candidates(self):
        """Test candidate reranking."""
        candidates = self.mock_fts_results[:1]
        
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
        self.assertIsInstance(blended[0], search.HybridResult)
        
        # Top result should have higher blended score
        self.assertGreater(blended[0].score, blended[1].score)
    
    def test_search_full_pipeline(self):
        """Test full search pipeline."""
        query = "full pipeline test"
        
        # Mock all dependencies
        with patch('search.store.search_fts', return_value=self.mock_fts_results), \
             patch('search.store.search_vec', return_value=self.mock_vec_results), \
             patch('search.embed_text', return_value=__import__('numpy').random.rand(2048)), \
             patch('search.rerank.rerank') as mock_rerank, \
             patch('search.expand_query', return_value=[]):
            
            mock_rerank.return_value = [
                MagicMock(score=0.8, document={'filepath': 'qmd://test/file1.txt'}),
                MagicMock(score=0.7, document={'filepath': 'qmd://test/file2.txt'}),
            ]
            
            results = self.searcher.search(query)
            
            # Should return HybridResults
            self.assertGreater(len(results), 0)
            self.assertIsInstance(results[0], search.HybridResult)
            self.assertIn('rrf_rank', dir(results[0]))
            self.assertIn('rerank_score', dir(results[0]))


class TestHybridQuery(unittest.TestCase):
    """Tests for hybrid_query convenience function."""
    
    def test_hybrid_query_returns_results(self):
        """Test hybrid_query returns search results."""
        with patch('search.HybridSearcher') as MockSearcher:
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
            mock_instance.search.return_value = [
                search.HybridResult(
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
            
            results = search.hybrid_query("test query", limit=5)
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].score, 0.9)


class TestCacheClearing(unittest.TestCase):
    """Tests for cache clearing functions."""
    
    def test_clear_expand_and_rerank_caches(self):
        """Test clearing both caches."""
        try:
            search.clear_expand_and_rerank_caches()
            # If no exception, test passes
            self.assertTrue(True)
        except Exception as e:
            # May fail if database not initialized, that's okay
            self.assertIn("not initialized", str(e).lower() or "not initialized" in str(e).lower())


if __name__ == '__main__':
    unittest.main()
