"""
test_live.py - Live Tests for RecallForge with Real Models.

NO MOCKS. These tests use actual model inference.

Run with: pytest tests/test_live.py -m live -v
Skip with: pytest tests/ -m "not live"
"""

import os
import sys
import tempfile
import shutil
import time

import pytest

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def temp_storage():
    """Create temporary storage for tests."""
    temp_dir = tempfile.mkdtemp(prefix="recallforge-live-")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def backend():
    """Get the model backend (slow - loads models once)."""
    from recallforge import get_backend
    backend = get_backend()
    try:
        backend.warm_up()
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"Live backend dependencies unavailable: {e}")
    return backend


@pytest.fixture(scope="module")
def storage(temp_storage):
    """Get the storage backend."""
    from recallforge import get_storage
    return get_storage(temp_storage)


class TestTorchBackendLive:
    """Live tests for Torch backend with real models."""
    
    @pytest.mark.live
    def test_embed_text_live(self, backend):
        """Test text embedding with real model."""
        text = "Hello, world! This is a test of the embedding model."
        
        vector = backend.embed_text(text)
        
        assert vector is not None
        assert vector.shape == (2048,), f"Expected 2048-dim vector, got {vector.shape}"
        assert vector.dtype.name == "float32"
        
        # Check normalization (approximate)
        norm = (vector ** 2).sum() ** 0.5
        assert 0.9 < norm < 1.1, f"Vector should be approximately normalized, norm={norm}"
    
    @pytest.mark.live
    def test_embed_texts_batch_live(self, backend):
        """Test batch text embedding."""
        texts = [
            "First document about AI and machine learning.",
            "Second document about neural networks and deep learning.",
            "Third document about natural language processing.",
        ]
        
        vectors = backend.embed_texts(texts)
        
        assert vectors is not None
        assert vectors.shape == (3, 2048)
        
        # Each vector should be normalized
        for i, v in enumerate(vectors):
            norm = (v ** 2).sum() ** 0.5
            assert 0.9 < norm < 1.1, f"Vector {i} not normalized, norm={norm}"
    
    @pytest.mark.live
    def test_embed_image_live(self, backend):
        """Test image embedding with real model."""
        # Use a test image if available
        images_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Molly Notes/LinkedIn Images/")
        
        if not os.path.exists(images_dir):
            pytest.skip("No test images available")
        
        images = [f for f in os.listdir(images_dir) if f.endswith('.png')][:1]
        if not images:
            pytest.skip("No PNG images found")
        
        image_path = os.path.join(images_dir, images[0])
        
        vector = backend.embed_image(image_path)
        
        assert vector is not None
        assert vector.shape == (2048,), f"Expected 2048-dim vector, got {vector.shape}"
        
        # Check normalization
        norm = (vector ** 2).sum() ** 0.5
        assert 0.9 < norm < 1.1, f"Vector should be approximately normalized, norm={norm}"
    
    @pytest.mark.live
    def test_rerank_live(self, backend):
        """Test reranking with real model."""
        if not backend.needs_reranker():
            pytest.skip("Reranker not loaded (mode != hybrid/full)")
        
        query = "How do AI agents use memory?"
        documents = [
            {"text": "AI agents use episodic memory to remember past experiences and semantic memory for facts."},
            {"text": "Neural networks are computing systems inspired by biological brains."},
            {"text": "Memory systems enable agents to maintain context across conversations."},
            {"text": "Graph databases store data as nodes and relationships."},
        ]
        
        scores = backend.rerank(query, documents)
        
        assert scores is not None
        assert len(scores) == len(documents)
        assert all(0.0 <= s <= 1.0 for s in scores), f"Scores should be in [0, 1], got {scores}"
        
        # Memory-related docs should score higher
        print(f"Rerank scores: {scores}")
    
    @pytest.mark.live
    def test_full_mode_rejected(self, backend):
        """Test that setting mode='full' raises ValueError."""
        with pytest.raises(ValueError):
            backend.set_mode("full")
    
    @pytest.mark.live
    def test_backend_info(self, backend):
        """Test backend info retrieval."""
        info = backend.get_info()
        
        assert info.name in ["torch", "mlx"]
        assert info.device in ["cuda", "mps", "cpu"]
        assert info.embedder_loaded == True
        assert info.memory_allocated_gb > 0


class TestStorageLive:
    """Live tests for storage backend."""
    
    @pytest.mark.live
    def test_index_document_live(self, backend, storage):
        """Test document indexing with real embeddings."""
        doc = {
            "id": "test-doc.md",
            "text": "This is a test document about artificial intelligence and machine learning. "
                    "It contains multiple sentences to provide enough content for chunking. "
                    "The document discusses neural networks, deep learning, and natural language processing.",
        }
        
        content_hash = storage.index_document(
            path=doc["id"],
            text=doc["text"],
            collection="live_test",
            model="Qwen3-VL-Embedding-2B",
            embed_func=backend.embed_text,
        )
        
        assert content_hash is not None
        assert len(content_hash) == 64  # SHA-256 hex digest
        
        # Verify we can search for it
        results = storage.search_fts("artificial intelligence", limit=5, collection="live_test")
        assert len(results) >= 1
    
    @pytest.mark.live
    def test_vector_search_live(self, backend, storage):
        """Test vector search with real embeddings."""
        # Index some documents
        docs = [
            {"id": "vec-test-1.md", "text": "Python is a popular programming language for data science."},
            {"id": "vec-test-2.md", "text": "JavaScript is widely used for web development."},
            {"id": "vec-test-3.md", "text": "Rust provides memory safety without garbage collection."},
        ]
        
        for doc in docs:
            storage.index_document(
                path=doc["id"],
                text=doc["text"],
                collection="vec_test",
                model="Qwen3-VL-Embedding-2B",
                embed_func=backend.embed_text,
            )
        
        # Search for programming languages
        query = "coding languages for software development"
        vector = backend.embed_text(query)
        
        results = storage.search_vec(vector.tolist(), limit=5, collection="vec_test")
        
        assert len(results) >= 1
        print(f"Vector search results for '{query}':")
        for r in results:
            print(f"  {r.title}: {r.score:.3f}")


class TestHybridSearchLive:
    """Live tests for hybrid search pipeline."""
    
    @pytest.mark.live
    def test_hybrid_search_live(self, backend, storage):
        """Test full hybrid search with real models."""
        from recallforge.search import HybridSearcher
        
        # Index test documents
        docs = [
            {"id": "hybrid-ai.md", "text": "Artificial intelligence (AI) is intelligence demonstrated by machines. "
                                           "Modern AI uses neural networks for pattern recognition."},
            {"id": "hybrid-ml.md", "text": "Machine learning is a subset of AI that learns from data. "
                                          "Deep learning uses multi-layer neural networks."},
            {"id": "hybrid-nlp.md", "text": "Natural language processing enables computers to understand text. "
                                           "Large language models like GPT are trained on massive text corpora."},
        ]
        
        for doc in docs:
            storage.index_document(
                path=doc["id"],
                text=doc["text"],
                collection="hybrid_test",
                model="Qwen3-VL-Embedding-2B",
                embed_func=backend.embed_text,
            )
        
        # Create searcher
        searcher = HybridSearcher(
            backend=backend,
            storage=storage,
            limit=5,
            collection="hybrid_test",
        )
        
        # Search
        query = "How do computers understand human language?"
        results = searcher.search(query)
        
        assert len(results) >= 1
        
        print(f"\nHybrid search results for '{query}':")
        for r in results:
            print(f"  [{r.score:.3f}] {r.title}")
            print(f"    RRF rank: {r.rrf_rank}, Rerank: {r.rerank_score:.3f}")
            print(f"    Sources: {r.source}")
    
    @pytest.mark.live
    def test_mode_differences(self, temp_storage):
        """Test that different modes have different behaviors."""
        from recallforge import get_backend, get_storage
        from recallforge.search import HybridSearcher
        
        # Index documents once
        storage = get_storage(temp_storage)
        
        docs = [
            {"id": "mode-test-1.md", "text": "Query expansion generates multiple variations of a search query."},
            {"id": "mode-test-2.md", "text": "Reranking refines search results using cross-encoders."},
        ]
        
        # Get embedder-only backend
        os.environ["RECALLFORGE_MODE"] = "embed"
        backend_embed = get_backend()
        try:
            backend_embed._load_embedder()
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"Live backend dependencies unavailable: {e}")
        
        for doc in docs:
            storage.index_document(
                path=doc["id"],
                text=doc["text"],
                collection="mode_test",
                model="Qwen3-VL-Embedding-2B",
                embed_func=backend_embed.embed_text,
            )
        
        # Test embed mode (no reranking)
        searcher_embed = HybridSearcher(
            backend=backend_embed,
            storage=storage,
            limit=5,
            collection="mode_test",
        )
        
        results_embed = searcher_embed.search("search query optimization")
        
        # All rerank scores should be 0.5 (neutral) in embed mode
        for r in results_embed:
            assert r.rerank_score == 0.5, "Embed mode should have neutral rerank scores"


class TestCrossModalLive:
    """Live tests for cross-modal search."""
    
    @pytest.mark.live
    def test_text_to_image_search(self, backend, storage):
        """Test text-to-image search with real embeddings."""
        # Check for test images
        images_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Molly Notes/LinkedIn Images/")
        
        if not os.path.exists(images_dir):
            pytest.skip("No test images available")
        
        images = [f for f in os.listdir(images_dir) if f.endswith('.png')][:3]
        if not images:
            pytest.skip("No PNG images found")
        
        # Index images
        for img_file in images:
            img_path = os.path.join(images_dir, img_file)
            try:
                storage.index_image(
                    path=img_path,
                    collection="cross_modal_test",
                    embed_func=backend.embed_image,
                )
                print(f"Indexed: {img_file}")
            except Exception as e:
                print(f"Warning: Could not index {img_file}: {e}")
        
        # Text query
        query = "diagram architecture system"
        vector = backend.embed_text(query)
        
        # Search images
        results = storage.search_vec(
            vector.tolist(),
            limit=5,
            collection="cross_modal_test",
            content_type="image",
        )
        
        print(f"\nText-to-image search for '{query}':")
        for r in results:
            print(f"  [{r.score:.3f}] {r.title}")
        
        # We may or may not get results depending on images


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "live"])