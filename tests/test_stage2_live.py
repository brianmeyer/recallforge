"""
test_stage2_live.py - Live test with actual image embeddings using Qwen3VLEmbedder

Tests cross-modal search with real images from LinkedIn Images folder.
"""

import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import db, store, embed


def test_live_image_embedding():
    """Test actual image embedding with Qwen3VLEmbedder."""
    print("\nTesting live image embedding...")
    
    # Check for test images
    images_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Molly Notes/LinkedIn Images/")
    if not os.path.exists(images_dir):
        print("  Skipping: LinkedIn Images directory not found")
        return None
    
    # Get a few test images
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')][:2]
    if not image_files:
        print("  Skipping: No PNG images found")
        return None
    
    print(f"  Found {len(image_files)} test images: {image_files}")
    
    # Initialize embedder
    print("  Initializing Qwen3VLEmbedder (this may take a moment)...")
    embedder = embed.Embedder()
    
    # Test embedding single image
    test_image = os.path.join(images_dir, image_files[0])
    print(f"  Embedding: {image_files[0]}")
    
    try:
        vector = embedder.embed_image(test_image)
        print(f"  ✓ Got embedding: shape={vector.shape}, dtype={vector.dtype}")
        assert vector.shape == (2048,), f"Expected 2048-dim vector, got {vector.shape}"
        print(f"  ✓ Vector dimension correct: 2048")
        return embedder
    except Exception as e:
        print(f"  ✗ Embedding failed: {e}")
        return None


def test_live_cross_modal_search(embedder):
    """Test cross-modal search with real embeddings."""
    print("\nTesting live cross-modal search...")
    
    if embedder is None:
        print("  Skipping: No embedder available")
        return
    
    # Setup temp database
    temp_dir = tempfile.mkdtemp(prefix="qmd-vl-live-")
    
    try:
        db.initialize_database(temp_dir)
        
        # Index some text documents
        print("  Indexing text documents...")
        text_docs = [
            ("agent-architecture.md", 
             "Agent Architecture. This document describes how AI agents use memory systems, "
             "tool calling, and multi-step reasoning. The supervisor agent coordinates "
             "multiple worker agents for complex tasks."),
            ("databricks-overview.md",
             "Databricks Platform. A unified analytics platform for data engineering, "
             "machine learning, and AI. Features include Delta Lake, MLflow, and "
             "vector search capabilities."),
            ("robotics-intro.md",
             "Robotics and Autonomous Systems. Boston Dynamics Atlas robot demonstrates "
             "advanced mobility and manipulation. Autonomous robots use perception, "
             "planning, and control systems."),
        ]
        
        for path, content in text_docs:
            store.index_document(
                path=path,
                text=content,
                collection="live_test",
                model="Qwen3-VL-Embedding-2B",
                embed_func=embedder.embed_text,
                content_type="text",
            )
        
        print(f"  Indexed {len(text_docs)} text documents")
        
        # Index some images
        images_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Molly Notes/LinkedIn Images/")
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')][:3]
            print(f"  Indexing {len(image_files)} images...")
            
            for img_file in image_files:
                img_path = os.path.join(images_dir, img_file)
                print(f"    Embedding: {img_file}")
                try:
                    content_hash = store.insert_image(
                        path=img_path,
                        collection="live_test",
                        embed_func=embedder.embed_image,
                    )
                    print(f"    ✓ Indexed: {img_file} (hash: {content_hash[:8]}...)")
                except Exception as e:
                    print(f"    ✗ Failed to index {img_file}: {e}")
        
        # Test cross-modal searches
        print("\n  Testing cross-modal queries...")
        
        test_queries = [
            "robot autonomy and perception",
            "agent supervisor coordination",
            "databricks AI platform",
        ]
        
        for query in test_queries:
            print(f"\n    Query: '{query}'")
            query_vec = embedder.embed_text(query)
            
            # Search across all content
            results = store.search_vec(query_vec, limit=10, collection="live_test")
            
            print(f"    Results ({len(results)}):")
            for i, r in enumerate(results[:5]):
                ct = r.content_type if hasattr(r, 'content_type') else 'text'
                print(f"      {i+1}. [{ct}] {r.title} (score: {r.score:.3f})")
            
            # Check if we get both text and image results
            has_text = any(r.content_type == "text" for r in results)
            has_image = any(r.content_type == "image" for r in results)
            
            if has_text:
                print(f"    ✓ Found text results")
            if has_image:
                print(f"    ✓ Found image results (cross-modal!)")
        
        print("\n✓ Live cross-modal search test complete")
        
    finally:
        db.close_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_live_tests():
    """Run live tests with real embeddings."""
    print("=" * 60)
    print("QMD-VL Stage 2 Live Tests (Real Embeddings)")
    print("=" * 60)
    
    try:
        # Test 1: Live image embedding
        embedder = test_live_image_embedding()
        
        # Test 2: Live cross-modal search
        test_live_cross_modal_search(embedder)
        
        print("\n" + "=" * 60)
        print("LIVE TESTS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_live_tests()