#!/usr/bin/env python3
"""
COCO 1K Retrieval Benchmark for RecallForge

Measures Recall@1/5/10 on:
- Text -> Image retrieval
- Image -> Text retrieval

Uses COCO 2017 validation set (first 1K images)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# Add recallforge to path
sys.path.insert(0, "/Users/brianmeyer/recallforge/src")

from recallforge import get_backend


# Configuration
COCO_ROOT = Path("/tmp/recallforge-bench/coco")
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
RESULTS_DIR = Path("/Users/brianmeyer/recallforge/benchmarks/results")
NUM_IMAGES = 1000
BATCH_SIZE = 4  # Image batch size for embedding (reduced for memory)
TEXT_BATCH_SIZE = 32  # Text batch size for embedding (reduced for memory)


@dataclass
class ImageSample:
    image_id: int
    image_path: Path
    captions: List[str]


def download_file(url: str, dest: Path, desc: str) -> None:
    """Download a file with progress."""
    if dest.exists():
        print(f"  {desc} already exists, skipping download")
        return
    
    print(f"  Downloading {desc}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        print(f"\r    Progress: {percent}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="")
    
    urllib.request.urlretrieve(url, dest, reporthook=report_progress)
    print()  # Newline after progress


def download_coco() -> Tuple[Path, Path]:
    """Download COCO validation images and annotations."""
    print("Downloading COCO 2017 validation dataset...")
    
    # Download annotations
    ann_zip = COCO_ROOT / "annotations_trainval2017.zip"
    download_file(ANNOTATIONS_URL, ann_zip, "annotations")
    
    # Download images
    img_zip = COCO_ROOT / "val2017.zip"
    download_file(IMAGES_URL, img_zip, "validation images")
    
    return ann_zip, img_zip


def extract_coco(ann_zip: Path, img_zip: Path) -> Tuple[Path, Path]:
    """Extract COCO dataset."""
    print("Extracting COCO dataset...")
    
    # Extract annotations
    ann_dir = COCO_ROOT / "annotations"
    if not ann_dir.exists():
        print("  Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as z:
            z.extractall(COCO_ROOT)
    
    # Extract images
    img_dir = COCO_ROOT / "val2017"
    if not img_dir.exists():
        print("  Extracting images...")
        with zipfile.ZipFile(img_zip, 'r') as z:
            z.extractall(COCO_ROOT)
    
    captions_file = ann_dir / "captions_val2017.json"
    return captions_file, img_dir


def load_coco_data(captions_file: Path, img_dir: Path) -> List[ImageSample]:
    """Load COCO captions and create image samples."""
    print(f"Loading COCO captions from {captions_file}...")
    
    with open(captions_file) as f:
        data = json.load(f)
    
    # Build image_id -> file_name mapping
    images = {img["id"]: img["file_name"] for img in data["images"]}
    
    # Group captions by image_id
    captions_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in captions_by_image:
            captions_by_image[img_id] = []
        captions_by_image[img_id].append(ann["caption"])
    
    # Create ImageSample objects (first 1K images)
    samples = []
    for img_id in sorted(images.keys())[:NUM_IMAGES]:
        file_name = images[img_id]
        img_path = img_dir / file_name
        captions = captions_by_image.get(img_id, [])
        if len(captions) >= 5 and img_path.exists():
            samples.append(ImageSample(
                image_id=img_id,
                image_path=img_path,
                captions=captions[:5]  # Use first 5 captions
            ))
    
    print(f"  Loaded {len(samples)} images with 5 captions each")
    return samples


def compute_recall_at_k(similarities: np.ndarray, ground_truth_indices: np.ndarray, k: int) -> float:
    """
    Compute Recall@K.
    
    Args:
        similarities: (num_queries, num_candidates) similarity matrix
        ground_truth_indices: (num_queries,) ground truth indices for each query
        k: Top-K to consider
    
    Returns:
        Recall@K as a float between 0 and 1
    """
    num_queries = similarities.shape[0]
    recalls = 0
    
    for i in range(num_queries):
        # Get top-k indices by similarity (descending order)
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        # Check if ground truth is in top-k
        if ground_truth_indices[i] in top_k_indices:
            recalls += 1
    
    return recalls / num_queries


def embed_all_captions(samples: List[ImageSample], backend) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Embed all captions once and return captions, caption->image map, embeddings."""
    all_captions = []
    caption_to_image = []
    for img_idx, sample in enumerate(samples):
        for caption in sample.captions:
            all_captions.append(caption)
            caption_to_image.append(img_idx)

    print(f"\nEmbedding {len(all_captions)} captions...")
    caption_embeddings = []
    total_batches = (len(all_captions) + TEXT_BATCH_SIZE - 1) // TEXT_BATCH_SIZE
    for i in range(0, len(all_captions), TEXT_BATCH_SIZE):
        batch = all_captions[i:i + TEXT_BATCH_SIZE]
        batch_num = i // TEXT_BATCH_SIZE + 1
        print(f"  Text batch {batch_num}/{total_batches}: captions {i+1}-{i+len(batch)}")
        batch_embeddings = backend.embed_texts(batch)
        caption_embeddings.append(batch_embeddings)
        gc.collect()

    return all_captions, np.array(caption_to_image), np.vstack(caption_embeddings)


def run_text_to_image_benchmark(
    samples: List[ImageSample],
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Run Text->Image retrieval benchmark.
    
    For each caption, find the correct image among all images.
    """
    print("\nRunning Text->Image retrieval benchmark...")
    
    # Ground truth image index for each caption query
    ground_truth_indices = []
    for img_idx, sample in enumerate(samples):
        for _caption in sample.captions:
            ground_truth_indices.append(img_idx)

    print(f"  Computing similarities...")

    # Compute cosine similarities (embeddings are already L2 normalized)
    similarities = caption_embeddings @ image_embeddings.T

    ground_truth_indices = np.array(ground_truth_indices)
    
    # Compute Recall@1, @5, @10
    r1 = compute_recall_at_k(similarities, ground_truth_indices, 1)
    r5 = compute_recall_at_k(similarities, ground_truth_indices, 5)
    r10 = compute_recall_at_k(similarities, ground_truth_indices, 10)
    
    return r1, r5, r10


def run_image_to_text_benchmark(
    samples: List[ImageSample],
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    caption_to_image: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Run Image->Text retrieval benchmark.
    
    For each image, find any of its 5 captions among all captions.
    """
    print("\nRunning Image->Text retrieval benchmark...")
    
    print(f"  Computing similarities...")
    
    # Compute cosine similarities (embeddings are already L2 normalized)
    similarities = image_embeddings @ caption_embeddings.T
    
    # For each image, check if any of its 5 captions is in top-k
    num_images = len(samples)
    recalls_1 = 0
    recalls_5 = 0
    recalls_10 = 0
    
    for img_idx in range(num_images):
        # Get indices of captions belonging to this image
        caption_indices = np.where(caption_to_image == img_idx)[0]
        
        # Get top-k caption indices by similarity
        top_1 = np.argsort(similarities[img_idx])[::-1][:1]
        top_5 = np.argsort(similarities[img_idx])[::-1][:5]
        top_10 = np.argsort(similarities[img_idx])[::-1][:10]
        
        # Check if any correct caption is in top-k
        if np.any(np.isin(top_1, caption_indices)):
            recalls_1 += 1
        if np.any(np.isin(top_5, caption_indices)):
            recalls_5 += 1
        if np.any(np.isin(top_10, caption_indices)):
            recalls_10 += 1
    
    r1 = recalls_1 / num_images
    r5 = recalls_5 / num_images
    r10 = recalls_10 / num_images
    
    return r1, r5, r10


def main():
    """Main benchmark function."""
    print("=" * 60)
    print("COCO 1K Retrieval Benchmark for RecallForge")
    print("=" * 60)
    
    start_time = time.time()
    
    # Download and extract COCO
    ann_zip, img_zip = download_coco()
    captions_file, img_dir = extract_coco(ann_zip, img_zip)
    
    # Load data
    samples = load_coco_data(captions_file, img_dir)
    
    if len(samples) < NUM_IMAGES:
        print(f"Warning: Only {len(samples)} images available, expected {NUM_IMAGES}")
    
    # Initialize backend
    print("\nInitializing RecallForge backend...")
    os.environ["RECALLFORGE_BACKEND"] = "mlx"
    os.environ["RECALLFORGE_MODE"] = "embed"
    os.environ["RECALLFORGE_MLX_QUANTIZE"] = "4bit"
    
    backend = get_backend()
    print("Backend created. Skipping explicit warm_up to avoid long compile stall.")
    
    # Get model info
    model_name = backend.EMBEDDER_MODEL if hasattr(backend, "EMBEDDER_MODEL") else "Qwen3-VL-Embedding-2B"
    
    # Embed all images
    print(f"\nEmbedding {len(samples)} images...")
    image_embeddings = []
    total_image_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        paths = [str(s.image_path) for s in batch]
        print(f"  Image batch {i//BATCH_SIZE + 1}/{total_image_batches}: images {i+1}-{min(i+len(batch), len(samples))}")
        batch_embeddings = backend.embed_images(paths)
        image_embeddings.append(batch_embeddings)
        gc.collect()
    
    image_embeddings = np.vstack(image_embeddings)
    print(f"  Image embeddings shape: {image_embeddings.shape}")

    # Embed captions once for both retrieval directions
    _all_captions, caption_to_image, caption_embeddings = embed_all_captions(samples, backend)
    print(f"  Caption embeddings shape: {caption_embeddings.shape}")
    
    # Run benchmarks
    t2i_r1, t2i_r5, t2i_r10 = run_text_to_image_benchmark(samples, image_embeddings, caption_embeddings)
    i2t_r1, i2t_r5, i2t_r10 = run_image_to_text_benchmark(samples, image_embeddings, caption_embeddings, caption_to_image)
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Quantization: 4-bit")
    print(f"Backend: MLX")
    print(f"Number of images: {len(samples)}")
    print(f"Total runtime: {total_time:.1f}s")
    print()
    print("Text -> Image Retrieval:")
    print(f"  R@1:  {t2i_r1*100:.2f}%")
    print(f"  R@5:  {t2i_r5*100:.2f}%")
    print(f"  R@10: {t2i_r10*100:.2f}%")
    print()
    print("Image -> Text Retrieval:")
    print(f"  R@1:  {i2t_r1*100:.2f}%")
    print(f"  R@5:  {i2t_r5*100:.2f}%")
    print(f"  R@10: {i2t_r10*100:.2f}%")
    print("=" * 60)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": model_name,
        "quantization": "4bit",
        "backend": "MLX",
        "num_images": len(samples),
        "total_runtime_seconds": total_time,
        "text_to_image": {
            "R@1": round(t2i_r1, 4),
            "R@5": round(t2i_r5, 4),
            "R@10": round(t2i_r10, 4)
        },
        "image_to_text": {
            "R@1": round(i2t_r1, 4),
            "R@5": round(i2t_r5, 4),
            "R@10": round(i2t_r10, 4)
        }
    }
    
    results_file = RESULTS_DIR / "coco_1k_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
