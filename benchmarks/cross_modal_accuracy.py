#!/usr/bin/env python3
"""
Cross-modal retrieval benchmark for RecallForge.

Measures Recall@1/5/10 on:
- Text -> Image retrieval
- Image -> Text retrieval

Datasets:
- Flickr30k test split (preferred)
- COCO captions validation split (fallback)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "recallforge" / "benchmarks"
CACHE_ROOT = DEFAULT_CACHE_ROOT
HF_CACHE_DIR = CACHE_ROOT / "hf"
IMAGE_CACHE_DIR = CACHE_ROOT / "images"
EMBED_CACHE_DIR = CACHE_ROOT / "embeddings"
RESULT_JSON_PATH = Path("benchmarks/cross_modal_results.json")
RESULT_MD_PATH = Path("benchmarks/CROSS_MODAL.md")


@dataclass
class ImageSample:
    image_id: str
    image_path: str
    captions: List[str]


@dataclass
class DatasetBundle:
    dataset_name: str
    dataset_source: str
    split: str
    images: List[ImageSample]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-modal Recall@K benchmark for RecallForge."
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "mlx", "auto"],
        default="auto",
        help="Backend to evaluate (default: auto)",
    )
    parser.add_argument(
        "--mode",
        choices=["embed", "hybrid", "full"],
        default="embed",
        help="Search mode to evaluate (default: embed)",
    )
    parser.add_argument(
        "--dataset",
        choices=["flickr30k", "coco"],
        default="flickr30k",
        help="Dataset to benchmark (default: flickr30k)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of images to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=8,
        help="Image embedding batch size (default: 8)",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=64,
        help="Text embedding batch size (default: 64)",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=50,
        help="Top-K candidates passed to reranker for hybrid/full (default: 50)",
    )
    return parser.parse_args()


def _set_cache_root(root: Path) -> None:
    global CACHE_ROOT, HF_CACHE_DIR, IMAGE_CACHE_DIR, EMBED_CACHE_DIR
    CACHE_ROOT = root
    HF_CACHE_DIR = CACHE_ROOT / "hf"
    IMAGE_CACHE_DIR = CACHE_ROOT / "images"
    EMBED_CACHE_DIR = CACHE_ROOT / "embeddings"


def ensure_paths() -> None:
    override = os.environ.get("RECALLFORGE_BENCH_CACHE")
    if override:
        _set_cache_root(Path(override).expanduser())

    targets = (CACHE_ROOT, HF_CACHE_DIR, IMAGE_CACHE_DIR, EMBED_CACHE_DIR, RESULT_JSON_PATH.parent)
    try:
        for p in targets:
            p.mkdir(parents=True, exist_ok=True)
        return
    except PermissionError:
        fallback = Path(tempfile.gettempdir()) / "recallforge" / "benchmarks"
        print(f"[warn] Cache path {CACHE_ROOT} is not writable; using {fallback} instead.")
        _set_cache_root(fallback)

    for p in (CACHE_ROOT, HF_CACHE_DIR, IMAGE_CACHE_DIR, EMBED_CACHE_DIR, RESULT_JSON_PATH.parent):
        p.mkdir(parents=True, exist_ok=True)


def _safe_text(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_caption_strings(value: Any) -> List[str]:
    captions: List[str] = []
    if value is None:
        return captions
    if isinstance(value, str):
        text = _safe_text(value)
        if text:
            captions.append(text)
        return captions
    if isinstance(value, (list, tuple)):
        for item in value:
            captions.extend(_extract_caption_strings(item))
        return captions
    if isinstance(value, dict):
        for key in ("caption", "captions", "text", "raw", "sentence", "sentences"):
            if key in value:
                captions.extend(_extract_caption_strings(value[key]))
        if captions:
            return captions
        for nested in value.values():
            if isinstance(nested, (str, list, tuple, dict)):
                captions.extend(_extract_caption_strings(nested))
        return captions
    return captions


def extract_captions_from_row(row: Dict[str, Any]) -> List[str]:
    keys = (
        "captions",
        "caption",
        "sentences",
        "annotations",
        "texts",
        "description",
    )
    found: List[str] = []
    for key in keys:
        if key in row:
            found.extend(_extract_caption_strings(row[key]))
    return _dedupe_keep_order([c for c in found if c])


def extract_image_payload(row: Dict[str, Any]) -> Any:
    keys = (
        "image",
        "img",
        "jpg",
        "image_file",
        "image_path",
        "filepath",
        "file_path",
        "filename",
    )
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    # Fallback: download from URL (COCO datasets store coco_url/flickr_url)
    for url_key in ("coco_url", "flickr_url", "url", "image_url"):
        url = row.get(url_key)
        if url and isinstance(url, str) and url.startswith("http"):
            import urllib.request
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()  # returns bytes, handled by materialize_image
    raise ValueError("No image payload field found in row.")


def extract_image_id(row: Dict[str, Any], row_idx: int) -> str:
    keys = ("image_id", "img_id", "id", "filename", "file_name", "image_name")
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int)):
            return str(value)
        if isinstance(value, dict):
            maybe = value.get("id") or value.get("path")
            if maybe is not None:
                return str(maybe)
    return f"row_{row_idx:08d}"


def _image_filename_for_id(image_id: str, suffix: str = ".jpg") -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", image_id)[:80] or "img"
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{digest}{suffix}"


def _save_pil_image(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    rgb.save(out_path, format="JPEG", quality=95)


def materialize_image(payload: Any, image_id: str, image_dir: Path) -> str:
    if isinstance(payload, dict):
        if payload.get("path"):
            payload = payload["path"]
        elif payload.get("bytes") is not None:
            payload = payload["bytes"]

    if isinstance(payload, str):
        src = Path(payload).expanduser()
        if src.exists() and src.is_file():
            suffix = src.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
                suffix = ".jpg"
            out_path = image_dir / _image_filename_for_id(image_id, suffix=suffix)
            if out_path.exists():
                return str(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if suffix == src.suffix.lower():
                shutil.copy2(src, out_path)
                return str(out_path)
            with Image.open(src) as img:
                _save_pil_image(img, out_path)
            return str(out_path)

    if isinstance(payload, bytes):
        out_path = image_dir / _image_filename_for_id(image_id, suffix=".jpg")
        if out_path.exists():
            return str(out_path)
        with Image.open(io.BytesIO(payload)) as img:
            _save_pil_image(img, out_path)
        return str(out_path)

    if isinstance(payload, Image.Image):
        out_path = image_dir / _image_filename_for_id(image_id, suffix=".jpg")
        if out_path.exists():
            return str(out_path)
        _save_pil_image(payload, out_path)
        return str(out_path)

    maybe_path = getattr(payload, "path", None)
    if isinstance(maybe_path, str):
        return materialize_image(maybe_path, image_id=image_id, image_dir=image_dir)

    raise ValueError(f"Unsupported image payload type: {type(payload).__name__}")


def _candidate_specs(dataset_name: str) -> List[Tuple[str, Optional[str], str]]:
    if dataset_name == "flickr30k":
        return [
            ("nlphuia/flickr30k", None, "test"),
            ("nlphuia/flickr30k", None, "validation"),
            ("nlphuji/flickr30k", None, "test"),
            ("nlphuji/flickr30k", None, "validation"),
        ]
    return [
        ("phiyodr/coco2017", None, "validation"),
        ("phiyodr/coco2017", None, "val"),
        ("HuggingFaceM4/COCO", None, "validation"),
    ]


def _load_hf_split(dataset_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required for benchmarks. "
            "Install with: pip install -e '.[benchmark]'"
        ) from exc

    errors: List[str] = []
    for path, config, split in _candidate_specs(dataset_name):
        kwargs: Dict[str, Any] = {
            "path": path,
            "split": split,
            "cache_dir": str(HF_CACHE_DIR),
        }
        if config:
            kwargs["name"] = config
        try:
            dataset = load_dataset(**kwargs)
            return dataset, path, split
        except Exception as exc:  # pragma: no cover - depends on remote datasets
            errors.append(f"{path}:{split} -> {exc}")
    summary = "\n".join(f"  - {err}" for err in errors)
    raise RuntimeError(f"Could not load {dataset_name} from HuggingFace datasets:\n{summary}")


def build_bundle(
    dataset_name: str,
    limit: int,
) -> DatasetBundle:
    requested = dataset_name
    try:
        dataset, source, split = _load_hf_split(dataset_name)
        resolved_name = requested
    except Exception as exc:
        if requested != "flickr30k":
            raise
        print(f"[warn] Flickr30k unavailable ({exc}). Falling back to COCO.")
        dataset, source, split = _load_hf_split("coco")
        resolved_name = "coco"

    image_dir = IMAGE_CACHE_DIR / resolved_name
    image_dir.mkdir(parents=True, exist_ok=True)

    records: Dict[str, Dict[str, Any]] = {}
    kept_ids: set[str] = set()

    total_rows = len(dataset)
    for idx, row in enumerate(dataset):
        if idx % 2500 == 0 and idx > 0:
            print(f"[dataset] parsed {idx}/{total_rows} rows, kept {len(records)} images")
        if not isinstance(row, dict):
            continue

        image_id = extract_image_id(row, idx)
        if image_id not in kept_ids and len(kept_ids) >= limit:
            continue

        captions = extract_captions_from_row(row)
        if not captions and image_id not in records:
            continue

        if image_id not in records:
            try:
                payload = extract_image_payload(row)
                path = materialize_image(payload, image_id=image_id, image_dir=image_dir)
            except Exception:
                continue
            records[image_id] = {"image_path": path, "captions": []}
            kept_ids.add(image_id)

        records[image_id]["captions"].extend(captions)

    images: List[ImageSample] = []
    for image_id, entry in records.items():
        caps = _dedupe_keep_order([_safe_text(c) for c in entry["captions"] if _safe_text(c)])[:5]
        if not caps:
            continue
        images.append(
            ImageSample(
                image_id=image_id,
                image_path=entry["image_path"],
                captions=caps,
            )
        )

    images = images[:limit]
    if not images:
        raise RuntimeError(f"No usable samples were parsed for dataset={resolved_name} from {source}:{split}.")

    return DatasetBundle(
        dataset_name=resolved_name,
        dataset_source=source,
        split=split,
        images=images,
    )


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def batched_embed_images(
    backend: Any,
    image_paths: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    total = len(image_paths)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_paths = list(image_paths[start:end])
        print(f"[embed:image] {end}/{total}")
        try:
            batch_vecs = backend.embed_images(batch_paths)
        except Exception:
            batch_vecs = np.stack([backend.embed_image(path) for path in batch_paths], axis=0)
        vectors.append(np.asarray(batch_vecs, dtype=np.float32))
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 2048), dtype=np.float32)


def batched_embed_texts(
    backend: Any,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = list(texts[start:end])
        print(f"[embed:text] {end}/{total}")
        try:
            batch_vecs = backend.embed_texts(batch_texts)
        except Exception:
            batch_vecs = np.stack([backend.embed_text(t) for t in batch_texts], axis=0)
        vectors.append(np.asarray(batch_vecs, dtype=np.float32))
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 2048), dtype=np.float32)


def flatten_caption_view(images: Sequence[ImageSample]) -> Tuple[List[str], np.ndarray]:
    captions: List[str] = []
    caption_to_image: List[int] = []
    for img_idx, sample in enumerate(images):
        for cap in sample.captions:
            captions.append(cap)
            caption_to_image.append(img_idx)
    return captions, np.asarray(caption_to_image, dtype=np.int32)


def manifest_hash(images: Sequence[ImageSample]) -> str:
    payload = {
        "image_ids": [s.image_id for s in images],
        "captions": [s.captions for s in images],
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def backend_cache_key(backend_info: Any) -> str:
    name = getattr(backend_info, "name", "unknown")
    quant = getattr(backend_info, "quantization", None)
    dtype = getattr(backend_info, "dtype", None)
    device = getattr(backend_info, "device", None)
    raw = f"{name}_{quant or dtype}_{device}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def load_or_build_embeddings(
    backend: Any,
    backend_info: Any,
    bundle: DatasetBundle,
    image_batch_size: int,
    text_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Path]:
    images = bundle.images
    captions, caption_to_image = flatten_caption_view(images)
    if not captions:
        raise RuntimeError("Parsed dataset has no captions.")

    mhash = manifest_hash(images)
    cache_name = f"{bundle.dataset_name}_{bundle.split}_{len(images)}_{mhash}_{backend_cache_key(backend_info)}.npz"
    cache_path = EMBED_CACHE_DIR / cache_name

    if cache_path.exists():
        try:
            data = np.load(cache_path)
            image_emb = np.asarray(data["image_embeddings"], dtype=np.float32)
            caption_emb = np.asarray(data["caption_embeddings"], dtype=np.float32)
            c2i = np.asarray(data["caption_to_image"], dtype=np.int32)
            if image_emb.shape[0] == len(images) and caption_emb.shape[0] == len(captions) and c2i.shape[0] == len(captions):
                print(f"[cache] hit: {cache_path}")
                return image_emb, caption_emb, c2i, True, cache_path
        except Exception:
            print(f"[cache] invalid cache file, rebuilding: {cache_path}")

    image_paths = [s.image_path for s in images]
    image_emb_raw = batched_embed_images(backend, image_paths, batch_size=image_batch_size)
    caption_emb_raw = batched_embed_texts(backend, captions, batch_size=text_batch_size)
    image_emb = normalize_rows(image_emb_raw)
    caption_emb = normalize_rows(caption_emb_raw)

    np.savez_compressed(
        cache_path,
        image_embeddings=image_emb,
        caption_embeddings=caption_emb,
        caption_to_image=caption_to_image,
    )
    print(f"[cache] wrote: {cache_path}")
    return image_emb, caption_emb, caption_to_image, False, cache_path


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0 or k <= 0:
        return np.empty((0,), dtype=np.int32)
    kk = min(k, scores.shape[0])
    part = np.argpartition(-scores, kk - 1)[:kk]
    order = part[np.argsort(-scores[part])]
    return order.astype(np.int32)


def merge_unique(primary: Sequence[int], fallback: Sequence[int], limit: int) -> np.ndarray:
    merged: List[int] = []
    seen = set()
    for seq in (primary, fallback):
        for item in seq:
            idx = int(item)
            if idx in seen:
                continue
            seen.add(idx)
            merged.append(idx)
            if len(merged) >= limit:
                return np.asarray(merged, dtype=np.int32)
    return np.asarray(merged, dtype=np.int32)


def recall_from_rankings(
    rankings: Sequence[np.ndarray],
    positives: Sequence[Sequence[int]],
) -> Dict[str, float]:
    total = len(rankings)
    if total == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0}

    hits = {1: 0, 5: 0, 10: 0}
    for ranked, positive in zip(rankings, positives):
        pos = set(int(p) for p in positive)
        for k in (1, 5, 10):
            subset = set(int(x) for x in ranked[:k])
            if pos.intersection(subset):
                hits[k] += 1
    return {
        "recall@1": hits[1] / total,
        "recall@5": hits[5] / total,
        "recall@10": hits[10] / total,
    }


def query_variants(
    backend: Any,
    query: str,
    mode: str,
    expansion_cache: Dict[str, List[str]],
) -> List[str]:
    if mode != "full":
        return [query]
    if query in expansion_cache:
        return expansion_cache[query]
    try:
        expanded = backend.expand_query(query)
        variants = [query]
        for key in ("lex", "vec", "hyde"):
            value = _safe_text(expanded.get(key, ""))
            if value:
                variants.append(value)
        variants = _dedupe_keep_order(variants)
    except Exception:
        variants = [query]
    expansion_cache[query] = variants
    return variants


def evaluate_text_to_image(
    mode: str,
    backend: Any,
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    captions: Sequence[str],
    caption_to_image: np.ndarray,
    image_doc_texts: Sequence[str],
    rerank_top_k: int,
) -> Dict[str, float]:
    base_scores = caption_embeddings @ image_embeddings.T
    rankings: List[np.ndarray] = []
    positives: List[List[int]] = []
    expansion_cache: Dict[str, List[str]] = {}

    total_queries = len(captions)
    for idx, caption in enumerate(captions):
        if idx % 100 == 0:
            print(f"[t2i:{mode}] {idx}/{total_queries}")

        if mode == "embed":
            ranked = top_k_indices(base_scores[idx], 10)
        else:
            query_text = caption
            candidate_scores = base_scores[idx]
            variants = query_variants(backend, query_text, mode, expansion_cache)
            if len(variants) > 1:
                variant_emb = normalize_rows(np.asarray(backend.embed_texts(variants), dtype=np.float32))
                expanded_scores = variant_emb @ image_embeddings.T
                candidate_scores = np.max(expanded_scores, axis=0)
                query_text = " ".join(variants)

            candidate_idx = top_k_indices(candidate_scores, max(rerank_top_k, 10))
            docs = [{"text": image_doc_texts[int(i)]} for i in candidate_idx]
            rerank_scores = np.asarray(backend.rerank(query_text, docs), dtype=np.float32)
            if rerank_scores.shape[0] != candidate_idx.shape[0]:
                rerank_scores = np.full(candidate_idx.shape[0], 0.5, dtype=np.float32)
            rerank_order = np.argsort(-rerank_scores)
            reranked = candidate_idx[rerank_order]
            fallback = top_k_indices(candidate_scores, 10)
            ranked = merge_unique(reranked, fallback, limit=10)

        rankings.append(ranked)
        positives.append([int(caption_to_image[idx])])

    return recall_from_rankings(rankings, positives)


def _pseudo_query_from_caption_candidates(
    candidate_idx: np.ndarray,
    captions: Sequence[str],
    max_captions: int = 3,
) -> str:
    pieces: List[str] = []
    for idx in candidate_idx[:max_captions]:
        text = _safe_text(captions[int(idx)])
        if text:
            pieces.append(text)
    joined = " ".join(_dedupe_keep_order(pieces))
    return joined if joined else "Describe the image."


def build_image_positive_map(caption_to_image: np.ndarray, num_images: int) -> List[List[int]]:
    out: List[List[int]] = [[] for _ in range(num_images)]
    for cap_idx, img_idx in enumerate(caption_to_image.tolist()):
        out[img_idx].append(cap_idx)
    return out


def evaluate_image_to_text(
    mode: str,
    backend: Any,
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    captions: Sequence[str],
    image_to_caption_indices: Sequence[Sequence[int]],
    rerank_top_k: int,
) -> Dict[str, float]:
    base_scores = image_embeddings @ caption_embeddings.T
    rankings: List[np.ndarray] = []
    positives: List[List[int]] = []
    expansion_cache: Dict[str, List[str]] = {}

    total_queries = image_embeddings.shape[0]
    for img_idx in range(total_queries):
        if img_idx % 50 == 0:
            print(f"[i2t:{mode}] {img_idx}/{total_queries}")

        if mode == "embed":
            ranked = top_k_indices(base_scores[img_idx], 10)
        else:
            base_row = base_scores[img_idx]
            seed_idx = top_k_indices(base_row, max(10, min(rerank_top_k, 25)))
            pseudo_query = _pseudo_query_from_caption_candidates(seed_idx, captions)
            query_text = pseudo_query

            candidate_scores = base_row
            variants = query_variants(backend, query_text, mode, expansion_cache)
            if len(variants) > 1:
                variant_emb = normalize_rows(np.asarray(backend.embed_texts(variants), dtype=np.float32))
                expanded_scores = variant_emb @ caption_embeddings.T
                expanded_row = np.max(expanded_scores, axis=0)
                candidate_scores = (0.7 * base_row) + (0.3 * expanded_row)
                query_text = " ".join(variants)

            candidate_idx = top_k_indices(candidate_scores, max(rerank_top_k, 10))
            docs = [{"text": captions[int(i)]} for i in candidate_idx]
            rerank_scores = np.asarray(backend.rerank(query_text, docs), dtype=np.float32)
            if rerank_scores.shape[0] != candidate_idx.shape[0]:
                rerank_scores = np.full(candidate_idx.shape[0], 0.5, dtype=np.float32)
            rerank_order = np.argsort(-rerank_scores)
            reranked = candidate_idx[rerank_order]
            fallback = top_k_indices(candidate_scores, 10)
            ranked = merge_unique(reranked, fallback, limit=10)

        rankings.append(ranked)
        positives.append([int(x) for x in image_to_caption_indices[img_idx]])

    return recall_from_rankings(rankings, positives)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    print()
    print("Cross-Modal Retrieval Recall")
    print("=" * 72)
    print(f"{'Direction':<18} {'R@1':>10} {'R@5':>10} {'R@10':>10}")
    print("-" * 72)
    for direction, metrics in results.items():
        print(
            f"{direction:<18} "
            f"{format_pct(metrics['recall@1']):>10} "
            f"{format_pct(metrics['recall@5']):>10} "
            f"{format_pct(metrics['recall@10']):>10}"
        )
    print("=" * 72)
    print()


def write_json_report(
    payload: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_markdown_report(payload: Dict[str, Any], output_path: Path) -> None:
    metrics = payload["metrics"]
    md = textwrap.dedent(
        f"""\
        # Cross-Modal Retrieval Benchmark

        - Date: {payload["timestamp"]}
        - Dataset requested: `{payload["dataset_requested"]}`
        - Dataset used: `{payload["dataset_used"]}` (`{payload["dataset_source"]}` split `{payload["split"]}`)
        - Images: {payload["num_images"]}
        - Captions: {payload["num_captions"]}
        - Backend requested: `{payload["backend_requested"]}`
        - Backend used: `{payload["backend_used"]}`
        - Device: `{payload["device"]}`
        - Mode: `{payload["mode"]}`
        - Embedding cache hit: `{payload["embedding_cache_hit"]}`

        ## Results

        | Direction | R@1 | R@5 | R@10 |
        |---|---:|---:|---:|
        | Text->Image | {metrics["text_to_image"]["recall@1"] * 100:.2f}% | {metrics["text_to_image"]["recall@5"] * 100:.2f}% | {metrics["text_to_image"]["recall@10"] * 100:.2f}% |
        | Image->Text | {metrics["image_to_text"]["recall@1"] * 100:.2f}% | {metrics["image_to_text"]["recall@5"] * 100:.2f}% | {metrics["image_to_text"]["recall@10"] * 100:.2f}% |
        """
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")


def load_backend(backend_name: str, mode: str):
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from recallforge.backends.torch_backend import TorchBackend

    # Prefer MLX 4-bit on Apple Silicon for default auto path.
    quantization = os.environ.get("RECALLFORGE_MLX_QUANTIZE", "4bit")
    if backend_name in {"auto", "mlx"} and platform.system() == "Darwin" and platform.machine() == "arm64":
        quantization = "4bit"

    def _mlx_runtime_ok() -> bool:
        cmd = [sys.executable, "-c", "import mlx.core as mx; import mlx_vlm; print('ok')"]
        probe = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        return probe.returncode == 0

    backend: Any
    if backend_name == "torch":
        backend = TorchBackend(mode=mode)
    elif backend_name == "mlx":
        if not _mlx_runtime_ok():
            raise RuntimeError(
                "MLX backend requested, but MLX runtime failed to initialize. "
                "Try --backend torch."
            )
        from recallforge.backends.mlx_backend import MLXBackend

        backend = MLXBackend(mode=mode, quantization=quantization)
    elif backend_name == "auto":
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        if is_apple_silicon and _mlx_runtime_ok():
            try:
                from recallforge.backends.mlx_backend import MLXBackend

                backend = MLXBackend(mode=mode, quantization=quantization)
            except Exception as exc:
                print(f"[warn] Failed to initialize MLX backend ({exc}); falling back to torch.")
                backend = TorchBackend(mode=mode)
        else:
            if is_apple_silicon:
                print("[warn] MLX preflight failed; falling back to torch.")
            backend = TorchBackend(mode=mode)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    backend.set_mode(mode)
    info = backend.get_info()
    return backend, info


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be > 0")

    ensure_paths()
    start_time = time.time()

    print(f"[setup] backend={args.backend} mode={args.mode} dataset={args.dataset} limit={args.limit}")
    backend, backend_info = load_backend(args.backend, args.mode)
    print(
        f"[setup] resolved backend={backend_info.name} "
        f"device={backend_info.device} dtype={backend_info.dtype}"
    )

    bundle = build_bundle(args.dataset, limit=args.limit)
    images = bundle.images
    captions, _ = flatten_caption_view(images)
    print(
        f"[dataset] source={bundle.dataset_source} split={bundle.split} "
        f"images={len(images)} captions={len(captions)}"
    )

    image_emb, caption_emb, caption_to_image, cache_hit, cache_path = load_or_build_embeddings(
        backend=backend,
        backend_info=backend_info,
        bundle=bundle,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
    )

    image_doc_texts = [" ".join(sample.captions) for sample in images]
    image_to_caption_indices = build_image_positive_map(caption_to_image, num_images=len(images))

    text_to_image = evaluate_text_to_image(
        mode=args.mode,
        backend=backend,
        image_embeddings=image_emb,
        caption_embeddings=caption_emb,
        captions=captions,
        caption_to_image=caption_to_image,
        image_doc_texts=image_doc_texts,
        rerank_top_k=args.rerank_top_k,
    )
    image_to_text = evaluate_image_to_text(
        mode=args.mode,
        backend=backend,
        image_embeddings=image_emb,
        caption_embeddings=caption_emb,
        captions=captions,
        image_to_caption_indices=image_to_caption_indices,
        rerank_top_k=args.rerank_top_k,
    )
    results = {
        "text_to_image": text_to_image,
        "image_to_text": image_to_text,
    }

    print_results_table(results)

    elapsed = time.time() - start_time
    payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_requested": args.dataset,
        "dataset_used": bundle.dataset_name,
        "dataset_source": bundle.dataset_source,
        "split": bundle.split,
        "limit": args.limit,
        "num_images": len(images),
        "num_captions": len(captions),
        "backend_requested": args.backend,
        "backend_used": backend_info.name,
        "device": backend_info.device,
        "dtype": backend_info.dtype,
        "mode": args.mode,
        "embedding_cache_hit": cache_hit,
        "embedding_cache_path": str(cache_path),
        "cache_root": str(CACHE_ROOT),
        "metrics": results,
        "elapsed_seconds": elapsed,
    }
    write_json_report(payload, RESULT_JSON_PATH)
    write_markdown_report(payload, RESULT_MD_PATH)
    print(f"[output] JSON: {RESULT_JSON_PATH}")
    print(f"[output] Markdown: {RESULT_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
