"""
Dataset loaders for zero-shot learning (Ba et al. ICCV 2015).
Focuses on JSON-based data loading (List or Lines) for images and Wikipedia text.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

from .image_preprocessor import get_eval_transform
from .text_processor import texts_to_tfidf
from .text_sbert import texts_to_sbert
from .text_sbert_multi import texts_to_sbert_multi
from .text_clip import texts_to_clip


import functools


# Simple memory cache to avoid multiple disk scans
# Replaced with functools.lru_cache for better thread safety and memory management


class ImageClassDataset(Dataset):
    """
    Simple image classification dataset given paths and labels.
    Used by reproduce scripts for evaluation.
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform=None,
    ):
        """
        Args:
            image_paths: List of image file paths
            labels: List of integer labels (same length as image_paths)
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or get_eval_transform()

        assert len(image_paths) == len(labels), "image_paths and labels must have same length"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        image = self.transform(img)

        return image, label

def _parse_json_objects(content: str) -> list[dict]:
    """
    Parse JSON objects from file content.

    Supports JSONL (one JSON object per line), JSON arrays, and mixed formats.
    Uses line-by-line json.loads() first (handles braces inside string values
    correctly), then falls back to regex extraction for non-standard formats.

    Args:
        content: Raw file content containing JSON objects

    Returns:
        List of parsed JSON dictionaries
    """
    data = []

    # 1. Try JSONL: one JSON object per line (standard for .jsonl files)
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                data.append(item)
            elif isinstance(item, list):
                data.extend(obj for obj in item if isinstance(obj, dict))
        except json.JSONDecodeError:
            # Fix leading zeros in unquoted numbers (e.g. "idx": 001 → "idx": 1)
            # This is invalid JSON but present in our JSONL files
            line_fixed = re.sub(r':\s*0+(\d+)', r': \1', line)
            try:
                item = json.loads(line_fixed)
                if isinstance(item, dict):
                    data.append(item)
                elif isinstance(item, list):
                    data.extend(obj for obj in item if isinstance(obj, dict))
            except json.JSONDecodeError:
                continue

    if data:
        return data

    # 2. Try parsing the entire content as a JSON array
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [obj for obj in parsed if isinstance(obj, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # 3. Fallback: regex extraction for truly malformed files
    potential_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
    for obj_str in potential_objects:
        try:
            obj_fixed = re.sub(r':\s*0+(\d+)', r': \1', obj_str)
            item = json.loads(obj_fixed)
            data.append(item)
        except json.JSONDecodeError:
            continue
    return data


def _find_class_image_dir(
    images_base: Path,
    class_path: str,
    idx: int,
    class_name: str,
) -> Path | None:
    """
    Find the directory containing images for a given class.

    Args:
        images_base: Base directory containing images
        class_path: Class path from JSON metadata
        idx: Class index
        class_name: Name of the class

    Returns:
        Path to class image directory, or None if not found
    """
    candidates = [
        images_base / class_path.strip("/"),
        images_base / "birds" / class_path.strip("/"),
        images_base / "flowers" / class_path.strip("/"),
        images_base / f"{idx:03d}.{class_name.replace(' ', '_')}" if isinstance(class_name, str) else ""
    ]

    for cand in candidates:
        if cand and cand.exists() and cand.is_dir():
            return cand
    return None


def _collect_class_images(
    data: list[dict],
    images_base: Path,
    verbose: bool = True,
) -> tuple[list[str], list[int], dict[int, str], dict[int, str]]:
    """
    Collect image paths and metadata from parsed JSON data.

    Args:
        data: List of parsed JSON objects
        images_base: Base directory containing images
        verbose: Whether to show progress

    Returns:
        Tuple of (paths, labels, class_texts, class_names)
    """
    paths, labels = [], []
    class_texts = {}
    class_names = {}

    if verbose:
        logger.info(f"Scanning images from {images_base}...")

    for item in tqdm(data, desc="Loading classes", leave=False, disable=not verbose):
        idx_raw = item.get("idx")
        if idx_raw is None:
            continue
        idx = int(idx_raw)

        cname = item.get("class_name") or item.get("class") or f"class_{idx}"
        class_path = item.get("class_path") or f"{idx:03d}.{cname}"
        text = item.get("wikipedia_text") or item.get("text") or ""

        class_texts[idx] = text
        class_names[idx] = cname

        # Find images for this class
        found_dir = _find_class_image_dir(images_base, class_path, idx, cname)

        if found_dir:
            for img_path in found_dir.glob("*.jpg"):
                paths.append(str(img_path))
                labels.append(idx)

    return paths, labels, class_texts, class_names


@functools.lru_cache(maxsize=32)
def load_from_json(
    json_path: str | Path,
    images_base: str | Path,
    verbose: bool = True,
) -> tuple[list[str], list[int], dict[int, str], list[str]]:
    """
    Load images and text based on Wikipedia JSON file (highly robust extraction).

    Uses functools.lru_cache for automatic caching with LRU eviction policy.

    Args:
        json_path: Path to JSON/JSONL file containing class metadata
        images_base: Base directory containing image subdirectories
        verbose: Whether to print progress messages

    Returns:
        Tuple containing:
            - paths: List of image file paths
            - labels: List of integer labels corresponding to paths
            - class_texts: Dictionary mapping class ID to Wikipedia text
            - ordered_names: List of class names sorted by ID
    """
    json_path = Path(json_path).resolve()
    images_base = Path(images_base).resolve()

    if verbose:
        logger.info(f"Loading metadata from {json_path}...")

    # Read file content
    with open(json_path, encoding="utf-8") as f:
        content = f.read()

    # Parse JSON objects
    data = _parse_json_objects(content)

    # Collect images and metadata
    paths, labels, class_texts, class_names = _collect_class_images(
        data, images_base, verbose
    )

    # Sort class names by idx to ensure consistent ordering
    sorted_ids = sorted(class_names.keys())
    ordered_names = [class_names[i] for i in sorted_ids]

    return paths, labels, class_texts, ordered_names


_TEXT_ENCODER_CHOICES = ("tfidf", "sbert", "sbert_multi", "clip")


def _compute_text_features(ordered_texts: list[str], text_encoder: str) -> "np.ndarray":
    """Dispatch to the appropriate text encoding function.

    Args:
        ordered_texts: Class descriptions in class-index order.
        text_encoder: One of 'tfidf', 'sbert', 'sbert_multi', 'clip'.

    Returns:
        Float32 array of shape [num_classes, feature_dim].
    """
    if text_encoder not in _TEXT_ENCODER_CHOICES:
        raise ValueError(
            f"text_encoder must be one of {_TEXT_ENCODER_CHOICES}; got {text_encoder!r}."
        )
    if text_encoder == "tfidf":
        features, _ = texts_to_tfidf(ordered_texts, max_features=9763)
    elif text_encoder == "sbert":
        features = texts_to_sbert(ordered_texts)
    elif text_encoder == "sbert_multi":
        features = texts_to_sbert_multi(ordered_texts)
    elif text_encoder == "clip":
        features = texts_to_clip(ordered_texts)
    return features


class ZeroShotDataset(Dataset):
    """
    Zero-shot learning dataset loading from JSON.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        images_base: str | Path,
        mode: Literal["train", "test_seen", "test_unseen"] = "train",
        n_unseen: int = 40,
        train_ratio: float = 0.8,
        seed: int = 42,
        transform=None,
        verbose: bool = None,
        text_encoder: str = "tfidf",
    ):
        self.mode = mode
        self.transform = transform or get_eval_transform()

        # Only be verbose for the first dataset instance (train)
        is_verbose = verbose if verbose is not None else (mode == "train")

        # Load all data from JSON/JSONL
        self.all_paths, self.all_labels, self.class_texts, self.class_names = load_from_json(
            jsonl_path, images_base, verbose=is_verbose
        )

        # Use class_texts.keys() to identify all intended classes
        all_class_ids = sorted(list(self.class_texts.keys()))
        self.n_classes = len(all_class_ids)

        # Validate n_unseen parameter
        if n_unseen >= self.n_classes:
            raise ValueError(
                f"n_unseen ({n_unseen}) must be less than total classes ({self.n_classes}). "
                f"Need at least 1 seen class for zero-shot learning."
            )
        if n_unseen < 0:
            raise ValueError(f"n_unseen must be non-negative, got {n_unseen}")

        # Split classes into seen/unseen.
        # Shuffle positions [0..N-1] rather than the raw IDs so that the same seed
        # produces the same class partition as prepare_birds/flowers_zero_shot in
        # preparation.py (which always works with 0-indexed directory IDs).
        n_classes = len(all_class_ids)
        positions = list(range(n_classes))
        random.seed(seed)
        random.shuffle(positions)

        self.unseen_classes = sorted(all_class_ids[p] for p in positions[:n_unseen])
        self.seen_classes = sorted(all_class_ids[p] for p in positions[n_unseen:])

        # Group images by class for splitting.
        # Sort within each class so the initial order is deterministic across
        # platforms (glob() order is filesystem-dependent), matching the sorted()
        # call in _load_cub/flowers_images_and_classes used by prepare_*_zero_shot.
        class_to_images: dict[int, list[str]] = {lbl: [] for lbl in all_class_ids}
        for path, label in zip(self.all_paths, self.all_labels):
            class_to_images[label].append(path)
        for lbl in class_to_images:
            class_to_images[lbl].sort()

        # Select samples for this mode.
        # Re-seed with the same seed so the train/test image split within each
        # seen class is reproducible and matches prepare_*_zero_shot (split_seed).
        self.samples = []
        random.seed(seed)

        if mode == "test_unseen":
            for cls in self.unseen_classes:
                for path in class_to_images[cls]:
                    self.samples.append((path, cls))
        else:
            for cls in self.seen_classes:
                imgs = class_to_images[cls][:]  # copy; sort already applied above
                random.shuffle(imgs)
                split_idx = int(len(imgs) * train_ratio)

                selected = imgs[:split_idx] if mode == "train" else imgs[split_idx:]
                for path in selected:
                    self.samples.append((path, cls))

        # Compute text features once for all classes (encoder selected via text_encoder arg)
        ordered_texts = [self.class_texts.get(lbl, "") for lbl in all_class_ids]
        self.text_features = _compute_text_features(ordered_texts, text_encoder)
        # Keep alias for backward compatibility
        self.tfidf_matrix = self.text_features
        self.label_to_idx = {lbl: i for i, lbl in enumerate(all_class_ids)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        image = self.transform(img)

        text_idx = self.label_to_idx[label]
        # Note: text_features not returned here - use global text_features in training loop
        # to avoid creating thousands of redundant tensors per epoch

        return {
            "image": image,
            "label": text_idx,  # Relative index [0, C-1]
            "class_id": label,  # Original dataset ID
        }
