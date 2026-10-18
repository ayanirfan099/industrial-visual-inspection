"""Small, deterministic patch descriptors for CPU-friendly anomaly detection."""
from __future__ import annotations

import numpy as np
from PIL import Image

IMAGE_SIZE = 256
PATCH_SIZE = 16
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE


def grayscale_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or array to normalized 256x256 grayscale pixels."""
    if isinstance(image, Image.Image):
        image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32)
    else:
        arr = np.asarray(image)
        if arr.size == 0:
            raise ValueError("image must not be empty")
        scale = 255.0 if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        if arr.ndim == 3:
            arr = np.asarray(Image.fromarray(arr).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
        else:
            arr = np.asarray(Image.fromarray(arr).resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    if arr.size == 0:
        raise ValueError("image must not be empty")
    return np.clip(arr / (255.0 if arr.max(initial=0) > 1 else 1.0), 0.0, 1.0)


def patch_features(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return one 8-value descriptor per 16x16 patch, in row-major order."""
    gray = grayscale_array(image)
    patches = gray.reshape(GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE).transpose(0, 2, 1, 3).reshape(-1, PATCH_SIZE, PATCH_SIZE)
    gx = np.diff(patches, axis=2, prepend=patches[:, :, :1])
    gy = np.diff(patches, axis=1, prepend=patches[:, :1, :])
    flat = patches.reshape(len(patches), -1)
    return np.column_stack([
        flat.mean(axis=1), flat.std(axis=1),
        np.quantile(flat, 0.1, axis=1), np.quantile(flat, 0.9, axis=1),
        np.abs(gx).mean(axis=(1, 2)), np.abs(gy).mean(axis=(1, 2)),
        gx.std(axis=(1, 2)), gy.std(axis=(1, 2)),
    ]).astype(np.float32)
