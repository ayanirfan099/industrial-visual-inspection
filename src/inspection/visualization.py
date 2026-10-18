"""Save an annotated preview without extra plotting dependencies."""
from __future__ import annotations
import numpy as np
from PIL import Image
from .features import IMAGE_SIZE, grayscale_array


def save_overlay(image: Image.Image | np.ndarray, anomaly_map: np.ndarray, path: str, *, alpha: float = 0.45):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    base = np.asarray(Image.fromarray((grayscale_array(image) * 255).astype(np.uint8)).convert("RGB"), dtype=np.float32)
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"anomaly_map must have shape {(IMAGE_SIZE, IMAGE_SIZE)}")
    low, high = float(values.min()), float(values.max())
    norm = (values - low) / (high - low) if high > low else np.zeros_like(values)
    # Blue -> yellow -> red heatmap, built with NumPy for a light dependency footprint.
    red = np.clip(1.7 * norm, 0, 1)
    green = np.clip(1.7 - np.abs(norm - 0.58) * 2.4, 0, 1)
    blue = np.clip(1.0 - 1.8 * norm, 0, 1)
    color = np.stack([red, green, blue], axis=-1) * 255
    blended = np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    Image.fromarray(blended).save(path)
    return path
