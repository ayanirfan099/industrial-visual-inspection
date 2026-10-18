"""Normal-only patch anomaly detector with image scores and heatmaps."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from PIL import Image

from .features import GRID_SIZE, IMAGE_SIZE, patch_features


class PatchAnomalyDetector:
    """Fit only on good images; higher output scores mean more anomalous."""
    def __init__(self, *, contamination: float = 0.02, random_state: int = 42, n_estimators: int = 200):
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                     random_state=random_state, n_jobs=-1)
        self._fitted = False

    def fit(self, normal_images: list[Image.Image | np.ndarray]) -> "PatchAnomalyDetector":
        if not normal_images:
            raise ValueError("at least one defect-free training image is required")
        descriptors = np.concatenate([patch_features(image) for image in normal_images], axis=0)
        self.model.fit(descriptors)
        self._fitted = True
        return self

    def predict(self, image: Image.Image | np.ndarray) -> dict[str, object]:
        if not self._fitted:
            raise RuntimeError("fit the detector on normal images before prediction")
        descriptors = patch_features(image)
        grid = -self.model.score_samples(descriptors).reshape(GRID_SIZE, GRID_SIZE).astype(np.float32)
        score_map = Image.fromarray(grid, mode="F").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        heatmap = np.asarray(score_map, dtype=np.float32)
        return {"image_score": float(heatmap.max()), "anomaly_map": heatmap}
