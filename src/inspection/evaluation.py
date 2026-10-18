"""Image-level and pixel-level evaluation helpers."""
from __future__ import annotations
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate(detector, samples):
    labels, scores, pixel_labels, pixel_scores = [], [], [], []
    for sample in samples:
        prediction = detector.predict(sample["image"])
        labels.append(sample["label"])
        scores.append(prediction["image_score"])
        mask = sample.get("mask")
        if mask is not None:
            target = np.asarray(mask.resize((256, 256), Image.Resampling.NEAREST), dtype=np.uint8) > 0
            pixel_labels.extend(target.ravel().tolist())
            pixel_scores.extend(prediction["anomaly_map"].ravel().tolist())
    result = {"images": len(labels), "defective_images": int(sum(labels)),
              "image_auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
              "image_average_precision": float(average_precision_score(labels, scores)) if sum(labels) else None}
    if pixel_labels and len(set(pixel_labels)) == 2:
        result["pixel_auroc"] = float(roc_auc_score(pixel_labels, pixel_scores))
        result["pixel_average_precision"] = float(average_precision_score(pixel_labels, pixel_scores))
    else:
        result["pixel_auroc"] = None
        result["pixel_average_precision"] = None
    return result
