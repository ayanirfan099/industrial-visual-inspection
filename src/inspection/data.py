"""Reader for the official MVTec AD directory layout."""
from __future__ import annotations
from pathlib import Path
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def load_mvtec_category(root: str | Path, category: str, *, limit_good: int | None = None,
                        limit_defects: int | None = None):
    """Load train/good plus test images and masks for one MVTec AD category."""
    base = Path(root) / category
    good_paths = sorted((base / "train" / "good").glob("*"))
    good_paths = [p for p in good_paths if p.suffix.lower() in IMAGE_SUFFIXES]
    test_root = base / "test"
    if not good_paths or not test_root.is_dir():
        raise FileNotFoundError(f"expected MVTec layout at {base}: train/good and test folders")
    if limit_good:
        good_paths = good_paths[:limit_good]
    training = [Image.open(p).convert("RGB") for p in good_paths]
    samples = []
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        paths = sorted(p for p in defect_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for path in paths[:limit_defects] if limit_defects else paths:
            image = Image.open(path).convert("RGB")
            defective = defect_dir.name != "good"
            mask_path = (base / "ground_truth" / defect_dir.name / f"{path.stem}_mask.png") if defective else None
            mask = Image.open(mask_path).convert("L") if mask_path and mask_path.exists() else None
            samples.append({"image": image, "label": int(defective), "mask": mask, "path": str(path)})
    if not samples:
        raise FileNotFoundError(f"no test images found under {test_root}")
    return training, samples
