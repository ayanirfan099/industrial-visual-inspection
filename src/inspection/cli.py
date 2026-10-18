from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image
from .data import load_mvtec_category
from .evaluation import evaluate
from .model import PatchAnomalyDetector
from .visualization import save_overlay


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a normal-only patch anomaly detector")
    parser.add_argument("--data-root", required=True, help="MVTec AD extracted root directory")
    parser.add_argument("--category", default="bottle", help="MVTec category, e.g. bottle or capsule")
    parser.add_argument("--limit-good", type=int, default=None)
    parser.add_argument("--limit-defects", type=int, default=None, help="maximum test images per subfolder")
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--report", default=None, help="optional JSON report output path")
    parser.add_argument("--preview-image", default=None, help="optional image path to create an anomaly heatmap preview for")
    parser.add_argument("--preview-output", default="reports/preview.png", help="where to save the heatmap preview")
    args = parser.parse_args()
    good, test = load_mvtec_category(args.data_root, args.category,
                                     limit_good=args.limit_good, limit_defects=args.limit_defects)
    detector = PatchAnomalyDetector(contamination=args.contamination).fit(good)
    report = {"category": args.category, "training_good_images": len(good),
              "method": "IsolationForest over grayscale patch statistics", **evaluate(detector, test)}
    output = json.dumps(report, indent=2)
    print(output)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(output + "\n")
    if args.preview_image:
        sample = Image.open(args.preview_image).convert("RGB")
        prediction = detector.predict(sample)
        Path(args.preview_output).parent.mkdir(parents=True, exist_ok=True)
        save_overlay(sample, prediction["anomaly_map"], args.preview_output)
        print(f"Saved heatmap preview: {args.preview_output}")

if __name__ == "__main__":
    main()
