import unittest
import numpy as np
from PIL import Image
from inspection.features import patch_features
from inspection.model import PatchAnomalyDetector
from inspection.visualization import save_overlay
import tempfile
from pathlib import Path

class InspectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.good = [Image.fromarray(np.clip(205 + rng.normal(0, 5, (256, 256)), 0, 255).astype('uint8')) for _ in range(6)]

    def test_features_have_fixed_shape_and_finite_values(self):
        features = patch_features(self.good[0])
        self.assertEqual(features.shape, (256, 8))
        self.assertTrue(np.isfinite(features).all())

    def test_detector_requires_fit(self):
        with self.assertRaises(RuntimeError):
            PatchAnomalyDetector().predict(self.good[0])

    def test_localized_defect_scores_higher_than_normal(self):
        detector = PatchAnomalyDetector(random_state=3, n_estimators=100).fit(self.good)
        normal_score = detector.predict(self.good[0])["image_score"]
        defect = np.asarray(self.good[0]).copy()
        defect[96:160, 96:160] = 20
        result = detector.predict(Image.fromarray(defect))
        self.assertEqual(result["anomaly_map"].shape, (256, 256))
        self.assertGreater(result["image_score"], normal_score)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preview.png"
            save_overlay(Image.fromarray(defect), result["anomaly_map"], str(output))
            self.assertTrue(output.is_file())

if __name__ == "__main__":
    unittest.main()
