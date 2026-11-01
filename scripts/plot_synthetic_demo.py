"""Show actual detector output on generated images; this is not an MVTec benchmark."""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inspection.model import PatchAnomalyDetector

root = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(10)
def normal():
    return np.clip(205 + rng.normal(0, 5, (256, 256)), 0, 255).astype('uint8')
train = [Image.fromarray(normal()) for _ in range(6)]
detector = PatchAnomalyDetector(random_state=3, n_estimators=100).fit(train)
clean = normal()  # Separate generated test image; not a training image.
defect = clean.copy()
defect[96:160, 96:160] = 20
mask = np.zeros_like(clean)
mask[96:160, 96:160] = 255
fig, axes = plt.subplots(2, 3, figsize=(9, 6), layout='constrained')
for row, image, target, label in [(0, clean, np.zeros_like(mask), 'Generated normal'),
                                  (1, defect, mask, 'Generated square defect')]:
    result = detector.predict(Image.fromarray(image))
    for ax, pixels, title, cmap in zip(axes[row], [image, target, result['anomaly_map']],
                                     [label, 'Known synthetic mask', 'Actual patch anomaly score'],
                                     ['gray', 'gray', 'inferno']):
        ax.imshow(pixels, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
fig.suptitle('Synthetic illustration — no MVTec data or benchmark claims')
(root / 'reports').mkdir(exist_ok=True)
fig.savefig(root / 'reports/synthetic-demo.png', dpi=150)
plt.close(fig)
print('Saved synthetic normal/mask/prediction illustration; seed=10, six normal training images.')
