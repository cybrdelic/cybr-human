"""Verify real captures: lit skin, reversible transport, and no eye/background bleed."""

from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_propagation

out = Path("output/verification")
images = [
    np.array(Image.open(out / (name + ".png")).convert("RGB"), dtype=np.int16)
    for name in ["transport-zero", "transport-on", "transport-return"]
]
zero, on, restored = images
delta = np.abs(on - zero).max(axis=2)
lit = int((on.mean(axis=2) > 65).sum())
matches_background = (zero == zero[0, 0]).all(axis=2)
seeds = np.zeros(matches_background.shape, bool)
seeds[0, :] = seeds[-1, :] = True
seeds[:, 0] = seeds[:, -1] = True
background = binary_propagation(seeds & matches_background, mask=matches_background)
background_error = int(delta[background].max())
samples = json.loads((out / "diffusion.json").read_text(encoding="utf-8"))[
    "diagnostic"
]["eyeSamples"]
assert len(samples) == 4, "Missing projected iris samples"
eye_error = max(int(delta[y - 1 : y + 2, x - 1 : x + 2].max()) for x, y in samples)
restoration_error = int(np.abs(restored - zero).max())
result = {
    "lit_pixels": lit,
    "transport_changed_pixels": int((delta > 1).sum()),
    "background_error": background_error,
    "eye_error": eye_error,
    "restoration_error": restoration_error,
}
(out / "render-pixels.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
assert lit > 30000, "Face rendered black"
assert result["transport_changed_pixels"] > 500, "Transport had no visible effect"
assert background_error == 0, "Transport leaked into background"
assert eye_error <= 1, "Transport altered the unscattered iris"
assert restoration_error <= 1, "Transport did not restore the same image"
print(json.dumps(result))
