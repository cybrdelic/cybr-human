"""Regression checks for the loft pinch and mirrored midline crease.

These protect known construction defects; they do not certify human likeness.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from anatomical_surface import AnatomicalSurface
from build_neutral_head import CONTRACT

a = AnatomicalSurface(CONTRACT)
z = np.linspace(-65, 100, 3301)
eps = 0.001
# Check the facial chart, excluding lid/lip support sheets inside openings
# that are removed from the actual mesh (the mouth centre is a void).
centre = a.base_front(np.zeros_like(z), z)
left = a.base_front(np.full_like(z, -eps), z)
right = a.base_front(np.full_like(z, eps), z)
assert np.max(abs(left - right)) < 1e-10, "Asymmetric central chart"
midline_slope = float(np.max(abs((right - centre) / eps)))
assert midline_slope < 0.005, "Mirrored chart retains a midline crease"
neck_z = np.linspace(-85, -45, 1001)
neck_radius = a.section(neck_z)[0]
assert np.min(neck_radius) > 54, "Ring-shaped neck pinch returned"
crown_z = np.array([153.99, 153.999, 153.9999])
crown_rx = a.section(crown_z)[0]
pole_ratio = crown_rx**2 / (154 - crown_z)
assert np.ptp(pole_ratio) / np.mean(pole_ratio) < 0.001, "Crown has a conical pole"
report = {
    "midline_slope": midline_slope,
    "minimum_neck_radius_mm": float(neck_radius.min()),
    "crown_pole_ratio_variation": float(np.ptp(pole_ratio) / np.mean(pole_ratio)),
    "limitations": "Known-defect checks only; no photorealistic or anatomical acceptance.",
}
out = (
    Path(__file__).resolve().parent.parent
    / "output/verification/shape-check.json"
)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
