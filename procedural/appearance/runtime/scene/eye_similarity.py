"""A centered, uniform similarity for authored reference-eye coordinates.

The input globe and lids may already be scaled. Apply this frame only to
newly constructed reference-space details, exactly once before scene transform.
This changes authored geometry, not the documented optical approximation.
"""
import numpy as np


class Frame:
    """Map reference points to an actual eye about the current center ``c``.

    Radii and coordinates share design units. An identity frame deliberately
    does no subtraction/addition, preserving the legacy arrays bit for bit.
    """

    def __init__(self, c, reference_radius, actual_radius):
        self.c = np.array(c, dtype=float, copy=True)
        if self.c.shape != (3,) or not np.all(np.isfinite(self.c)):
            raise ValueError("Eye center must contain three finite coordinates")
        self.c.setflags(write=False)
        self.reference_radius = float(reference_radius)
        self.actual_radius = float(actual_radius)
        if (not np.isfinite(self.reference_radius)
                or not np.isfinite(self.actual_radius)
                or self.reference_radius <= 0 or self.actual_radius <= 0):
            raise ValueError("Eye radii must be finite and positive")
        self.scale = self.actual_radius / self.reference_radius

    def apply(self, points):
        """Return ``c + scale * (points - c)``; identity preserves its array."""
        points = np.asarray(points)
        if self.scale == 1.0:
            return points
        return self.c + self.scale * (points - self.c)

    def inverse(self, points):
        """Return reference coordinates; identity preserves its array."""
        points = np.asarray(points)
        if self.scale == 1.0:
            return points
        return self.c + (points - self.c) / self.scale

    def report(self):
        """JSON-safe per-eye metadata for the centered uniform mapping."""
        return {
            "center_input_units": self.c.tolist(),
            "reference_radius_input_units": self.reference_radius,
            "actual_radius_input_units": self.actual_radius,
            "uniform_scale": self.scale,
            "mapping": "current_center + scale * (reference_point - current_center)",
            "source_globe_and_lids_already_scaled": True,
        }
