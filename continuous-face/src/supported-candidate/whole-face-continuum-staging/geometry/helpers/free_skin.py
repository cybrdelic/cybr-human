"""Audited pure helper subset extracted from archived source. No archive main or target coordinates."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_closing,binary_fill_holes,binary_dilation,label

def part_vertices(data, meta, text):
    rows = []
    for p in meta['parts']:
        if text.lower() in p['name'].lower():
            rows.append(data['vertices'][p['vertex_start']:p['vertex_start'] + p['vertex_count']])
    return np.concatenate(rows) if rows else np.empty((0, 3))

class FatLayer:
    """Authored compressible columns localized by recovered muscle centroids.

 No skin coordinates are used to fit a thickness. Peak compartment thicknesses
 and extents are explicit design parameters, not recovered subject anatomy.
 """

    def __init__(self, anatomy, meta):
        self.compartments = []
        for side in ['l', 'r']:
            for name, extent, peak in [('Zygomaticus minor muscle.', (23, 22, 27), 5.0), ('Bucinator.', (24, 25, 29), 7.0), ('Depressor anguli oris.', (24, 18, 28), 4.5)]:
                v = part_vertices(anatomy, meta, name + side)
                if not len(v):
                    raise ValueError('Missing fat anatomical locator: ' + name + side)
                self.compartments.append((v.mean(0), np.asarray(extent), peak, name + side))

    def value_gradient(self, q):
        q = np.asarray(q)
        h = np.full(len(q), 1.0)
        gradient = np.zeros_like(q, dtype=float)
        for center, extent, peak, name in self.compartments:
            delta = q - center
            term = peak * np.exp(-0.5 * np.sum((delta / extent) ** 2, axis=1))
            h += term
            gradient -= term[:, None] * delta / extent ** 2
        delta = q - np.array([0.0, -1.0, 137.0])
        extent = np.array([16.0, 29.0, 20.0])
        nose = np.exp(-0.5 * np.sum((delta / extent) ** 2, axis=1))
        factor = 1 - 0.88 * nose
        gradient = gradient * factor[:, None] + h[:, None] * 0.88 * nose[:, None] * delta / extent ** 2
        return (h * factor, gradient)

    def thickness(self, q):
        return self.value_gradient(q)[0]

    def parameters(self):
        return [{'anatomical_locator': name, 'center': c.tolist(), 'extent': r.tolist(), 'peak_rest_thickness': p} for c, r, p, name in self.compartments]
