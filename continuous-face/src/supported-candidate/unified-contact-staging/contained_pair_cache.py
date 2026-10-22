"""One-bound exact superset reuse for the existing conservative broadphase.

Only a complete original raw pair set is cached, never eligibility-filtered
pairs. Both effective box arrays must be contained coordinatewise. The scope
is supplied by the caller; this object must not survive a safe_step_bound call.
"""
import numpy as np


class ContainedPairCache:
    def __init__(self, original):
        self.original = original
        self._entry = None

    @staticmethod
    def _effective_boxes(bounds, clearance):
        # All unsupported cases go through the original implementation, including
        # its validation, warnings, overflow behavior, and exceptions. The guard
        # is intentionally narrower than the original API, not a new tolerance.
        if np.geterr()['under'] != 'ignore':
            return None
        if type(clearance) not in (float, np.float64) or not np.isfinite(clearance) or clearance < 0:
            return None
        if not all(type(a) is np.ndarray and a.dtype == np.dtype(np.float64)
                   and a.ndim == 2 and a.shape[1] == 3 for a in bounds):
            return None
        alo, ahi, blo, bhi = bounds
        if alo.shape != ahi.shape or blo.shape != bhi.shape or not len(alo) or not len(blo):
            return None
        limit = np.finfo(np.float64).max / 16
        if clearance > limit or any(not np.isfinite(a).all() or np.any(np.abs(a) > limit) for a in bounds):
            return None
        if np.any(ahi < alo) or np.any(bhi < blo):
            return None
        # EXACT original final axis padding: no eps added to these boxes.
        # The original eps is a KD-search margin only, not a final overlap rule.
        return (np.nextafter(alo - clearance, -np.inf),
                np.nextafter(ahi + clearance, np.inf), blo, bhi)

    def __call__(self, alo, ahi, blo, bhi, clearance=0.):
        bounds = (alo, ahi, blo, bhi)
        effective = self._effective_boxes(bounds, clearance)
        if effective is not None and self._entry is not None:
            old, pairs = self._entry
            if (all(a.shape == b.shape for a, b in zip(effective, old))
                    and np.all(effective[0] >= old[0])
                    and np.all(effective[1] <= old[1])
                    and np.all(effective[2] >= old[2])
                    and np.all(effective[3] <= old[3])):
                # Boolean subsets of the original lexicographic order retain it.
                # Evaluate literally the original final predicate, in bounded
                # batches; do not infer monotonicity of alpha or float arithmetic.
                out = []
                for start in range(0, len(pairs), 100_000):
                    block = pairs[start:start + 100_000]
                    ai, j = block[:, 0], block[:, 1]
                    ok = np.all((np.nextafter(ahi[ai] + clearance, np.inf) >= blo[j])
                                & (np.nextafter(alo[ai] - clearance, -np.inf) <= bhi[j]), axis=1)
                    if np.any(ok):
                        out.append(block[ok])
                return np.concatenate(out) if out else np.empty((0, 2), int)
        result = self.original(alo, ahi, blo, bhi, clearance)
        if effective is not None:
            saved = tuple(np.array(a, copy=True) for a in effective)
            saved_pairs = np.array(result, copy=True)
            for a in (*saved, saved_pairs):
                a.flags.writeable = False
            self._entry = saved, saved_pairs
        return result
