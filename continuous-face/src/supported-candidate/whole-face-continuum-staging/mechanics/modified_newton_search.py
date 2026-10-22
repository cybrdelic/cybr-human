"""Ephemeral, one-use previous-source inverse for SEARCH candidates only.

Never a current-equation solve, force mixture, or stationarity certificate.
The owner creates one cache per step invocation, so process/checkpoint reloads,
new physical histories and barrier continuation calls start empty.
"""
import hashlib
import json
import numpy as np
from scipy.sparse import csr_matrix


def search_context_identity(q_previous, velocity_previous, prediction, dt,
                            activation, guard, constraints):
    """Bind history, loading, dt, barrier scale and source target, not iterate q."""
    h = hashlib.sha256()
    arrays = [q_previous, velocity_previous, prediction]
    if constraints is not None:
        arrays.append(constraints.target)
    for value in arrays:
        a = np.ascontiguousarray(value, dtype=float)
        h.update(np.asarray(a.shape, np.int64).tobytes())
        h.update(a.tobytes())
    h.update(json.dumps(dict(dt=float(dt), activation=activation,
        barrier_scale=getattr(guard, 'barrier_scale', None)), sort_keys=True,
        allow_nan=False).encode())
    return h.digest()


def _source_identity(C, n, context, equilibrate):
    if context is None:
        raise ValueError('Explicit physical/search context required for factor reuse')
    C = csr_matrix((0, n)) if C is None else csr_matrix(C)
    h = hashlib.sha256()
    for a in (np.asarray(C.shape, np.int64), C.indptr, C.indices, C.data):
        h.update(np.ascontiguousarray(a).tobytes())
    return context, h.digest(), bool(equilibrate)


class ModifiedNewtonSearchCache:
    """At most one LU, no seam columns, consumed by at most one inverse call."""
    def __init__(self):
        self.clear()

    def clear(self):
        self._factor = None
        self._identity = None

    def __getstate__(self):
        # A checkpoint must never retain/reconstitute a cached search factor.
        return {}

    def __setstate__(self, state):
        self.clear()

    def take_direction(self, gradient, C, *, context, equilibrate,
                       regularization, allowed):
        # Consume/drop before applying the inverse or allocating a new factor.
        factor, identity = self._factor, self._identity
        self.clear()
        if (not allowed or regularization != 0. or factor is None
                or identity != _source_identity(C, len(gradient), context, equilibrate)):
            return None
        factor.columns.clear()
        try:
            direction, _discarded_source_reactions, linear = factor.solve(-gradient)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None
        if not linear['finite_solution']:
            return None
        # Do not expose an old-K residual or any multiplier as current evidence.
        return direction, float(linear['solve_seconds'])

    def remember(self, factor, *, context, equilibrate, regularization):
        self.clear()
        if factor is not None and regularization == 0.:
            factor.columns.clear()
            self._identity = _source_identity(factor.C, factor.n, context, equilibrate)
            self._factor = factor
