"""Reversible diagonal congruence for Newton/KKT directions, not physics.

For A x=b, solve (S A S) y=S b and recover x=S y. Primal scales
normalize absolute tangent diagonals; multiplier scales normalize C D rows.
All reported residuals below are recomputed in the supplied unscaled equations.
If a caller uses direction-only numerical regularization, that must be declared
separately; this linear residual does not replace the physical dynamic residual.
No artificial mass, stiffness, support, damping, or equation truncation occurs.
"""
import time
import warnings
import numpy as np
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve


def solve_kkt(K, rhs, C=None, constraint_rhs=None, *, equilibrate=True):
    K = csr_matrix(K, dtype=float)
    rhs = np.asarray(rhs, float)
    n = K.shape[0]
    C = csr_matrix((0, n)) if C is None else csr_matrix(C, dtype=float)
    m = C.shape[0]
    constraint_rhs = np.zeros(m) if constraint_rhs is None else np.asarray(constraint_rhs, float)
    if K.shape != (n, n) or rhs.shape != (n,) or C.shape[1] != n or constraint_rhs.shape != (m,):
        raise ValueError("shape-matched primal/constraint equations required")
    if any(not np.isfinite(a).all() for a in (K.data, C.data, rhs, constraint_rhs)):
        raise ValueError("finite physical linear equations required")
    primal_scale = np.ones(n)
    multiplier_scale = np.ones(m)
    if equilibrate:
        diagonal = np.abs(K.diagonal())
        missing = diagonal == 0
        if np.any(missing):
            diagonal[missing] = np.asarray(abs(K).max(axis=1).toarray()).ravel()[missing]
        positive = diagonal > 0
        primal_scale[positive] = 1/np.sqrt(diagonal[positive])
        scaled_C = C @ diags(primal_scale)
        norms = np.sqrt(np.asarray(scaled_C.multiply(scaled_C).sum(axis=1)).ravel())
        if np.any(norms <= 0):
            raise ValueError("zero constraint rows require rank deduplication, not scaling")
        multiplier_scale = 1/norms
    D, E = diags(primal_scale), diags(multiplier_scale)
    transformed_K, transformed_C = D@K@D, E@C@D
    A = bmat([[transformed_K, transformed_C.T], [transformed_C, None]], format="csc") if m else transformed_K.tocsc()
    scaled_rhs = np.r_[primal_scale*rhs, multiplier_scale*constraint_rhs]
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatrixRankWarning)
        y = spsolve(A, scaled_rhs)
    solve_seconds = time.perf_counter()-t
    x, multipliers = primal_scale*y[:n], multiplier_scale*y[n:]
    primal_residual = K@x+C.T@multipliers-rhs
    constraint_residual = C@x-constraint_rhs
    maximum = lambda a: float(np.max(np.abs(a), initial=0))
    physical_scale = abs(K)@abs(x)+abs(C.T)@abs(multipliers)+abs(rhs)
    report = dict(equilibrated=bool(equilibrate),
        method="diagonal congruence S A S with exact recovery x=S y; equations unchanged",
        primal_scale_min=float(primal_scale.min()), primal_scale_max=float(primal_scale.max()),
        constraint_scale_min=None if not m else float(multiplier_scale.min()),
        constraint_scale_max=None if not m else float(multiplier_scale.max()),
        original_primal_residual_max_N=maximum(primal_residual),
        original_constraint_residual_max_mm=maximum(constraint_residual),
        original_system_componentwise_backward_error=maximum(abs(primal_residual)/np.maximum(physical_scale, np.finfo(float).tiny)),
        solve_seconds=solve_seconds, finite_solution=bool(np.isfinite(x).all() and np.isfinite(multipliers).all()),
        residual_scope="supplied unscaled direction system, never a substitute for physical dynamic residual",
        artificial_mass_stiffness_damping_or_pins=False)
    return x, multipliers, report
