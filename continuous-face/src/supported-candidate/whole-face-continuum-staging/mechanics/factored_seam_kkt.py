"""Exact factor reuse for a fixed physical/source KKT and changing seam rows.

No matrix regularization, constraint projection, changed energy, or new support.
The object is local to ONE Newton direction: K and source C are copied and frozen.
Additional A rows are numerical epigraph equations, not source attachment pins.
"""
import time
import numpy as np
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import splu


class FactoredSeamKKT:
    def __init__(self, K, C=None, *, equilibrate=True):
        started = time.perf_counter()
        self.K = csr_matrix(K, dtype=float).copy()
        self.n = self.K.shape[0]
        self.C = csr_matrix((0, self.n)) if C is None else csr_matrix(C, dtype=float).copy()
        self.m = self.C.shape[0]
        if self.K.shape != (self.n, self.n) or self.C.shape[1] != self.n:
            raise ValueError('Shape-matched fixed source KKT required')
        if not np.isfinite(self.K.data).all() or not np.isfinite(self.C.data).all():
            raise ValueError('Finite source KKT required')
        self.equilibrate = equilibrate
        dp, dc = np.ones(self.n), np.ones(self.m)
        if equilibrate:
            diagonal = np.abs(self.K.diagonal())
            missing = diagonal == 0
            if np.any(missing):
                diagonal[missing] = np.asarray(abs(self.K).max(axis=1).toarray()).ravel()[missing]
            positive = diagonal > 0
            dp[positive] = 1 / np.sqrt(diagonal[positive])
            scaled_C = self.C @ diags(dp)
            norms = np.sqrt(np.asarray(scaled_C.multiply(scaled_C).sum(axis=1)).ravel())
            if np.any(norms <= 0):
                raise ValueError('Zero source rows require explicit rank deduplication')
            dc = 1 / norms
        D, Q = diags(dp), diags(dc)
        transformed_K, transformed_C = D @ self.K @ D, Q @ self.C @ D
        B = bmat([[transformed_K, transformed_C.T], [transformed_C, None]], format='csc') if self.m else transformed_K.tocsc()
        self.scale = np.r_[dp, dc]
        self.factor = splu(B)
        self.factor_seconds = time.perf_counter() - started
        self.columns = {}
        self.solve_calls = 0
        self.inverse_column_solves = 0

    def _inverse(self, rhs):
        return self.scale * self.factor.solve(self.scale * rhs)

    def solve(self, rhs, A=None):
        """Solve [K C.T A.T; C 0 0; A 0 0] with zero constraint RHS.

        Return primal direction and [source multipliers, seam multipliers].
        Schur diagonal congruence changes variables only. Original residuals
        are recomputed after recovery; no tolerance truncates any equation.
        """
        started = time.perf_counter()
        rhs = np.asarray(rhs, float)
        A = csr_matrix((0, self.n)) if A is None else csr_matrix(A, dtype=float).copy()
        A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
        r = A.shape[0]
        if rhs.shape != (self.n,) or A.shape[1] != self.n or not np.isfinite(rhs).all() or not np.isfinite(A.data).all():
            raise ValueError('Finite shape-matched KKT right side and seam rows required')
        y0 = self._inverse(np.r_[rhs, np.zeros(self.m)])
        added = 0
        if r:
            columns = []
            for j in range(r):
                start, stop = A.indptr[j:j+2]
                indices, values = A.indices[start:stop], A.data[start:stop]
                key = (indices.tobytes(), values.tobytes())
                if key not in self.columns:
                    v = np.zeros(self.n + self.m)
                    v[indices] = values
                    self.columns[key] = self._inverse(v)
                    self.inverse_column_solves += 1
                    added += 1
                columns.append(self.columns[key])
            Z = np.column_stack(columns)
            S = np.asarray(A @ Z[:self.n])
            b = np.asarray(A @ y0[:self.n]).ravel()
            diagonal = np.abs(np.diag(S))
            # No zero-curvature regularization: dependent rows must be removed
            # by the working-set rank filter, not silently weakened here.
            if np.any(diagonal == 0):
                raise ValueError('Zero Schur curvature; rank filtering or search Hessian inspection required')
            e = 1 / np.sqrt(diagonal) if self.equilibrate else np.ones(r)
            eta = e * np.linalg.solve(e[:, None] * S * e[None, :], e * b)
            y = y0 - Z @ eta
        else:
            eta, y = np.empty(0), y0
        x, source_multiplier = y[:self.n], y[self.n:]
        multipliers = np.r_[source_multiplier, eta]
        residual = self.K @ x + self.C.T @ source_multiplier + A.T @ eta - rhs
        source_error, seam_error = self.C @ x, A @ x
        magnitude = abs(self.K) @ abs(x) + abs(self.C.T) @ abs(source_multiplier) + abs(A.T) @ abs(eta) + abs(rhs)
        maximum = lambda v: float(np.max(np.abs(v), initial=0))
        first = self.solve_calls == 0
        self.solve_calls += 1
        report = dict(method='One fixed source KKT factorization plus exact active-seam Schur complement',
            equilibrated=bool(self.equilibrate), finite_solution=bool(np.isfinite(x).all() and np.isfinite(multipliers).all()),
            original_primal_residual_max_N=maximum(residual),
            original_constraint_residual_max_mm=max(maximum(source_error), maximum(seam_error)),
            original_source_constraint_residual_max_mm=maximum(source_error),
            original_seam_constraint_residual_max_N_mm=maximum(seam_error),
            original_system_componentwise_backward_error=maximum(abs(residual) / np.maximum(magnitude, np.finfo(float).tiny)),
            solve_seconds=time.perf_counter()-started + (self.factor_seconds if first else 0.),
            factorization_count=int(first), factorization_count_cumulative=1,
            source_factorization_seconds=self.factor_seconds,
            new_inverse_seam_columns=added, cached_inverse_seam_columns=len(self.columns),
            artificial_mass_stiffness_damping_or_pins=False,
            residual_scope='Supplied unscaled direction equations; source mm and numerical seam energy Nmm remain separately typed')
        return x, multipliers, report
