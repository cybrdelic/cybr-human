"""Exact source-projected seam rank on its nonzero column support only."""
import numpy as np
from scipy.linalg import qr
from scipy.sparse.linalg import splu


def independent_seams(A, rhs, C):
    if not A.shape[0]:
        return np.empty(0, int)
    seam_columns = np.unique(A.indices)
    source_columns = np.unique(C.indices)
    disjoint = not np.intersect1d(seam_columns, source_columns).size
    columns = seam_columns if disjoint else np.union1d(seam_columns, source_columns)
    small_A = A[:, columns]
    projected = small_A.toarray()
    if C.shape[0] and not disjoint:
        small_C = C[:, columns]
        factor = splu((small_C @ small_C.T).tocsc())
        projected -= (small_C.T @ factor.solve((small_C @ small_A.T).toarray())).T
    norms = np.linalg.norm(projected, axis=1)
    raw_norms = np.sqrt(np.asarray(small_A.multiply(small_A).sum(axis=1)).ravel())
    nonzero = np.flatnonzero(norms > 1e-14*np.maximum(raw_norms, 1e-30))
    if not len(nonzero):
        if np.max(np.abs(rhs), initial=0) > 1e-12:
            raise ValueError('Contact seam incompatible with actual source support constraints')
        return nonzero
    _, R, pivot = qr((projected[nonzero]/norms[nonzero, None]).T, mode='economic', pivoting=True)
    diagonal = np.abs(np.diag(R))
    rank = int(np.sum(diagonal > diagonal.max(initial=0)*1e-10))
    return nonzero[pivot[:rank]]
