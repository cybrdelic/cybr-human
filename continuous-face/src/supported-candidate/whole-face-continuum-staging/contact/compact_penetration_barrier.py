"""Numerical primal interior-point reaction for an unchanged finite site cap.

This is not a physical tissue/contact stiffness. For s=g+p, 0<s<s0,
B=mu*(-log(s/s0)+s/s0-1), lambda=mu*(1/s-1/s0).
The supplied normal tangent mu/s**2 is PSD but omits signed-gap curvature.
"""
import numpy as np


def compact_barrier_terms(gap, mu, *, penetration_limit=.05, band=.01):
    gap, mu = np.broadcast_arrays(np.asarray(gap, float), np.asarray(mu, float))
    if band <= 0 or penetration_limit <= 0 or np.any(mu <= 0) or not np.isfinite(mu).all():
        raise ValueError('Positive finite barrier parameters required')
    slack = gap+penetration_limit
    valid = np.isfinite(slack) & (slack > 0)
    active = valid & (slack < band)
    energy = np.zeros_like(slack)
    reaction = np.zeros_like(slack)
    curvature = np.zeros_like(slack)
    x = slack[active]/band
    # log1p avoids cancellation as the compact activation joins zero energy.
    energy[active] = mu[active]*(-np.log1p(x-1)+(x-1))
    reaction[active] = (mu[active]/slack[active])*(1-x)
    curvature[active] = mu[active]/slack[active]**2
    energy[~valid] = np.inf
    return dict(valid=valid, active=active, slack=slack, energy=energy,
                reaction=reaction, curvature=curvature,
                complementarity=reaction*np.maximum(slack, 0))
