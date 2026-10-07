"""Offline rest-volume repair; regularized determinant distortion energy.
Inspired by Garanzha et al., Foldover-free maps (2021). Repairs only the coarse
simulation cage. The detailed authored skin is preserved by its offset embedding.
"""

import numpy as np
from scipy.optimize import minimize


def untangle(outer, faces, normal, depth, skin, nodes, out):
    N = len(outer)
    base = outer * 1000
    total = depth * 1000
    middle = np.minimum(skin * 1000, total * 0.6)
    sf = np.sort(faces, axis=1)
    aa, bb, cc = sf.T
    fn = np.cross(
        outer[faces[:, 1]] - outer[faces[:, 0]], outer[faces[:, 2]] - outer[faces[:, 0]]
    )
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    refs = []
    pattern = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])
    for layer in range(2):
        lo = np.zeros(len(faces)) if layer == 0 else middle[sf].mean(1)
        hi = middle[sf].mean(1) if layer == 0 else total[sf].mean(1)
        prism = np.concatenate(
            (
                base[sf] - fn[:, None, :] * lo[:, None, None],
                base[sf] - fn[:, None, :] * hi[:, None, None],
            ),
            1,
        )
        p = prism[:, pattern].reshape(-1, 4, 3)
        refs.append(np.swapaxes(p[:, 1:] - p[:, :1], 1, 2))
    inv = np.linalg.inv(np.concatenate(refs))
    target = np.vstack(
        (base - normal * middle[:, None], base - normal * total[:, None])
    )
    x = np.vstack((base, target))
    history = []
    T = len(nodes)

    def objective(flat, eps):
        rest = flat.reshape(3 * N, 3)
        free = rest[N:]
        p = rest[nodes]
        D = np.swapaxes(p[:, 1:] - p[:, :1], 1, 2)
        F = D @ inv
        J = np.linalg.det(F)
        root = np.sqrt(J * J + eps * eps)
        chi = np.where(J >= 0, 0.5 * (J + root), 0.5 * eps * eps / (root - J))
        prime = 0.5 * (1 + J / root)
        square = np.einsum("tij,tij->t", F, F)
        power = chi ** (-2 / 3)
        energy = np.mean(square * power + 0.1 * (J * J + 1) / chi)
        cof = np.empty_like(F)
        cof[:, :, 0] = np.cross(F[:, :, 1], F[:, :, 2])
        cof[:, :, 1] = np.cross(F[:, :, 2], F[:, :, 0])
        cof[:, :, 2] = np.cross(F[:, :, 0], F[:, :, 1])
        dJ = -(2 / 3) * square * power / chi * prime + 0.1 * (
            2 * J / chi - (J * J + 1) * prime / (chi * chi)
        )
        P = (2 * F * power[:, None, None] + dJ[:, None, None] * cof) / T
        G = P @ np.swapaxes(inv, 1, 2)
        gradient = np.zeros_like(rest)
        np.add.at(gradient, nodes[:, 1], G[:, :, 0])
        np.add.at(gradient, nodes[:, 2], G[:, :, 1])
        np.add.at(gradient, nodes[:, 3], G[:, :, 2])
        np.add.at(gradient, nodes[:, 0], -G.sum(2))
        delta = free - target
        scale = np.r_[np.maximum(middle, 0.25), np.maximum(total, 0.5)]
        anchor = 5 / (2 * N)
        energy += anchor * np.sum((delta / scale[:, None]) ** 2)
        gradient[N:] += 2 * anchor * delta / scale[:, None] ** 2
        gaps = np.c_[
            np.einsum("ij,ij->i", rest[:N] - free[:N], normal),
            np.einsum("ij,ij->i", free[:N] - free[N:], normal),
        ]
        deficit = np.minimum(gaps - 0.135, 0)
        weight = 100000 / N
        energy += weight * np.sum(deficit * deficit)
        g0 = 2 * weight * deficit[:, 0, None] * normal
        g1 = 2 * weight * deficit[:, 1, None] * normal
        gradient[:N] += g0
        gradient[N : 2 * N] += -g0 + g1
        gradient[2 * N :] -= g1
        outer_delta = rest[:N] - base
        outer_weight = 1000 / N
        energy += outer_weight * np.sum(outer_delta * outer_delta)
        gradient[:N] += 2 * outer_weight * outer_delta
        return float(energy), gradient.ravel()

    # Independent central-difference check of the actual repair objective.
    rng = np.random.default_rng(42)
    indices = rng.choice(x.size, 8, replace=False)
    flat = x.ravel().copy()
    _, grad = objective(flat, 0.1)
    errors = []
    for i in indices:
        h = 1e-5
        flat[i] += h
        hi = objective(flat, 0.1)[0]
        flat[i] -= 2 * h
        lo = objective(flat, 0.1)[0]
        flat[i] += h
        fd = (hi - lo) / (2 * h)
        errors.append(abs(fd - grad[i]) / max(1, abs(fd), abs(grad[i])))
    if max(errors) > 1e-5:
        raise ValueError("Rest-volume repair gradient check failed")
    for stage, eps in enumerate(
        [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    ):
        result = minimize(
            lambda q: objective(q, eps),
            x.ravel(),
            jac=True,
            method="L-BFGS-B",
            bounds=[(v - 0.5, v + 0.5) for v in base.ravel()]
            + [(None, None)] * (2 * N * 3),
            options={
                "maxiter": 180,
                "maxls": 40,
                "ftol": 1e-11,
                "gtol": 1e-7,
                "maxcor": 8,
            },
        )
        x = result.x.reshape(3 * N, 3)
        rest = x
        p = rest[nodes]
        D = np.swapaxes(p[:, 1:] - p[:, :1], 1, 2)
        J = np.linalg.det(D @ inv)
        condition = np.linalg.cond(D)
        gaps = np.c_[
            np.einsum("ij,ij->i", x[:N] - x[N : 2 * N], normal),
            np.einsum("ij,ij->i", x[N : 2 * N] - x[2 * N :], normal),
        ]
        row = {
            "stage": stage,
            "eps": eps,
            "iterations": int(result.nit),
            "energy": float(result.fun),
            "minimum_J": float(J.min()),
            "negative_cells": int((J <= 0).sum()),
            "max_condition": float(condition.max()),
            "minimum_layer_gap_mm": float(gaps.min()),
        }
        history.append(row)
        print(row, flush=True)
        if J.min() > 0.01 and condition.max() < 10000 and gaps.min() > 0.125:
            break
    report = {
        "gradient_max_relative_error": max(errors),
        "history": history,
        "maximum_outer_displacement_mm": float(
            np.linalg.norm(x[:N] - base, axis=1).max()
        ),
        "maximum_inner_displacement_mm": float(
            np.linalg.norm(x[N:] - target, axis=1).max()
        ),
    }
    import json

    (out / "untangling.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return rest * 0.001, report
