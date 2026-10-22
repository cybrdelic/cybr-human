"""Bounded source-state derivative/cost check, never a physical trajectory."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np

from source_transfer import load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", type=Path, required=True)
    p.add_argument("--phases", type=Path, required=True)
    p.add_argument("--support-bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Preserve previous cut audits; choose a new output file")
    solid = load_geometry_with_source_fibres(a.domain, a.phases)
    constraints, _ = load_support_bundle(solid, a.support_bundle)
    activation = {"zygomaticus": .01, "frontalis": .01}
    t = time.perf_counter()
    e, g, j = solid.energy(solid.X, activation)
    eg_time = time.perf_counter()-t
    t = time.perf_counter()
    K = solid.tangent(solid.X, activation)
    tangent_time = time.perf_counter()-t
    d = np.random.default_rng(6501).normal(size=solid.X.shape)
    d /= np.max(np.linalg.norm(d, axis=1))
    eps = 1e-5
    ep, gp, _ = solid.energy(solid.X+eps*d, activation)
    em, gm, _ = solid.energy(solid.X-eps*d, activation)
    fd, exact = (gp-gm)/(2*eps), (K@d.ravel()).reshape(d.shape)
    g_probe = np.random.default_rng(183).normal(size=solid.X.shape)
    projected, reaction = constraints.projected_residual(g_probe)
    report = dict(scope="source-state material/tangent finite-difference check only, no nonlinear solve or physical frame",
        domain_sha256=solid.provenance["domain_sha256"], activation=activation,
        energy_gradient_seconds=eg_time, exact_tangent_seconds=tangent_time,
        matrix_shape=list(K.shape), sparse_nonzeros=K.nnz,
        peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        source_material_energy_N_mm=e, source_material_force_max_N=float(np.linalg.norm(g, axis=1).max()),
        source_minimum_J=j, finite_difference_virtual_probe_mm=eps,
        directional_energy_derivative_abs_error=float(abs((ep-em)/(2*eps)-np.sum(g*d))),
        directional_tangent_relative_error=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact), 1e-30)),
        source_constraint_error_mm=constraints.error(solid.X),
        projected_residual_relative_constraint_error=float(np.max(abs(constraints.C@projected))/np.max(abs(constraints.C@g_probe))),
        reaction_residual_reconstruction_error=float(np.linalg.norm(g_probe-projected-reaction)),
        physical_volume_mm3=float(solid.weights.sum()), mass_kg=float(solid.mass.sum()*1000),
        physical_time_advanced_s=0., accepted_facial_frames=0)
    if report["directional_tangent_relative_error"] > 1e-5 or report["projected_residual_relative_constraint_error"] > 1e-8:
        raise AssertionError(report)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
