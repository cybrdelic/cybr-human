"""Bounded first-step passive/active driver. Fails closed without real guards.

An external guard module must define create_guard(solid, constraints). It must
provide the finite energy/tangent, endpoint, safe-step and whole-path protocol
documented in implicit_dynamics.py. No default voxel/contact substitute exists.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

from source_transfer import load_geometry_with_source_fibres, source_origin_candidate, sha
from implicit_dynamics import ImplicitDynamics, MissingGeometryCertification, source_activation
from clipped_origin_support import load_support_bundle


def save_accepted_frames(path, *, positions, velocities, times, solid):
    """Bind the report to bytes saved on disk, not an in-memory approximation."""
    np.savez_compressed(path, positions_mm=np.asarray(positions),
        velocities_mm_s=np.asarray(velocities), physical_times_s=times,
        source_initial_positions_mm=solid.X, scale_mm_per_L0=solid.mm_per_L0)
    return {"accepted_npz_path": str(Path(path).resolve()), "accepted_npz_sha256": sha(path)}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, default=here.parent/"geometry/whole-face-continuum-final.npz")
    parser.add_argument("--phases", type=Path, default=here.parent/"geometry/whole-face-muscle-phases-final.npz")
    parser.add_argument("--guard-module", type=Path)
    parser.add_argument("--support-bundle", type=Path, help="matching finite occupied source-patch evidence, required for actual guarded motion")
    parser.add_argument("--output", type=Path, default=here/"bounded-first-step")
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--wall-seconds-per-step", type=float, default=30.)
    parser.add_argument("--damping-per-second", type=float, default=0.)
    parser.add_argument("--force-tolerance-N", type=float, default=1e-7)
    args = parser.parse_args()
    if not 1 <= args.steps <= 3 or not 1 <= args.iterations <= 16 or not 0 < args.dt <= .01 or not 0 < args.wall_seconds_per_step <= 60:
        parser.error("bounded driver:1-3 steps,1-16 iterations,dt<=.01s,and wall budget<=60s per step")
    if args.output.exists():
        parser.error("output directory already exists; use a new path to preserve prior evidence")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    solid = load_geometry_with_source_fibres(args.domain, args.phases)
    if args.support_bundle is not None:
        constraints, origin_report = load_support_bundle(solid, args.support_bundle)
    else:
        constraints, _, origin_report = source_origin_candidate(solid)
    report = dict(accepted_facial_frames=0, status="preparation", physical_dt_s=args.dt,
        requested_steps_per_case=args.steps, damping_per_second=args.damping_per_second,
        gravity_mm_s2=[0., -9810., 0.], length_scale="1 L0=1mm is an explicit engineering assumption",
        source_domain_sha256=sha(args.domain), source_phase_sha256=sha(args.phases),
        physical_mass_kg=float(solid.mass.sum()*1000), initial_velocity="zero",
        initial_state="registered source, non-equilibrated transient settling", runs={})
    try:
        if args.guard_module is not None and args.support_bundle is None:
            raise MissingGeometryCertification("Actual motion requires matching occupied clipped source-patch evidence, not original carrier-only origin candidates")
        if constraints is None:
            raise MissingGeometryCertification("Incomplete source-origin carrier map; no arbitrary fallback supports")
        guard = None
        if args.guard_module is not None:
            spec = importlib.util.spec_from_file_location("whole_face_supplied_guard", args.guard_module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report.update(guard_module=str(args.guard_module.resolve()), guard_module_sha256=sha(args.guard_module))
            guard = module.create_guard(solid, constraints)
        dynamics = ImplicitDynamics(solid, guard, constraints=constraints, damping_per_second=args.damping_per_second)
        dynamics.require_guards()
        for case in ("passive", "zygomaticus_frontalis", "zygomaticus_corrugator"):
            q, v, rows, frames, velocities, times = solid.X.copy(), np.zeros_like(solid.X), [], [], [], []
            for step in range(1, args.steps+1):
                physical_time = step*args.dt
                q, v, result = dynamics.step(q, v, dt=args.dt, activation=source_activation(physical_time, case=case),
                    max_iterations=args.iterations, max_wall_seconds=args.wall_seconds_per_step,
                    force_tolerance_N=args.force_tolerance_N)
                result["physical_time_s"] = physical_time
                rows.append(result)
                if not result["accepted"]:
                    break
                frames.append(q.copy()); velocities.append(v.copy()); times.append(physical_time)
                report["accepted_facial_frames"] += 1
            report["runs"][case] = dict(reports=rows, accepted_frames=len(frames), actual_times_s=times)
            if frames:
                report["runs"][case].update(save_accepted_frames(args.output/f"{case}-accepted.npz",
                    positions=frames, velocities=velocities, times=times, solid=solid))
        report["status"] = "bounded_attempt_finished"
    except MissingGeometryCertification as exc:
        report.update(status="blocked_before_motion", blocker=str(exc))
    report["wall_seconds"] = time.perf_counter()-started
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "accepted_facial_frames", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
