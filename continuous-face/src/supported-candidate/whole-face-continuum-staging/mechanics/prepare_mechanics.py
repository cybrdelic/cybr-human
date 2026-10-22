"""Prepare and audit the isolated whole-face adapter; never run facial motion."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from source_transfer import load_geometry_with_source_fibres, source_origin_candidate
from implicit_dynamics import REQUIRED_GUARDS


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, default=here.parent/"geometry/whole-face-continuum-final.npz")
    parser.add_argument("--phases", type=Path, default=here.parent/"geometry/whole-face-muscle-phases-final.npz")
    parser.add_argument("--mm-per-L0", type=float, default=1.)
    parser.add_argument("--evaluate-material", action="store_true", help="one actual-domain material evaluation only, no solve")
    args = parser.parse_args()
    t = time.perf_counter()
    solid = load_geometry_with_source_fibres(args.domain, args.phases, mm_per_L0=args.mm_per_L0)
    audit = solid.audit()
    audit["load_seconds"] = time.perf_counter()-t
    (here/"whole-face-mechanics-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    t = time.perf_counter()
    constraints, arrays, origins = source_origin_candidate(solid)
    origins["prepare_seconds"] = time.perf_counter()-t
    np.savez_compressed(here/"source-origin-candidate.npz", **arrays)
    (here/"source-origin-candidate.json").write_text(json.dumps(origins, indent=2)+"\n")
    if args.evaluate_material:
        t = time.perf_counter()
        e, g, j = solid.energy(solid.X, {"zygomaticus": .01, "frontalis": .01})
        report = dict(kind="constitutive evaluation only, no solve or motion", energy_N_mm=e,
            max_material_force_N=float(np.linalg.norm(g, axis=1).max()), Jmin=j,
            activation={"zygomaticus": .01, "frontalis": .01}, seconds=time.perf_counter()-t)
        (here/"whole-face-first-material-evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
    status = dict(status="tested material and transient adapter; actual whole-face source geometry ingested",
        accepted_facial_frames=0, facial_nonlinear_solve_run=False,
        blockers=["Finite initial occupied geometry must be outside true bones/cartilage/lumens, not only center samples.",
            "Finite contact forces and complete endpoint/whole-path guards must cover rigid anatomy, exterior self collision, separated sheets and cavities.",
            "Carrier point lookup does not certify an actual occupied source-origin receiving patch. Use positive-area clipped patches and declared idealized source-bone offsets; complete original patch recovery and fabricated connector geometry are not required."],
        guard_contract=list(REQUIRED_GUARDS),
        support_candidate=origins["constraint_rank_audit"] if "constraint_rank_audit" in origins else {"unmapped": origins["unmapped_origin_sites"]},
        weak_cuts=dict(parent_tets_below_one_percent=audit["cut_fraction_below_1_percent"],
            minimum_parent_volume_fraction=audit["cut_fraction_min"],
            action="report and assess conditioning at first guarded step; no hidden stabilization, artificial mass, or near-zero independent supports"),
        remaining_approximations=["Voxelized anatomical domain is not exact source-triangle clipping.",
            "Nearest source rest-tet fibre transfer is source-part restricted but is a coarse constitutive interpolation.",
            "Mass is physical-volume row-sum lumped; parent-centroid consistent matrix is underintegrated and not used for dynamics.",
            "Rigid origin bonding can overconstrain coarse parent cells; affected cells and rank are explicit."],
        passive_active_plan=dict(initial="geometry-safe source reference, zero velocity, transient settling rather than static neutral",
            cases=["passive", "zygomaticus_frontalis", "zygomaticus_corrugator"],
            timestep_s=.001, ramp_duration_s=.2, gravity_mm_s2=[0., -9810., 0.],
            mass_proportional_damping_s_inverse=0., no_physical_damping_inflation=True,
            dynamic_force_tolerance_N=1e-7, nonlinear_iteration_budget_per_step=16,
            note="proposed bounded initial test settings, not a run or accepted trajectory"))
    (here/"mechanics-readiness.json").write_text(json.dumps(status, indent=2)+"\n")
    print(json.dumps({"nodes": len(solid.X), "tets": len(solid.T), "volume_L0_3": audit["physical_volume_mm3"]/args.mm_per_L0**3,
        "origin_sites": origins["total_origin_sites"], "mapped_origins": origins["mapped_origin_sites"],
        "accepted_facial_frames": 0, "status": status["status"]}, indent=2))


if __name__ == "__main__":
    main()
