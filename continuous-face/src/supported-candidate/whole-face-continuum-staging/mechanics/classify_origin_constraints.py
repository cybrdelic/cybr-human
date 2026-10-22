"""Classify existing source support candidates without changing constraints."""
from pathlib import Path
import json
import numpy as np
from source_transfer import SOURCE


def main():
    here = Path(__file__).resolve().parent
    output = here/"source-origin-constraint-eligibility.json"
    if output.exists():
        raise FileExistsError("Preserve existing support audit")
    report = json.loads((here/"source-origin-occupied-tissue-audit.json").read_text())
    with np.load(here/"source-origin-occupied-tissue-audit.npz", allow_pickle=False) as z:
        status, incidence = z["site_status"], z["finite_origin_face_incidence"]
        area, occupied = z["triangle_original_area_L0_2"], z["triangle_occupied_area_L0_2"]
        segments = z["bone_origin_segment_coverage"]
    full = area-occupied <= np.maximum(1e-8*area, 1e-10)
    full_incidence = np.zeros(len(status), int)
    tri_start = 0
    per_patch = []
    for patch in report["source_patches"]:
        family, part = patch["family"], patch["muscle_source_part"]
        with np.load(SOURCE/"source-body"/f"body-{family}.npz", allow_pickle=False) as z:
            side = int(np.where(z["data__prepared_body_packet__body_part_ids"] == part)[0][0])
            ids = z[f"data__prepared_body_packet__origin_{side}_nodes"]
            faces = z["data__boundary_faces"]
            faces = faces[np.isin(faces, ids).all(1)]
            lookup = {int(node): i for i, node in enumerate(ids)}
            local = np.asarray([[lookup[int(n)] for n in f] for f in faces])
            retained = local[full[tri_start:tri_start+len(local)]]
            begin = patch["origin_begin"]
            full_incidence[begin:begin+len(ids)] = np.bincount(retained.ravel(), minlength=len(ids))
            tri_start += len(local)
    # Codes are candidate eligibility, not permission to activate supports.
    code = np.zeros(len(status), np.int8)
    code[(status > 0) & (incidence == 0)] = 1
    code[(status > 0) & (incidence > 0) & (full_incidence == 0)] = 2
    code[(status > 0) & (full_incidence > 0)] = 3
    labels = ["carrier_only_void_exclude_or_repair", "occupied_but_no_complete_source_origin_facet_define_finite_patch",
              "occupied_on_partial_source_patch_needs_explicit_clipped_patch", "incident_to_fully_occupied_source_facet_candidate_bonding"]
    for patch in report["source_patches"]:
        begin, end = patch["origin_begin"], patch["origin_begin"]+patch["origin_count"]
        per_patch.append(dict(family=patch["family"], source_part=patch["muscle_source_part"],
            counts={name: int(np.sum(code[begin:end] == i)) for i, name in enumerate(labels)},
            whole_patch_accepted=False))
    result = dict(status="source support eligibility only; no supports changed and no attachment certificate",
        counts={name: int(np.sum(code == i)) for i, name in enumerate(labels)},
        fully_occupied_original_source_triangles=int(full.sum()), total_original_source_triangles=len(full),
        candidate_finite_facet_sites_with_fully_occupied_centerline=int(np.sum((code == 3) & (segments >= 1-1e-10))),
        original_whole_patches_eligible=0, per_patch=per_patch,
        interpretation="Code3 sites lie on actual source-body finite triangles fully represented in tissue. They are legitimate source-faithful subpatch candidates for idealized bonding, not yet collision-qualified finite connectors. Code0 sites cannot be pinned; codes1/2 need an explicit finite source-patch definition/clip or geometry repair.",
        native_source_limit="Source-body finite origin facets are verified packet geometry. Closest distance to literal native muscle facets is reported separately, up to0.233L0; these are not all exact native atlas surface vertices.",
        no_snapping=True, no_exterior_targets=True, constraints_mutated=False,
        finite_attachment_certified=False, accepted_facial_frames=0)
    output.write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(here/"source-origin-constraint-eligibility.npz", eligibility_code=code,
        fully_occupied_source_facet_incidence=full_incidence, original_source_facet_fully_occupied=full)
    print(json.dumps({"counts": result["counts"], "fully_occupied_triangles": int(full.sum()),
        "candidate_sites_with_occupied_centerline": result["candidate_finite_facet_sites_with_fully_occupied_centerline"]}, indent=2))


if __name__ == "__main__":
    main()
