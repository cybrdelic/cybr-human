"""Read-only audit of a mapping, with a separate report next to the input.

Run under the HUMAN lock/resource guard. The exact winding cross-check is a
deterministic sample, not a certification of every containment query.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_layered_fibres_v1 import (FAMILIES, SOURCE_REGIONS, QUADRATURE,
                                   ClosedVolume, assign_fractions, closed_boundary,
                                   pinned, sha256)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Audit output already exists")
    report = json.loads((args.mapping / "report.json").read_text())
    records = json.loads((args.mapping / "families.json").read_text())
    arrays = np.load(args.mapping / "mapping.npz", allow_pickle=False)
    pinned(args.mapping / "mapping.npz", report["mapping_sha256"])
    pinned(args.mapping / "families.json", report["families_sha256"])
    for item in report["pins"].values():
        pinned(item["path"], item["sha256"])
    assigned = assign_fractions(arrays["membership"])
    for source, target in (("family_weights", "family_weights"), ("passive_fraction", "passive_fraction"),
                           ("unresolved_fraction", "unresolved_fraction")):
        np.testing.assert_array_equal(arrays[target], assigned[source])
    np.testing.assert_allclose(arrays["family_weights"].sum(axis=1) + arrays["passive_fraction"]
                               + arrays["unresolved_fraction"], 1, atol=1e-15)
    assert tuple(arrays["family_names"]) == FAMILIES
    assert tuple(arrays["source_region_names"]) == SOURCE_REGIONS
    for k, record in enumerate(records["families"]):
        assert record["id"] == FAMILIES[k] == record["activationChannel"]
        assert record["activation"] == 0 and "stiffness" not in record
        ids = np.flatnonzero(arrays["family_weights"][:, k] > 0)
        np.testing.assert_array_equal(record["tetrahedra"], ids)
        np.testing.assert_array_equal(record["volumeFractions"], arrays["family_weights"][ids, k])
        np.testing.assert_array_equal(record["fibreDirections"], arrays["fibre_directions"][ids, k])
        np.testing.assert_allclose(np.linalg.norm(arrays["fibre_directions"][ids, k], axis=1), 1, atol=1e-14)
    mesh = np.load(report["pins"]["tet"]["path"], allow_pickle=False)
    parts = np.load(report["pins"]["parts"]["path"], allow_pickle=False)
    original = np.load(report["pins"]["zygomatic"]["path"], allow_pickle=False)
    samples = np.einsum("qi,tij->tqj", QUADRATURE, mesh["vertices"][mesh["tetrahedra"]]).reshape(-1, 3)
    passive = np.setdiff1d(np.arange(len(original["tetrahedra"])), original["active_tetrahedra"])
    sources = [ClosedVolume(n, parts[n + "_vertices"], parts[n + "_triangles"])
               for n in SOURCE_REGIONS[:-1]]
    sources.append(ClosedVolume(SOURCE_REGIONS[-1], original["vertices"],
                                closed_boundary(original["tetrahedra"][passive])))
    rng = np.random.default_rng(20261011)
    exact_checks = {}
    flattened = arrays["membership"].reshape(-1, len(sources))
    for k, source in enumerate(sources):
        local = samples - source.offset
        in_box = ((local >= source.minimum) & (local <= source.maximum)).all(axis=1)
        # Balance confirmed inside and box-contained outside points, so the
        # test is not dominated by trivial AABB rejections.
        ids = []
        for value in (0, 1):
            candidates = np.flatnonzero(in_box & (flattened[:, k] == value))
            ids.extend(rng.choice(candidates, min(256, len(candidates)), replace=False).tolist())
        actual = np.array([source.exact_winding(local[i]) for i in ids])
        expected = flattened[ids, k]
        mismatches = np.flatnonzero(actual != expected)
        exact_checks[source.name] = {"sample_count": len(ids), "mismatches": len(mismatches),
                                      "mismatch_sample_ids": [ids[i] for i in mismatches[:20]]}
    passed = all(q["mismatches"] == 0 for q in exact_checks.values())
    result = {"passed": passed, "mapping_sha256": report["mapping_sha256"],
              "mesh_status": report["mesh_status"],
              "family_json_roundtrip_exact": True, "fractions_recomputed_exact": True,
              "all_five_activation_channels_distinct_and_zero": True,
              "source_geometry_pins_unchanged": True,
              "sampled_double_precision_winding_crosscheck": exact_checks,
              "audit_source_sha256": sha256(__file__),
              "scope": "Mapping invariants and deterministic sampled containment cross-check only; no solve or geometry admission."}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(argv))
