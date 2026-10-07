"""Layered material volume for the authored head, with regional adipose depth.

Depths/materials are explicit authoring assumptions, not measured anatomy.
The coarse volume is repaired independently; no exterior skin is imported.
"""

from pathlib import Path
import json, hashlib, argparse
import numpy as np, trimesh
from build_volume import build, ROOT

OUT = ROOT / "output/neutral-tissue/v6"


def main(target_faces=4500, output_dir=OUT, head_dir=None):
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    head_dir = (
        Path(head_dir) if head_dir is not None else ROOT / "output/neutral-head/v5"
    )
    source = np.load(head_dir / "geometry.npz")
    glb = trimesh.load(head_dir / "head.glb", force="scene")
    # GLB part names are paired with the stored authored geometry in build order.
    names = list(glb.graph.nodes_geometry)
    arrays = {}
    parts = []
    count = len([k for k in source.files if k.endswith("_v")])
    skin = source["0_v"]
    x, y, z = skin.T
    from anatomical_surface import AnatomicalSurface
    from build_neutral_head import CONTRACT

    anatomy = AnatomicalSurface(CONTRACT)
    depth = np.where(y < 20, anatomy.tissue_depth(x, z), 3.0)
    thin = np.ones(len(x))
    for side in [-1, 1]:
        ellipse = ((x - side * 31) / 18) ** 2 + ((z - 30) / 12) ** 2
        thin = np.minimum(
            thin,
            np.where((ellipse < 1) & (y < 0), 0.35 + 0.65 * np.clip(ellipse, 0, 1), 1),
        )
    radius, _, _ = anatomy.section(z)
    thin = np.where((abs(x) > radius + 2) & (z > -30) & (z < 40), 0.2, thin)
    dermis = np.maximum(0.35, 1.2 * thin)
    fat = np.maximum(0.2, (depth - 1.3) * thin)
    for i in range(count):
        v = source[f"{i}_v"]
        f = source[f"{i}_f"]
        arrays[f"{i}_v"] = v * 0.001
        arrays[f"{i}_f"] = f
        arrays[f"{i}_n"] = trimesh.Trimesh(v, f, process=False).vertex_normals
        # Exporter inserts nodes deterministically using the original authored name.
        name = names[i]
        delta = OUT / f"{i}-delta.f32"
        np.zeros_like(v, dtype="<f4").tofile(delta)
        parts.append(
            {
                "node": name,
                "part": i,
                "url": delta.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(delta.read_bytes()).hexdigest(),
            }
        )
    arrays["0_component_ids"] = np.zeros(len(source["0_f"]), np.uint32)
    arrays["0_epidermis_mm"] = np.full(len(skin), 0.1)
    arrays["0_dermis_mm"] = dermis
    arrays["0_superficial_fat_mm"] = fat
    path = OUT / "skin.npz"
    np.savez_compressed(path, **arrays)
    (OUT / "skin_manifest.json").write_text(
        json.dumps({"parts": [{"name": p["node"]} for p in parts]}), encoding="utf-8"
    )
    shape = OUT / "shape.json"
    shape.write_text(json.dumps({"parts": parts}), encoding="utf-8")
    build(path, OUT, shape, target_faces=target_faces)
    model = json.loads((OUT / "model.json").read_text(encoding="utf-8"))
    model["skin_url"] = (head_dir / "head.glb").relative_to(ROOT).as_posix()
    model["skin_glb_sha256"] = hashlib.sha256(
        (ROOT / model["skin_url"]).read_bytes()
    ).hexdigest()
    model["internal_anatomy_url"] = (
        (head_dir / "internal-anatomy.glb").relative_to(ROOT).as_posix()
    )
    model["internal_anatomy_sha256"] = hashlib.sha256(
        (head_dir / "internal-anatomy.glb").read_bytes()
    ).hexdigest()
    model["registration_url"] = (
        (head_dir / "registration.json").relative_to(ROOT).as_posix()
    )
    model["internal_anatomy_license_url"] = (
        (head_dir / "LICENSE.txt").relative_to(ROOT).as_posix()
    )
    model["internal_anatomy_upstream_notice_url"] = (
        (head_dir / "UPSTREAM-LICENSE.txt").relative_to(ROOT).as_posix()
    )
    model["registration_mode"] = (
        "shared native atlas coordinates; independently checked chart containment"
    )
    model["layer_description"] = {
        "skin": "0.1 mm epidermis plus regional dermis",
        "adipose": "continuous authored tissue-depth lattice, thin eyelids/pinnae",
        "inner_boundary": "repaired prescribed substrate; not fascia or a segmented muscle volume",
    }
    model["limitations"] += [
        "Tissue depths are authored estimates, not segmented fat compartments.",
        "Native-coordinate registration does not establish moving contact or activated muscle mechanics.",
    ]
    (OUT / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "nodes": model["nodes"],
                "cells": model["tetrahedra"],
                "skin_url": model["skin_url"],
                "anatomical_acceptance": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--target-faces", type=int, default=4500)
    args = parser.parse_args()
    main(args.target_faces, output_dir=args.out.resolve())
