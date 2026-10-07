"""Code-authored neutral head; references are validation, never surface inputs.

Independent of envelope fitting, renderer morphs and FEM. One authored adult
morphology, not a universal norm or anatomically accepted simulation mesh.
"""

from pathlib import Path
import json, hashlib
import numpy as np
import trimesh
import head_features as face
from ear_surface import attached_pinna

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output/neutral-head/v5"
CONTRACT = {
    "units": "mm",
    "external_mesh_input": False,
    "image_input": False,
    "eye_spacing": 62.0,
    "eye_width": 26.5,
    "eye_opening": 7.6,
    "eye_tilt": 0.7,
    "mouth_width": 51.0,
    "construction": "one authored section loft and bicubic facial chart with explicit aperture topology",
    "validation_references": [
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7139934/",
        "https://arxiv.org/abs/0803.3924",
    ],
    "acceptance": {
        "visual": False,
        "internal_registration": "native atlas frame; sampled chart containment only",
        "fem": False,
    },
}


from anatomical_surface import AnatomicalSurface as NeutralAnatomy, head_mesh


def main(output_dir=OUT):
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    a = NeutralAnatomy(CONTRACT)
    surface, aperture_proof = head_mesh(a)
    head = trimesh.Trimesh(surface.vertices, surface.faces, process=True)
    ear_proofs = []
    for side in [-1, 1]:
        center = a.point(np.asarray(side * 1.57), np.asarray(8.0))
        ev, ef, remove, proof = attached_pinna(head.vertices, head.faces, side, center)
        keep = np.ones(len(head.faces), bool)
        keep[remove] = False
        head = trimesh.Trimesh(
            np.vstack([head.vertices, ev]),
            np.vstack([head.faces[keep], ef + len(head.vertices)]),
            process=True,
        )
        ear_proofs.append(proof)
    head.remove_unreferenced_vertices()
    head.fix_normals()
    parts = [
        face.surface_part("Continuous_anatomical_skin", head.vertices, head.faces),
        face.mouth_interior(a),
    ]
    for side in [-1, 1]:
        prefix = "Right" if side < 0 else "Left"
        c = a.eye(side)
        parts.extend(face.nasal_cavity(a, side))
        parts.extend(face.conjunctival_crescents(a, side))
        parts.append(
            face.ellipsoid(
                prefix + "_sclera",
                c,
                [face.EYE_RADIUS] * 3,
                face.SCLERA,
                nu=96,
                nv=64,
                iris_cut=True,
            )
        )
        parts.append(
            face.disk(
                prefix + "_iris",
                c,
                5.72,
                face.IRIS,
                lambda r: -11.2 + 0.018 * r * r,
                rmin=1.9,
            )
        )
        parts.append(
            face.disk(
                prefix + "_pupil",
                c,
                1.92,
                face.PUPIL,
                lambda r: np.full_like(r, -10.90),
                nr=12,
            )
        )
        theta = np.linspace(0, 2 * np.pi, 257)
        x = c[0] + side * a.p.eye_width / 2 * np.cos(theta)
        _, upper, lower = a.eye_opening(x, side)
        z = np.where(np.sin(theta) >= 0, upper, lower)
        margin = np.column_stack([x, a.front(x, z) - 0.055, z])
        parts.append(
            face.tube_collection(
                prefix + "_lid_margin", [margin], [0.055], face.LID, a, 6
            )
        )
        parts.append(
            face.eyebrow_and_lashes(
                a, side, margin, np.random.default_rng(271828 + side)
            )
        )
    palette = [
        [130, 111, 99, 255],
        [128, 74, 63, 255],
        [35, 12, 12, 255],
        [215, 216, 204, 255],
        [99, 79, 42, 255],
        [3, 3, 3, 255],
        [245, 250, 255, 50],
        [45, 30, 22, 255],
    ]
    scene = trimesh.Scene()
    arrays = {}
    for i, p in enumerate(parts):
        vv = p.vertices[:, [0, 2, 1]] * [1, 1, -1] * 0.001
        material = trimesh.visual.material.PBRMaterial(
            name=str(p.material),
            baseColorFactor=palette[p.material],
            metallicFactor=0,
            roughnessFactor=0.65 if p.material == face.SKIN else 0.4,
        )
        mesh = trimesh.Trimesh(vv, p.faces, process=False)
        mesh.vertex_normals
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        scene.add_geometry(mesh, node_name=p.name, geom_name=p.name)
        arrays[f"{i}_v"] = p.vertices
        arrays[f"{i}_f"] = p.faces
    data = scene.export(file_type="glb")
    (OUT / "head.glb").write_bytes(data)
    np.savez_compressed(OUT / "geometry.npz", **arrays)
    atlas = np.load(ROOT / "assets/anatomy/geometry.npz")
    internal = trimesh.Scene()
    clearance = []
    for record in a.proof["parts"]:
        i = record["part"]
        native = atlas[f"{i}_v"].astype(float)
        mv = native[:, [0, 2, 1]] * [1, 1, -1]
        mesh = trimesh.Trimesh(mv, atlas[f"{i}_f"], process=False)
        rgba = (
            [215, 198, 157, 255]
            if record["kind"] == "bone"
            else [166, 61, 54, 255]
            if record["kind"] == "muscle"
            else [154, 188, 199, 255]
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=rgba, roughnessFactor=0.7
            )
        )
        internal.add_geometry(mesh, node_name=record["name"], geom_name=record["name"])
        points = native[(native[:, 2] >= -0.1800001) & (native[:, 2] <= 0.147)] * 1000
        if not len(points):
            continue
        x, y, z = points.T
        rx, _, _ = a.section(z)
        gap = np.minimum.reduce(
            [
                rx - abs(x),
                y - a.front(x, z),
                a.back(x, z) - y,
            ]
        )
        clearance.append(
            {
                "part": record["name"],
                "kind": record["kind"],
                "sampled_vertices": len(points),
                "minimum_chart_clearance_mm": float(gap.min()),
                "outside_vertices": int((gap < -0.01).sum()),
            }
        )
    internal_data = internal.export(file_type="glb")
    (OUT / "internal-anatomy.glb").write_bytes(internal_data)
    registration = {
        "frame": a.proof["frame"],
        "transform": np.eye(4).tolist(),
        "external_skin_input": False,
        "validation_z_range_mm": [-180.0001, 147],
        "parts": clearance,
        "method": "native-coordinate chart containment; not triangle collision or a biomechanics calibration",
        "sha256": hashlib.sha256(internal_data).hexdigest(),
    }
    (OUT / "registration.json").write_text(
        json.dumps(registration, indent=2), encoding="utf-8"
    )
    for notice in ["LICENSE.txt", "UPSTREAM-LICENSE.txt"]:
        (OUT / notice).write_bytes((ROOT / "assets/anatomy" / notice).read_bytes())
    eye_gaps = []
    for side in [-1, 1]:
        c = a.eye(side)
        x = np.linspace(c[0] - 10, c[0] + 10, 101)
        _, up, lo = a.eye_opening(x, side)
        for z in [up, lo]:
            points = np.column_stack([x, a.front(x, z), z])
            eye_gaps.extend(
                (np.linalg.norm(points - c, axis=1) - face.EYE_RADIUS).tolist()
            )
    report = {
        "construction": a.proof,
        "aperture_topology": aperture_proof,
        "skin_vertices": len(head.vertices),
        "skin_triangles": len(parts[0].faces),
        "skin_components": len(
            trimesh.Trimesh(parts[0].vertices, parts[0].faces, process=False).split(
                only_watertight=False
            )
        ),
        "ear_attachments": ear_proofs,
        "eye_rim_sphere_distance_mm": [min(eye_gaps), max(eye_gaps)],
        "eye_spacing_mm": a.p.eye_spacing,
        "mouth_width_mm": a.p.mouth_width,
        "head_glb_sha256": hashlib.sha256(data).hexdigest(),
        "geometry_input_files": [],
        "limitations": [
            "Eye corners use authored conjunctival continuation, not sphere support alone.",
            "Parameter dimensions are authored and require independent morphology validation.",
            "Native atlas support envelope; not global contact or calibrated biomechanics.",
        ],
        "acceptance": CONTRACT["acceptance"],
    }
    (OUT / "parameters.json").write_text(
        json.dumps(CONTRACT, indent=2), encoding="utf-8"
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "skin_vertices",
                    "skin_triangles",
                    "skin_components",
                    "eye_rim_sphere_distance_mm",
                    "acceptance",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
