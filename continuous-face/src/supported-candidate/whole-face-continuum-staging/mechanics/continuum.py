"""One affine carrier continuum, physical cut-volume material integration.

The constitutive energy and exact tangent are imported unchanged from the
current production implementations. Geometry is never taken from a settled
regional frame. All positions supplied here are already-registered source L0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.util
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

B = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mechanics = _load("whole_face_exact_material", B / "mechanics.py")
_tangent = _load("whole_face_exact_tangent", B / "current_tangent.py")
PHASE_NAMES = ("matrix", "muscle", "fascia", "fat", "dermis")
MUSCLE_COLUMN = PHASE_NAMES.index("muscle")


@dataclass(frozen=True)
class Material:
    mu_N_mm2: float
    lam_N_mm2: float
    density_N_s2_mm4: float = 1e-9


# Explicit generic engineering assumptions, copied from the existing material
# categories, not patient estimates or a calibration to an exterior target.
DEFAULT_MATERIALS = {
    "matrix": Material(.0025, .0225),
    "muscle": Material(.00208, .01872),
    "fascia": Material(.0025, .0225),
    "fat": Material(.00084, .00756),
    "dermis": Material(.005, .045),
}


def implementation_provenance():
    return {name: {"path": str(B / name), "sha256": hashlib.sha256(
        (B / name).read_bytes()).hexdigest()} for name in ("mechanics.py", "current_tangent.py")}


class CutContinuum:
    """Material integration points have no independent mechanical DOFs.

    quadrature_weights are physical volume in L0^3, not carrier-tet volumes.
    Every quadrature site is counted once, and fractions sum to one there.
    A source muscle family may own a site's muscle fraction only with a unit
    source fibre. Unknown muscles remain passive (family -1).
    """

    def __init__(self, *, nodes, tets, quadrature_points, quadrature_weights,
                 quadrature_tet, quadrature_barycentric, material_fractions,
                 quadrature_muscle_family, family_names, quadrature_fibre,
                 mm_per_L0=1., materials=None, source_part=None,
                 provenance=None):
        self.mm_per_L0 = float(mm_per_L0)
        if not np.isfinite(self.mm_per_L0) or self.mm_per_L0 <= 0:
            raise ValueError("positive explicit engineering length scale required")
        self.X = np.asarray(nodes, float) * self.mm_per_L0
        self.T = np.asarray(tets, np.int64)
        self.points = np.asarray(quadrature_points, float) * self.mm_per_L0
        self.weights = np.asarray(quadrature_weights, float) * self.mm_per_L0**3
        self.parent = np.asarray(quadrature_tet, np.int64)
        self.bary = np.asarray(quadrature_barycentric, float)
        self.fractions = np.asarray(material_fractions, float)
        self.family = np.asarray(quadrature_muscle_family, np.int64)
        self.family_names = tuple(str(x) for x in family_names)
        self.fibre = np.asarray(quadrature_fibre, float)
        self.materials = dict(DEFAULT_MATERIALS if materials is None else materials)
        self.provenance = {} if provenance is None else dict(provenance)
        self.source_part = None if source_part is None else np.asarray(source_part)
        self._validate()
        dm = np.stack([self.X[self.T[:, j]]-self.X[self.T[:, 0]] for j in (1, 2, 3)], axis=2)
        self.inv = np.linalg.inv(dm)
        self.carrier_volume = np.linalg.det(dm)/6
        if np.any(self.carrier_volume <= 0):
            raise ValueError("carrier tetrahedra must have positive orientation")
        self.parent_physical_volume = np.bincount(self.parent, self.weights, minlength=len(self.T))
        if np.any(self.parent_physical_volume > self.carrier_volume * (1+1e-8)):
            raise ValueError("integrated tissue exceeds carrier volume: possible double counting")
        if np.any(self.parent_physical_volume <= 0):
            raise ValueError("unused carrier tetrahedra must be removed before mechanics")
        self.batches = self._material_batches()
        self.mass, self.consistent_mass = self._mass_matrices()
        if np.any(self.mass <= 0):
            raise ValueError("zero-mass carrier nodes require elimination, not invented support")

    def _validate(self):
        q = len(self.weights)
        shapes = ((self.X, (len(self.X), 3)), (self.T, (len(self.T), 4)),
                  (self.points, (q, 3)), (self.parent, (q,)), (self.bary, (q, 4)),
                  (self.fractions, (q, len(PHASE_NAMES))), (self.family, (q,)), (self.fibre, (q, 3)))
        if not q or not len(self.T) or any(a.shape != s or not np.isfinite(a).all() for a, s in shapes):
            raise ValueError("nonempty finite shape-matched domain arrays required")
        if len(set(self.family_names)) != len(self.family_names):
            raise ValueError("muscle family names must be unique")
        if self.T.min() < 0 or self.T.max() >= len(self.X) or self.parent.min() < 0 or self.parent.max() >= len(self.T):
            raise ValueError("invalid parent connectivity")
        if np.any(self.weights <= 0) or np.any(self.fractions < 0) or not np.allclose(self.fractions.sum(1), 1, atol=1e-12, rtol=0):
            raise ValueError("positive physical weights and partition-of-unity phase fractions required")
        if np.any(self.bary < -1e-12) or not np.allclose(self.bary.sum(1), 1, atol=1e-12, rtol=0):
            raise ValueError("quadrature must be inside its parent with partition-of-unity barycentrics")
        reconstructed = np.einsum("qa,qai->qi", self.bary, self.X[self.T[self.parent]])
        if not np.allclose(reconstructed, self.points, atol=1e-8*self.mm_per_L0, rtol=1e-10):
            raise ValueError("quadrature points do not match parent/barycentric map")
        if np.any(self.family < -1) or np.any(self.family >= len(self.family_names)):
            raise ValueError("unknown source muscle family")
        active = self.family >= 0
        if np.any(active & (self.fractions[:, MUSCLE_COLUMN] <= 0)):
            raise ValueError("source muscle family assigned outside muscle material")
        if not np.allclose(np.linalg.norm(self.fibre[active], axis=1), 1., atol=1e-10, rtol=0):
            raise ValueError("active sites need unit verified source fibres")
        for name in PHASE_NAMES:
            m = self.materials[name]
            if not np.isfinite([m.mu_N_mm2, m.lam_N_mm2, m.density_N_s2_mm4]).all() or m.mu_N_mm2 <= 0 or m.lam_N_mm2 < 0 or m.density_N_s2_mm4 <= 0:
                raise ValueError("invalid physical material parameters")

    def _material_batches(self):
        batches = []
        # Passive integration may be combined exactly within each parent and
        # phase because the parent F is constant and W is linear in mu/lambda.
        for phase, name in enumerate(PHASE_NAMES):
            w = self.weights * self.fractions[:, phase]
            passive = np.ones(len(w), bool) if phase != MUSCLE_COLUMN else self.family < 0
            pv = np.bincount(self.parent[passive], w[passive], minlength=len(self.T))
            keep = pv > 0
            if np.any(keep):
                batches.append((None, self._solid(np.where(keep)[0], pv[keep], name, None)))
        # Exact source direction distributions are retained, not averaged into
        # a made-up direction; identical parent/fibre rows may combine exactly.
        for family, name in enumerate(self.family_names):
            keep = (self.family == family) & (self.fractions[:, MUSCLE_COLUMN] > 0)
            if not np.any(keep):
                continue
            keys = np.c_[self.parent[keep], self.fibre[keep]]
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            volume = np.bincount(inverse, self.weights[keep]*self.fractions[keep, MUSCLE_COLUMN])
            batches.append((name, self._solid(unique[:, 0].astype(int), volume, "muscle", unique[:, 1:])))
        return batches

    def _solid(self, parents, volumes, phase, fibres):
        m = self.materials[phase]
        # Solid's constructor-derived carrier mass/volume are not physical cut
        # amounts. Replace volume and explicitly discard its mass immediately.
        active = np.full(len(parents), fibres is not None, bool)
        f = np.tile([1., 0, 0], (len(parents), 1)) if fibres is None else fibres
        s = _mechanics.Solid(self.X, self.T[parents], m.mu_N_mm2, m.lam_N_mm2, active, f)
        s.volume = np.asarray(volumes, float)
        s.mass = None
        return s

    def _mass_matrices(self):
        density = np.array([self.materials[p].density_N_s2_mm4 for p in PHASE_NAMES])
        site_mass = self.weights * (self.fractions @ density)
        nodes = self.T[self.parent]
        mass = np.zeros(len(self.X))
        np.add.at(mass, nodes.ravel(), (site_mass[:, None]*self.bary).ravel())
        local = site_mass[:, None, None] * self.bary[:, :, None] * self.bary[:, None, :]
        consistent = coo_matrix((local.ravel(),
            (np.repeat(nodes, 4, axis=1).ravel(), np.tile(nodes, (1, 4)).ravel())),
            shape=(len(self.X), len(self.X))).tocsr()
        return mass, consistent

    def activation(self, values):
        values = {} if values is None else dict(values)
        if set(values)-set(self.family_names):
            raise ValueError("activation refers to a muscle family absent from source labels")
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError("source-labelled activation must be in [0,1]")
        return values

    def deformation(self, q):
        q = np.asarray(q, float)
        return np.stack([q[self.T[:, j]]-q[self.T[:, 0]] for j in (1, 2, 3)], axis=2) @ self.inv

    def energy(self, q, activation=None):
        a = self.activation(activation)
        energy, gradient = 0., np.zeros_like(q, dtype=float)
        jmin = float(np.linalg.det(self.deformation(q)).min())
        if not np.isfinite(jmin) or jmin <= 0:
            return np.inf, np.full_like(q, np.nan), jmin
        for family, s in self.batches:
            e, g, _ = s.energy(q, a.get(family, 0.))
            energy += e
            gradient += g
        return float(energy), gradient, jmin

    def tangent(self, q, activation=None):
        a = self.activation(activation)
        K = csr_matrix((self.X.size, self.X.size))
        for family, s in self.batches:
            K += _tangent.tangent(s, q, a.get(family, 0.))
        return K

    def audit(self):
        phase_volume = self.weights @ self.fractions
        physical_volume = float(self.weights.sum())
        expected_mass = sum(phase_volume[k]*self.materials[name].density_N_s2_mm4 for k, name in enumerate(PHASE_NAMES))
        cut_ratio = self.parent_physical_volume/self.carrier_volume
        return dict(nodes=len(self.X), carrier_tetrahedra=len(self.T), quadrature_sites=len(self.weights),
            material_integration_rows=sum(len(s.T) for _, s in self.batches),
            physical_volume_mm3=physical_volume,
            phase_volume_mm3=dict(zip(PHASE_NAMES, map(float, phase_volume))),
            active_source_volume_mm3={name: float(np.sum(self.weights[self.family == i]*self.fractions[self.family == i, MUSCLE_COLUMN])) for i, name in enumerate(self.family_names)},
            partition_error_mm3=float(phase_volume.sum()-physical_volume),
            mass_N_s2_per_mm=float(self.mass.sum()), mass_kg=float(1000*self.mass.sum()),
            mass_conservation_error=float(self.mass.sum()-expected_mass),
            consistent_mass_row_sum_error=float(np.max(np.abs(np.asarray(self.consistent_mass.sum(1)).ravel()-self.mass))),
            cut_fraction_min=float(cut_ratio.min()), cut_fraction_below_1_percent=int(np.sum(cut_ratio < .01)),
            nodal_mass_min=float(self.mass.min()),
            length_scale_assumption=f"1 source L0 = {self.mm_per_L0:g} mm; not measured patient calibration",
            material_assumption={k: vars(v) for k, v in self.materials.items()},
            mass_discretization="row-sum lumped physical quadrature mass; consistent quadrature mass retained for audit",
            weak_cut_policy="none hidden; no artificial mass, stiffness, or independent weak supports",
            source=self.provenance, exact_implementations=implementation_provenance())


def load_domain(path, *, mm_per_L0=1.):
    with np.load(path, allow_pickle=False) as z:
        required = ("nodes", "tets", "quadrature_points", "quadrature_weights", "quadrature_tet",
                    "quadrature_barycentric", "material_fractions", "quadrature_muscle_family",
                    "family_names", "quadrature_fibre")
        arrays = {key: z[key] for key in required}
        arrays["source_part"] = z["source_part"] if "source_part" in z else None
    return CutContinuum(**arrays, mm_per_L0=mm_per_L0,
        provenance={"domain_path": str(Path(path).resolve()), "domain_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
