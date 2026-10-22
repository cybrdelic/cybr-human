"""Isolated numerical-tangent factory; inherits unchanged physical contact law."""
from pathlib import Path
import copy
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
CONTACT = HERE.parent
for path in (CONTACT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from coarse_physical_site_contact import sha
import coarse_physical_site_contact_epigraph as epigraph
from interior_feature_curvature import interior_feature_data, assemble_interior_curvature


class InteriorCurvaturePhysicalContact(epigraph.ExteriorBranchPhysicalContact):
    @classmethod
    def from_epigraph(cls, base):
        if not isinstance(base, epigraph.ExteriorBranchPhysicalContact):
            raise TypeError('Existing exterior epigraph contact required')
        self = cls.__new__(cls)
        self.__dict__.update(base.__dict__)
        self.state_cache = list(base.state_cache)
        self.provenance = copy.deepcopy(base.provenance)
        self.provenance.update(
            interior_feature_tangent_revision='interior_signed_distance_curvature_v1',
            interior_feature_tangent_source_sha256=sha(Path(__file__)),
            interior_feature_curvature_source_sha256=sha(HERE/'interior_feature_curvature.py'),
            base_epigraph_factory_sha256=sha(Path(epigraph.__file__)),
            unchanged_energy_gradient_guards=True,
            effective_contact_tangent='Existing PSD normal approximation plus R/abs(gap) times stable interior edge/vertex tangential projector; exterior and unresolved curvature omitted; not exact Hessian everywhere')
        self.report = self.provenance
        self._interior_curvature_cache = None
        return self

    def interior_curvature_data(self, q):
        cached = self._interior_curvature_cache
        if cached is not None and cached[1] == self.barrier_scale and np.array_equal(cached[0],q):
            return cached[2]
        data = interior_feature_data(self, self._state(q))
        self._interior_curvature_cache = (np.asarray(q).copy(), self.barrier_scale, data)
        return data

    def tangent(self, q):
        original = super().tangent(q)
        return (original+assemble_interior_curvature(self,self.interior_curvature_data(q))).tocsr()


def create_guard(solid, constraints):
    return InteriorCurvaturePhysicalContact.from_epigraph(epigraph.create_guard(solid, constraints))
