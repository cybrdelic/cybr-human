"""Separate checkpoint-selected provider of unchanged exterior energy branches."""
from pathlib import Path
import numpy as np
from coarse_physical_site_contact import create_guard as create_base_guard, sha
from coarse_physical_site_contact_rolling import RollingPhysicalSiteContact
from exterior_feature_bundles import build_contact_branches


class ExteriorBranchPhysicalContact(RollingPhysicalSiteContact):
    def contact_branches(self, q):
        cached=getattr(self,'_branch_cache',None)
        if cached is not None and cached[1]==self.barrier_scale and np.array_equal(cached[0],q):
            return cached[2]
        result=build_contact_branches(self,q)
        self._branch_cache=(np.asarray(q).copy(),self.barrier_scale,result)
        return result


def create_guard(solid, constraints):
    guard=ExteriorBranchPhysicalContact.from_base(create_base_guard(solid,constraints))
    here=Path(__file__).resolve().parent
    guard.provenance.update(exterior_branch_representation='complete_exterior_finite_energy_branches_v1',
        exterior_branch_provider_sha256=sha(Path(__file__)),
        exterior_branch_source_sha256=sha(here/'exterior_feature_bundles.py'),
        original_minimum_distance_energy_unchanged=True,
        generalized_stationarity_responsibility='Mechanics energy-epigraph KKT; provider does not select mixture reactions')
    return guard
