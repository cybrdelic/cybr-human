"""Checkpoint-selected exact screening acceleration; frozen physical law reused.

This module is a NEW factory. It does not alter the pre-existing adapter/helper.
The native distance/sign/force law, barrier, positive-J and sampled path guards
remain implemented by the frozen base adapter. No wider physical contact band.
"""
from pathlib import Path
import copy
import time
import numpy as np
from coarse_physical_site_contact import CoarsePhysicalSiteContact, create_guard as create_base_guard, sha
from rolling_exterior_bounds import RollingExteriorBounds

HERE = Path(__file__).resolve().parent


class _RecordedNativeQuery:
    def __init__(self, native):
        self.native = native
        self.calls = 0
        self.last_result = None

    def __getattr__(self, key):
        return getattr(self.native, key)

    def query_band(self, *args, **kwargs):
        result = self.native.query_band(*args, **kwargs)
        self.calls += 1
        self.last_result = result
        return result


class RollingPhysicalSiteContact(CoarsePhysicalSiteContact):
    @classmethod
    def from_base(cls, base):
        self = cls.__new__(cls)
        self.__dict__.update(base.__dict__)
        self.state_cache = list(base.state_cache)
        self.provenance = copy.deepcopy(base.provenance)
        self.rigid = _RecordedNativeQuery(base.rigid)
        self.rolling = RollingExteriorBounds(self.rigid, self.P0, self.lower,
                                            clearance=self.h, refresh_radius=.5*self.solid.mm_per_L0)
        self.provenance.update(query_revision='complete-band-owner-sum-interior-point-rolling-v4',
            base_contact_source_sha256=sha(HERE/'coarse_physical_site_contact.py'),
            rolling_screen_source_sha256=sha(HERE/'rolling_exterior_bounds.py'),
            rolling_adapter_source_sha256=sha(Path(__file__)),
            rolling_screen=dict(native_geometry_and_owner_semantics_sha256=self.rolling.token,
                numerical_refresh_radius_mm=self.rolling.refresh_radius,
                eligibility='Independent all-closed-owner valid exterior query AND inactive finite band',
                bound='Maximum of padded reference and refreshed all-facet Lipschitz lower bounds',
                physical_clearance_and_penetration_unchanged=True,
                unsigned_open_sheets_remain_contact_active=True,
                query_timing_includes_screen_and_refresh=True,
                global_surface_collision_certified=False))
        self.report = self.provenance
        return self

    def _state(self, q):
        q = np.asarray(q, float)
        if any(np.array_equal(q, cached['q']) for cached in self.state_cache):
            return super()._state(q)
        started = time.perf_counter()
        P = self.C@q
        motion = np.linalg.norm(P-self.P0, axis=1)
        effective = self.rolling.lower_bound(P)
        original = self.lower
        # Base subtracts reference motion. This exact algebraic overlay supplies
        # the stronger conservative bound without copying its energy/guard code.
        self.lower = effective+motion
        calls = self.rigid.calls
        screen_seconds = time.perf_counter()-started
        try:
            state = super()._state(q)
        finally:
            self.lower = original
        refresh_started = time.perf_counter()
        refresh = dict(stored_useful_anchors=0, refresh_seconds=0.)
        if self.rigid.calls > calls:
            refresh = self.rolling.refresh(P, state['ids'], self.rigid.last_result)
        extra = screen_seconds+time.perf_counter()-refresh_started
        self.query_seconds += extra
        state['seconds'] += extra
        state['rolling_screen'] = dict(screen_seconds=screen_seconds, **refresh)
        return state


def create_guard(solid, constraints):
    return RollingPhysicalSiteContact.from_base(create_base_guard(solid, constraints))
