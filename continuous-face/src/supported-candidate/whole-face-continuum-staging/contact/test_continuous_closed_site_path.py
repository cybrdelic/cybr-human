import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import eye
from coarse_rigid_query import NativeRigidQuery
from continuous_closed_site_path import certify_closed_owner_site_chord
from test_compact_penetration_barrier import fixture


def point_guard(native):
    def state(q):
        r=native.query_band(q,owner_terms=True)
        return dict(ids=np.arange(len(q)),gap=r['signed_gap'],valid=r['valid'],
            force_ids=r['contact_site_index'],force_owner=r['contact_owner_id'],force_gap=r['contact_signed_gap'])
    return SimpleNamespace(C=eye(1,format='csr'),h=.1,penetration_limit=.05,rigid=native,_state=state)


class PathTests(unittest.TestCase):
    def test_tangential_near_cap_shared_triangle_no_subdivision(self):
        solid,guard=fixture()
        result=certify_closed_owner_site_chord(guard,solid.X,solid.X+[.3,0,0],max_midpoint_queries=0)
        self.assertTrue(result['feasible']);self.assertEqual(result['midpoint_queries'],0)
        self.assertEqual(result['triangle_tube_owner_interval_certificates'],2)
        no_tube=certify_closed_owner_site_chord(guard,solid.X,solid.X+[.3,0,0],max_midpoint_queries=0,enable_triangle_tubes=False)
        self.assertFalse(no_tube['feasible'])

    def test_closed_slab_hidden_crossing_rejected(self):
        V=np.array([[-1,-1,-.08],[1,-1,-.08],[1,1,-.08],[-1,1,-.08],[-1,-1,.08],[1,-1,.08],[1,1,.08],[-1,1,.08]])
        F=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        native=NativeRigidQuery(V,F,np.ones(len(F),int));guard=point_guard(native)
        self.assertTrue(native.owners[1]['closed'])
        result=certify_closed_owner_site_chord(guard,np.array([[.13,.07,.11]]),np.array([[.13,.07,-.11]]))
        self.assertFalse(result['feasible']);self.assertEqual(result['offenders'][0]['alpha'],.5)
        self.assertAlmostEqual(result['offenders'][0]['clipped_query_gap_mm'],-.08)

    def test_open_sheet_never_claims_non_crossing(self):
        native=NativeRigidQuery([[-2,-2,0],[3,-2,0],[-2,3,0]],[[0,1,2]],[1]);guard=point_guard(native)
        result=certify_closed_owner_site_chord(guard,np.array([[0,0,.2]]),np.array([[0,0,-.2]]))
        self.assertTrue(result['feasible']);self.assertFalse(result['unsigned_open_sheet_non_crossing_certified'])

    def test_cap_and_nonfinite_fail_closed(self):
        solid,guard=fixture()
        self.assertFalse(certify_closed_owner_site_chord(guard,solid.X,solid.X-[0,0,.006])['feasible'])
        with self.assertRaises(ValueError):certify_closed_owner_site_chord(guard,solid.X,solid.X*np.nan)

    def test_contradictory_anchor_cannot_hide_known_contact(self):
        solid,guard=fixture()
        guard.lower[:]=1.
        result=certify_closed_owner_site_chord(guard,solid.X,solid.X)
        self.assertFalse(result['feasible'])
        self.assertIn('contradicts',result['reason'])

    def test_stronger_far_anchor_and_invalid_bound(self):
        _,base=fixture();guard=point_guard(base.rigid)
        q0=np.array([[3.,3.,3.]]);q1=q0+[.5,0,0]
        unanchored=certify_closed_owner_site_chord(guard,q0,q1,max_midpoint_queries=0)
        self.assertFalse(unanchored['feasible'])
        guard.P0=q0.copy();guard.lower=np.array([1.])
        anchored=certify_closed_owner_site_chord(guard,q0,q1,max_midpoint_queries=0)
        self.assertTrue(anchored['feasible']);self.assertEqual(anchored['midpoint_queries'],0)
        guard.lower[:]=np.nan
        with self.assertRaises(ValueError):certify_closed_owner_site_chord(guard,q0,q1)


if __name__=='__main__':unittest.main()
