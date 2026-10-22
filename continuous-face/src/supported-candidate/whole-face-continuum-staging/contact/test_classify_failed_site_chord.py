import unittest
import numpy as np
from coarse_rigid_query import NativeRigidQuery
from classify_failed_site_chord import finite_cap_witnesses


def slab():
    V=np.array([[-1,-1,-.08],[1,-1,-.08],[1,1,-.08],[-1,1,-.08],[-1,-1,.08],[1,-1,.08],[1,1,.08],[-1,1,.08]])
    F=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return NativeRigidQuery(V,F,np.ones(len(F),int))


class WitnessTests(unittest.TestCase):
    def test_real_crossing_despite_query_valid_false(self):
        n=slab();P=np.array([[.13,.07,0.]])
        self.assertFalse(n.query_band(P)['valid'][0])
        r=finite_cap_witnesses(n,P,[3],[.5],.05)
        self.assertTrue(r['genuine_cap_crossing']);self.assertEqual(r['classification'],'genuine_cap_crossing')
        self.assertAlmostEqual(r['witnesses'][0]['finite_penetration_depth_mm'],.08)
        self.assertEqual(r['witnesses'][0]['all_owner_facets_enumerated'],12)

    def test_exterior_and_exact_cap_are_not_evidence(self):
        n=slab()
        for z in (.11,.03):
            r=finite_cap_witnesses(n,[[.13,.07,z]],[3],[.5],.05)
            self.assertFalse(r['genuine_cap_crossing'])

    def test_open_sheet_not_a_solid(self):
        n=NativeRigidQuery([[-2,-2,0],[3,-2,0],[-2,3,0]],[[0,1,2]],[1])
        self.assertFalse(finite_cap_witnesses(n,[[0,0,-.08]],[2],[.5],.05)['genuine_cap_crossing'])

    def test_invalid_sign_is_inconclusive(self):
        n=slab();old=n.containment
        n.containment=lambda P:(old(P)[0],np.zeros(len(P),bool))
        r=finite_cap_witnesses(n,[[.13,.07,0.]],[3],[.5],.05)
        self.assertFalse(r['genuine_cap_crossing']);self.assertFalse(r['unclassified_checked_sites'][0]['signs_verified'])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):finite_cap_witnesses(slab(),[[np.nan,0,0]],[1],[.5],.05)


if __name__=='__main__':unittest.main()
