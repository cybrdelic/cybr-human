import unittest
from types import SimpleNamespace
import numpy as np
from barrier_slack_search import make_slack_anchor,check_slack_retention
class Guard:
    barrier_band=.01;h=.1;penetration_limit=.05
    provenance=dict(query_revision='complete-band-owner-sum-interior-point-v3')
    def __init__(self,rows):self.rows=rows;self.calls=0
    def _state(self,q):
        self.calls+=1;r=self.rows[int(q)];slack=np.array([x[2] for x in r])
        return dict(valid=np.ones(len(r),bool),force_ids=np.array([x[0] for x in r]),force_owner=np.array([x[1] for x in r]),barrier=dict(valid=np.ones(len(r),bool),slack=slack))
class Tests(unittest.TestCase):
    def test_curved_edge_linear_bound_misses_slack_collapse(self):
        r=.0499;dnormal=-.0001;tangent=np.sqrt((.05-.000002)**2-(r+dnormal)**2)
        slack=lambda a:.05-np.sqrt((r+a*dnormal)**2+(a*tangent)**2)
        guard=Guard([[(17,4,slack(0))],[(17,4,slack(1))],[(17,4,slack(.5))]])
        anchor=make_slack_anchor(guard,0);self.assertLess(dnormal,0) # linear gap is opening
        self.assertFalse(check_slack_retention(guard,1,anchor)['qualified'])
        self.assertTrue(check_slack_retention(guard,2,anchor)['qualified'])
    def test_identity_matching_not_global_minimum(self):
        g=Guard([[(1,2,.001),(3,4,.009)],[(3,4,.001),(1,2,.001)]])
        r=check_slack_retention(g,1,make_slack_anchor(g,0));self.assertFalse(r['qualified']);self.assertEqual(r['offenders'][0]['site'],3)
    def test_exact_half_and_released_owner(self):
        g=Guard([[(1,2,.001),(3,4,.002)],[(1,2,.0005)]])
        r=check_slack_retention(g,1,make_slack_anchor(g,0));self.assertTrue(r['qualified']);self.assertEqual(r['released_owner_terms'],1)
    def test_nonfinite_nonpositive_duplicate_and_unknown_contract_fail(self):
        for rows in [[(1,2,float('nan'))],[(1,2,0)],[(1,2,.001),(1,2,.002)]]:
            with self.assertRaises(ValueError):make_slack_anchor(Guard([rows]),0)
        g=Guard([[]]);g.provenance={}
        with self.assertRaises(ValueError):make_slack_anchor(g,0)
    def test_no_barrier_and_no_active_terms_are_vacuous(self):
        self.assertIsNone(make_slack_anchor(SimpleNamespace(),0));self.assertTrue(check_slack_retention(None,0,None)['qualified'])
        g=Guard([[(1,2,.1)],[]]);self.assertTrue(check_slack_retention(g,1,make_slack_anchor(g,0))['qualified'])
    def test_current_saved_direction_should_pass(self):
        g=Guard([[(163828,4,9.068198e-9)],[(163828,4,9.068198e-9*1.2697)]])
        r=check_slack_retention(g,1,make_slack_anchor(g,0));self.assertTrue(r['qualified'])
    def test_new_entry_uses_band_reference(self):
        for initial in [[],[(1,2,.02)]]:
            g=Guard([initial,[(1,2,.000001)],[(1,2,.006)]])
            a=make_slack_anchor(g,0)
            self.assertFalse(check_slack_retention(g,1,a)['qualified'])
            self.assertTrue(check_slack_retention(g,2,a)['qualified'])
    def test_actual_qualified_rolling_revision(self):
        g=Guard([[(1,2,.001)],[(1,2,.0006)]]);g.provenance=dict(query_revision='complete-band-owner-sum-interior-point-rolling-v4')
        self.assertTrue(check_slack_retention(g,1,make_slack_anchor(g,0))['qualified'])
    def test_anchor_is_detached(self):
        g=Guard([[(1,2,.001)],[(1,2,.0001)]]);a=make_slack_anchor(g,0)
        check_slack_retention(g,1,a);self.assertEqual(a['slack_mm'][0],.001)
if __name__=='__main__':unittest.main()
