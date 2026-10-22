import unittest
import numpy as np
from scipy.sparse import csr_matrix,diags,eye
from factored_seam_kkt import FactoredSeamKKT
from second_order_gap_search import source_metric_correction,make_gap_anchor,gap_corrected_candidate,true_max_slope
from exterior_epigraph import epigraph_direction
class CurvedGuard:
    C=eye(1,format='csr');penetration_limit=.05;barrier_band=.01
    changed=False
    def interior_curvature_data(self,q):
        r=np.linalg.norm(q[0,:2]);n=np.r_[-q[0,:2]/r,0.]
        key=('edge',1,2) if not self.changed else ('edge',3,4)
        return {'rows':[dict(site=0,owner=1,gap_mm=-r,normal=n,key=key,tied_faces=np.array([2,3]))]}
class GapTests(unittest.TestCase):
    def test_source_compatible_inverse(self):
        f=FactoredSeamKKT(diags([2.,3.]),csr_matrix([[1.,0.]]));c,r=source_metric_correction(f,csr_matrix([[0.,1.]]),[.01]);np.testing.assert_allclose(c,[0,.01]);self.assertEqual(r['source_correction_residual_mm'],0.)
    def test_dependent_rows_reject(self):
        f=FactoredSeamKKT(eye(2));
        with self.assertRaises(ValueError):source_metric_correction(f,csr_matrix([[1,0],[1,0]]),[1,2])
    def test_nonpositive_inverse_reject(self):
        f=FactoredSeamKKT(diags([-1.,1.]));
        with self.assertRaises(ValueError):source_metric_correction(f,csr_matrix([[1.,0.]]),[.01])
    def setup_case(self):
        g=CurvedGuard();q=np.array([[.049,0.,0.]]);d=np.array([[-.0001,.001,0.]]);f=FactoredSeamKKT(diags([1e8,1.,1.]));a=make_gap_anchor(g,q,f);return g,q,d,f,a
    def test_stable_edge_opening_second_order(self):
        g,q,d,f,a=self.setup_case();x,r=gap_corrected_candidate(g,q,q+d,f,a,np.array([[1.,0,0]]),None)
        self.assertTrue(r['qualified']);self.assertLess(np.linalg.norm(x-q-d),.1*np.linalg.norm(d));self.assertEqual(x[0,1],d[0,1]);self.assertFalse(r['usable_for_stationarity'])
        predicted=-.049+(-d[0,0]);plain=-np.linalg.norm((q+d)[0,:2]);fixed=-np.linalg.norm(x[0,:2]);self.assertLess(abs(fixed-predicted),.001*abs(plain-predicted))
        x2,r2=gap_corrected_candidate(g,q,q+.5*d,f,a,np.array([[1.,0,0]]),None)
        self.assertAlmostEqual(r2['maximum_site_correction_mm']/r['maximum_site_correction_mm'],.25,delta=.002)
    def test_feature_switch_reject(self):
        g,q,d,f,a=self.setup_case();g.changed=True;x,r=gap_corrected_candidate(g,q,q+d,f,a,np.ones_like(q),None);self.assertIsNone(x)
    def test_locality_reject(self):
        g,q,d,f,a=self.setup_case();d=np.array([[0.,.02,0.]]);x,r=gap_corrected_candidate(g,q,q+d,f,a,np.array([[1.,0,0]]),None);self.assertIsNone(x)
    def test_wrong_slope_reject(self):
        g,q,d,f,a=self.setup_case();x,r=gap_corrected_candidate(g,q,q+d,f,a,np.array([[-1.,0,0]]),None);self.assertIsNone(x)
    def test_true_max_slope_not_mixture(self):
        b=dict(branch_gradient_N=csr_matrix([[1.,0],[-1.,0]]),branch_energy_N_mm=np.array([1.,1.]),branch_group=np.array([0,0]),group_base_branch=np.array([0]));self.assertEqual(true_max_slope([1,0],b,[-2,0]),2.)
    def test_fresh_factor_callback(self):
        b=dict(branch_gradient_N=csr_matrix([[0.]]),branch_energy_N_mm=np.array([0.]),branch_group=np.array([0]),group_base_branch=np.array([0]));f=[];epigraph_direction(eye(1),np.array([1.]),None,b,fresh_source_factor_callback=f.append);self.assertEqual(len(f),1);self.assertEqual(f[0].n,1)
class IntegrationTests(unittest.TestCase):
    def test_corrected_endpoint_and_armijo_rejections_preserve_plain_trial(self):
        from unittest.mock import patch
        import implicit_dynamics,force_state_dynamics
        for module in [implicit_dynamics,force_state_dynamics]:
            for bad in [1.,-.0003]:
                with self.subTest(module=module.__name__,bad=bad):
                    class Solid:
                        X=np.zeros((1,3));mass=np.array([.1]);provenance={'kind':'synthetic_SOC_loop'}
                        def activation(self,a):return {}
                        def deformation(self,q):return np.eye(3)[None]
                        def energy(self,q,a):return 0.,np.zeros_like(q),1.
                        def tangent(self,q,a):return csr_matrix((3,3))
                    class Guard:
                        certification=dict.fromkeys(module.REQUIRED_GUARDS,True);fixture_only=True
                        def energy(self,q):return 0.,np.zeros_like(q)
                        def tangent(self,q):return csr_matrix((3,3))
                        def contact_accounting(self,q):return dict(barrier_gradient_N=np.zeros_like(q),penalty_gradient_N=np.zeros_like(q),barrier_energy_N_mm=0.,minimum_slack_mm=None,complementarity_bound_satisfied=True)
                        def contact_branches(self,q):return dict(branch_gradient_N=csr_matrix((1,3)),branch_energy_N_mm=np.array([0.]),branch_group=np.array([0]),group_base_branch=np.array([0]))
                        def endpoint(self,q):return dict(feasible=abs(q[0,0])<.01)
                        def path(self,a,b):return dict(feasible=True,whole_segment_certified=True)
                        def safe_step_bound(self,q,d,safety):return 1.
                    solver=module.ImplicitDynamics(Solid(),Guard(),gravity=[3.,0,0])
                    # This constant-F loop fixture has no volume change at any alpha.
                    solver.safe_volume_step_bound=lambda q,d:1.
                    info=dict(qualified=True,search_only=True,usable_for_stationarity=False,true_max_corrected_displacement_slope_N_mm=-1e-6)
                    with patch.object(module,'make_gap_anchor',return_value={1:1}),patch.object(module,'gap_corrected_candidate',return_value=(np.array([[bad,0,0]]),info)):
                        q,v,r=solver.step(Solid.X,np.zeros_like(Solid.X),dt=.01)
                    self.assertTrue(r['accepted'],r['status']);self.assertAlmostEqual(q[0,0],.0003)
                    trials=r['history'][0]['line_search_trials'];self.assertEqual(trials[0]['actual_search_candidate'],'gap_defect_corrected');self.assertEqual(trials[1]['actual_search_candidate'],'original_QP_direction')
                    self.assertFalse(r['history'][0]['actual_search_was_second_order_corrected'])
if __name__=='__main__':unittest.main()
