"""Focused cap-only diagnostic fixtures; no saved facial state or timestep."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import eye
from test_interior_feature_curvature import fixture
from factored_seam_kkt import FactoredSeamKKT
from second_order_gap_search import make_gap_anchor, gap_corrected_candidate
from raw_predictor_geometry import raw_predictor_geometry, cap_only_raw_endpoint
from barrier_slack_search import make_slack_anchor, check_slack_retention

class RawGeometryTests(unittest.TestCase):
    def setup_case(self, radius=.0499999):
        n=np.array([1.,1.,0.])/np.sqrt(2);t=np.array([1.,-1.,0.])/np.sqrt(2)
        solid,old,guard=fixture(-radius*n+[0.,0.,.713]);q=solid.X.copy()
        factor=FactoredSeamKKT(eye(q.size,format='csr'))
        anchor=make_gap_anchor(guard,q,factor)
        plain=q+1e-6*n+.001*t
        return solid,old,guard,q,plain,factor,anchor

    def endpoint_detail(self,guard,q):
        return dict(feasible=False,reason='finite endpoint guard',endpoint=guard.endpoint(q))

    def test_cap_only_repaired_with_original_guards_and_force(self):
        solid,old,g,q,p,f,a=self.setup_case();energy,gradient=g.energy(q)
        slack=make_slack_anchor(g,q);detail=self.endpoint_detail(g,p)
        self.assertFalse(detail['endpoint']['feasible']);self.assertTrue(cap_only_raw_endpoint(detail))
        before_cache=g.cache;before_states=list(g.state_cache);before_query_count=g.query_count
        physical_report=copy.deepcopy(g.rigid.report);physical_provenance=copy.deepcopy(g.provenance)
        # The diagnostic must not invoke ANY raw barrier, energy or physical state builder.
        with patch.object(g,'_state',side_effect=AssertionError('physical state forbidden')),patch.object(g,'interior_curvature_data',side_effect=AssertionError('raw curvature forbidden')),patch.object(g,'energy',side_effect=AssertionError('raw energy forbidden')):
            fixed,report=gap_corrected_candidate(g,q,p,f,a,gradient,None,raw_endpoint=detail)
        self.assertIsNotNone(fixed,report);self.assertTrue(report['qualified'])
        self.assertIs(g.cache,before_cache);self.assertEqual(g.query_count,before_query_count)
        self.assertEqual([id(x) for x in g.state_cache],[id(x) for x in before_states])
        self.assertEqual(g.rigid.report,physical_report);self.assertEqual(g.provenance,physical_provenance)
        self.assertIsNot(g._soc_geometry_query_context['query'],g.rigid)
        self.assertTrue(g.endpoint(fixed)['feasible']);self.assertTrue(g.path(q,fixed)['feasible'])
        self.assertTrue(check_slack_retention(g,fixed,slack)['qualified'])
        e,grad=g.energy(fixed);oe,og=old.energy(fixed)
        self.assertEqual(e,oe);np.testing.assert_array_equal(grad,og)
        self.assertLess(e,energy+1e-4*report['true_max_corrected_displacement_slope_N_mm'])
        self.assertEqual(g.penetration_limit,.05);self.assertEqual(g.barrier_scale,1.)
        # Baseline raw predictor cannot enter SOC; backtracking needs alpha<=1/8.
        self.assertFalse(g.endpoint(p)['feasible'])
        self.assertFalse(g.endpoint(q+.5*(p-q))['feasible'])
        self.assertFalse(g.endpoint(q+.25*(p-q))['feasible'])
        self.assertTrue(g.endpoint(q+.125*(p-q))['feasible'])

    def test_feasible_raw_behavior_exactly_unchanged(self):
        solid,old,g,q,p,f,a=self.setup_case(radius=.049)
        e,gradient=g.energy(q);self.assertTrue(g.endpoint(p)['feasible'])
        with patch('raw_predictor_geometry.raw_predictor_geometry',side_effect=AssertionError('must not use diagnostic')):
            x,r=gap_corrected_candidate(g,q,p,f,a,gradient,None)
        self.assertIsNotNone(x,r);self.assertNotIn('raw_predictor_geometry',r)

    def test_native_ambiguous_invalid_never_reinterpreted(self):
        solid,old,g,q,p,f,a=self.setup_case();self.endpoint_detail(g,p)
        native=getattr(g.rigid,'native',g.rigid);original=native.containment
        def ambiguous(points,**kw):
            inside,valid=original(points,**kw);return inside,np.zeros_like(valid)
        with patch.object(native,'containment',side_effect=ambiguous):
            r=raw_predictor_geometry(g,p,a)
        self.assertFalse(r['report']['qualified']);self.assertEqual(r['rows'],[])

    def test_nonfinite_native_result_rejected(self):
        solid,old,g,q,p,f,a=self.setup_case();self.endpoint_detail(g,p)
        self.assertTrue(raw_predictor_geometry(g,p,a)['report']['qualified'])
        diagnostic=g._soc_geometry_query_context['query'];original=diagnostic.query_band
        def nonfinite(*args,**kw):
            result=original(*args,**kw);result['contact_normal'][0,0]=np.nan;return result
        with patch.object(diagnostic,'query_band',side_effect=nonfinite):
            r=raw_predictor_geometry(g,p,a)
        self.assertFalse(r['report']['qualified']);self.assertEqual(r['rows'],[])

    def test_nonfinite_input_rejected(self):
        solid,old,g,q,p,f,a=self.setup_case();self.endpoint_detail(g,p)
        p[0,0]=np.nan
        r=raw_predictor_geometry(g,p,a);self.assertFalse(r['report']['qualified'])

    def test_missing_containing_witness_and_deeper_than_band_rejected(self):
        solid,old,g,q,p,f,a=self.setup_case()
        p=q+[-.1,-.1,0];self.endpoint_detail(g,p)
        r=raw_predictor_geometry(g,p,a);self.assertFalse(r['report']['qualified']);self.assertEqual(r['rows'],[])

    def test_missing_anchor_owner_rejected(self):
        solid,old,g,q,p,f,a=self.setup_case();self.endpoint_detail(g,p)
        fake={(0,999):next(iter(a.values()))}
        r=raw_predictor_geometry(g,p,fake);self.assertFalse(r['report']['qualified']);self.assertIn('missing',r['report']['reason'])

    def test_changed_and_ambiguous_features_cannot_supply_correction(self):
        solid,old,g,q,p,f,a=self.setup_case();detail=self.endpoint_detail(g,p);_,gradient=old.energy(q)
        for key in [('edge',999,1000),('vertex',999)]:
            changed={k:dict(v,key=key) for k,v in a.items()}
            x,r=gap_corrected_candidate(g,q,p,f,changed,gradient,None,raw_endpoint=detail)
            self.assertIsNone(x);self.assertEqual(r['reason'],'No positive same-feature gap defect')
        # Distinct closest witnesses are rejected by the actual same geometry verifier.
        solid,old,g=fixture([.06,.06,.06],cavity=False);p=solid.X;self.endpoint_detail(g,p)
        r=raw_predictor_geometry(g,p,{(0,1):{}})
        self.assertEqual(r['rows'],[])
        self.assertEqual(r['report']['skipped_ambiguous_or_unresolved']['ambiguous_different_closest_points'],1)

    def test_every_invalid_site_is_in_query_union(self):
        # Add a second independently invalid cached site outside anchor. Its native
        # ambiguity must invalidate the diagnostic rather than be silently ignored.
        solid,old,g,q,p,f,a=self.setup_case();self.endpoint_detail(g,p)
        from scipy.sparse import vstack
        g.C=vstack([g.C,g.C]).tocsr()
        state=g.cache
        state['ids']=np.array([0,1]);state['valid']=np.array([False,False]);state['gap']=np.repeat(state['gap'],2)
        native=getattr(g.rigid,'native',g.rigid);original=native.containment;seen=[]
        def ambiguous_second(points,**kw):
            seen.append(len(points));inside,valid=original(points,**kw);valid[-1]=False;return inside,valid
        with patch.object(native,'containment',side_effect=ambiguous_second):
            r=raw_predictor_geometry(g,p,a)
        self.assertEqual(seen,[2]);self.assertFalse(r['report']['qualified'])
        self.assertEqual(r['report']['invalid_physical_sites_rechecked'],2)

    def test_original_force_and_guard_implementations_unmodified(self):
        solid,old,g,q,p,f,a=self.setup_case()
        for name in ('energy','_state','endpoint','path','safe_step_bound','contact_branches','contact_accounting','set_barrier_scale'):
            self.assertIs(getattr(type(g),name),getattr(type(old),name))

class LoopOptInTests(unittest.TestCase):
    def test_both_loops_are_opt_in_and_keep_original_acceptance_checks(self):
        from scipy.sparse import csr_matrix
        import implicit_dynamics,force_state_dynamics
        for module in [implicit_dynamics,force_state_dynamics]:
            for enabled in [None,False,True]:
                with self.subTest(module=module.__name__,enabled=enabled):
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
                        def endpoint(self,q):return dict(feasible=abs(q[0,0])<.0002,minimum_penetration_slack_mm=.0002-abs(q[0,0]))
                        def path(self,a,b):return dict(feasible=True,whole_segment_certified=True)
                        def safe_step_bound(self,q,d,safety):return 1.
                    solver=module.ImplicitDynamics(Solid(),Guard(),gravity=[3.,0,0]);solver.safe_volume_step_bound=lambda q,d:1.
                    info=dict(qualified=True,search_only=True,usable_for_stationarity=False,true_max_corrected_displacement_slope_N_mm=-1e-6)
                    kwargs={} if enabled is None else dict(allow_cap_only_gap_search=enabled)
                    with patch.object(module,'make_gap_anchor',return_value={1:1}),patch.object(module,'gap_corrected_candidate',return_value=(np.array([[.00015,0,0]]),info)) as corrected:
                        q,v,report=solver.step(Solid.X,np.zeros_like(Solid.X),dt=.01,max_iterations=1,**kwargs)
                    row=report['history'][0];trials=row['line_search_trials']
                    self.assertEqual(row['step'],1. if enabled else .5)
                    if enabled:
                        self.assertEqual(trials[0]['actual_search_candidate'],'gap_defect_corrected')
                        self.assertFalse(corrected.call_args.kwargs['raw_endpoint']['feasible'])
                    else:
                        self.assertEqual(trials[0]['actual_search_candidate'],'original_QP_direction')
                        self.assertFalse(trials[0]['endpoint_feasible'])
                        self.assertIsNone(corrected.call_args.kwargs['raw_endpoint'])
                    self.assertTrue(row['actual_search_was_second_order_corrected'])
                    self.assertTrue(trials[-1]['endpoint_feasible']);self.assertTrue(trials[-1]['armijo_pass'])
                    self.assertTrue(trials[-1]['constraint_pass']);self.assertTrue(trials[-1]['path_feasible'])
                    self.assertTrue(trials[-1]['barrier_slack_search']['qualified'])

if __name__=='__main__':unittest.main(verbosity=2)
