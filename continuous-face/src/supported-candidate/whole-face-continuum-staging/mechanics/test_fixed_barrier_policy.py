"""Tiny deterministic wrapper tests. No face model is loaded or solved."""
import copy
import unittest
import numpy as np
from barrier_reconstruction_continuation import barrier_step
from fixed_barrier_policy import *
from accepted_history import validate_accepted_history
from paired_reconstruction_020_curvature_sequence import prospective_fixed_scale, driver_decision


class Guard:
    def __init__(self): self.scale = None; self.path_calls=[]; self.account_calls=[]; self.path_valid=True; self.slack=.002
    def set_barrier_scale(self, scale): self.scale=scale
    def max_physical_site_difference(self,a,b): return float(np.linalg.norm(a-b,axis=1).max())
    def contact_accounting(self,q,include_vectors=False):
        self.account_calls.append((self.scale,q.copy()))
        return dict(barrier_scale=self.scale,minimum_slack_mm=self.slack,active_barrier_owner_terms=1,
            complementarity_bound_satisfied=True,max_lambda_s_N_mm=self.scale*1e-5,
            per_owner=[],penalty_energy_N_mm=1.,barrier_energy_N_mm=self.scale,
            sum_lambda_s_N_mm=self.scale*1e-5,maximum_active_mu_N_mm=self.scale*1.5e-5)
    def path(self,a,b):
        self.path_calls.append((self.scale,a.copy(),b.copy()))
        return dict(feasible=self.path_valid,whole_segment_certified=self.path_valid)


class Dynamics:
    def __init__(self, offsets=(.02,.021,.0211,.02111), fail_stage=None):
        self.guard=Guard();self.calls=[];self.offsets=offsets;self.fail_stage=fail_stage
    def path_is_valid(self,r):return r.get('feasible') is True and r.get('whole_segment_certified') is True
    def step(self,p,v,*,dt,activation,initial_iterate,**kw):
        i=round(np.log(.04/self.guard.scale)/np.log(5))
        self.calls.append(dict(previous=p.copy(),velocity=v.copy(),dt=dt,activation=copy.deepcopy(activation),
            initial=None if initial_iterate is None else initial_iterate.copy(),scale=self.guard.scale,options=kw,
            qp=copy.deepcopy(getattr(self,'_epigraph_qp_checkpoint',None))))
        q=p.copy();q[:,0]+=self.offsets[i]
        accepted=i!=self.fail_stage
        inner=dict(accepted=accepted,status='converged' if accepted else 'iteration_limit',dt_s=dt,
            contact_accounting=self.guard.contact_accounting(q),dynamic_residual_max_N=self.guard.scale*1e-8,
            dynamic_residual_includes_barrier_reaction=True, contact_generalized_stationarity_qualified=True,
            generalized_contact={'selected_scale_marker':self.guard.scale}, minimum_J=.99,
            physical_frame_path_deferred=True,acceptance_scope='internal_same_timestep_stationarity_only',
            support_reaction_max_N=self.guard.scale, endpoint={'feasible':True},feasible=True,wall_seconds=.01)
        if not accepted:
            kw['diagnostic_iterate_callback'](q,inner)
        return q,(q-p)/dt,inner


class FixedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[1.,2.,3.],[4.,5.,6.]])
        self.v=np.array([[2.,3.,4.],[5.,6.,7.]])
        self.activation={'muscle':.17};self.plan=make_plan()
    def run_fixed(self,d,checked=False,**kwargs):
        t0,t1=(.024,.026) if checked else (.014,.016)
        row_request=check_request(self.plan,t0,t1)
        q,v,r=barrier_step(d,self.p,self.v,dt=t1-t0,activation=self.activation,
            barrier_policy=POLICY,sensitivity_request=row_request,**kwargs)
        r['physical_time_s']=t1
        return q,v,r
    def test_one_stage_ordinary_no_false_fresh_sensitivity(self):
        d=Dynamics();q,v,r=self.run_fixed(d)
        self.assertTrue(r['accepted']);self.assertEqual(len(d.calls),1)
        self.assertFalse(r['barrier_continuation_qualified']);self.assertFalse(r['lower_mu_comparison_passed'])
        self.assertIsNone(r['lower_mu_site_difference_mm']);self.assertNotIn('barrier_continuation',r)
        self.assertFalse(r['barrier_qualification']['comparison_performed'])
        validate_fixed_frame(r,self.p,self.v,q)
        np.testing.assert_array_equal(v,(q-self.p)/r['dt_s'])
        self.assertEqual(len(d.guard.path_calls),1)
    def test_original_own_history_and_activation_unchanged_all_stages(self):
        d=Dynamics();p=self.p.copy();v=self.v.copy();self.run_fixed(d,True,first_initial_iterate=p+5)
        for c in d.calls:
            np.testing.assert_array_equal(c['previous'],p);np.testing.assert_array_equal(c['velocity'],v)
            self.assertEqual(c['activation'],self.activation);self.assertTrue(c['options']['defer_physical_frame_path'])
        np.testing.assert_array_equal(self.p,p);np.testing.assert_array_equal(self.v,v)
    def test_check_selects_upper_report_reactions_velocity_and_restores_guard(self):
        d=Dynamics();q,v,r=self.run_fixed(d,True)
        self.assertEqual(len(d.calls),2);self.assertEqual(d.guard.scale,.04)
        self.assertEqual(r['support_reaction_max_N'],.04)
        self.assertEqual(r['generalized_contact']['selected_scale_marker'],.04)
        self.assertEqual(r['contact_accounting']['barrier_scale'],.04)
        np.testing.assert_allclose(q[:,0],self.p[:,0]+.02)
        np.testing.assert_array_equal(v,(q-self.p)/r['dt_s'])
        self.assertEqual(d.guard.path_calls[0][0],.04);np.testing.assert_array_equal(d.guard.path_calls[0][2],q)
        validate_fixed_frame(r,self.p,self.v,q)
        comparison=r['barrier_qualification']['comparisons'][0]
        self.assertAlmostEqual(comparison['maximum_occupied_site_difference_mm'],.001)
        self.assertAlmostEqual(comparison['dt_scaled_velocity_difference_mm'],.001)
    def test_failed_comparison_promotes_then_selects_first_passing_upper(self):
        d=Dynamics(offsets=(.02,.03,.031,.0311));q,v,r=self.run_fixed(d,True)
        self.assertEqual(len(d.calls),3);self.assertTrue(r['accepted'])
        self.assertEqual(r['contact_accounting']['barrier_scale'],.008)
        self.assertEqual(r['support_reaction_max_N'],.008);self.assertEqual(d.guard.scale,.008)
        self.assertTrue(r['barrier_qualification']['prospective_scale_reduction'])
        np.testing.assert_allclose(q[:,0],self.p[:,0]+.03)
        validate_fixed_frame(r,self.p,self.v,q)
    def test_failed_lower_solve_rolls_back_retains_upper_and_diagnostic(self):
        d=Dynamics(fail_stage=1);kept=[];failed=[]
        q,v,r=self.run_fixed(d,True,sensitivity_candidate_callback=lambda q,p,v,m:kept.append(dict(m,q=q)),
            checkpoint_callback=lambda *a:failed.append(a))
        self.assertFalse(r['accepted']);self.assertEqual(r['status'],'fixed_barrier_sensitivity_inconclusive')
        self.assertEqual(len(kept),1);self.assertEqual(len(failed),1)
        np.testing.assert_array_equal(q,self.p);np.testing.assert_array_equal(v,self.v)
        self.assertEqual(r['barrier_qualification']['committed_physical_time_steps'],0)
        self.assertEqual(len(d.guard.path_calls),0)
    def test_retained_prefix_resumes_without_re_solving_upper(self):
        kept=[];d=Dynamics(fail_stage=1)
        self.run_fixed(d,True,sensitivity_candidate_callback=lambda q,p,v,m:kept.append(dict(m,q=q)))
        d=Dynamics();q,v,r=self.run_fixed(d,True,resumed_candidates=kept)
        self.assertTrue(r['accepted']);self.assertEqual([c['scale'] for c in d.calls],[.008])
        validate_fixed_frame(r,self.p,self.v,q)
    def test_exact_lower_partial_stage_resume_applies_qp_only_at_its_stage(self):
        kept=[];partial=[]
        self.run_fixed(Dynamics(fail_stage=1),True,
            sensitivity_candidate_callback=lambda q,p,v,m:kept.append(dict(m,q=q)),
            checkpoint_callback=lambda q,p,v,m:partial.append(dict(m,q=q)))
        partial[0]['q'][:,0]+=.0002
        partial[0]['failed_inner_report']['history']=[{'epigraph_qp_checkpoint':{'marker':'lower_only'}}]
        d=Dynamics();q,v,row=self.run_fixed(d,True,resumed_candidates=kept,resumed_partial_stage=partial[0])
        self.assertTrue(row['accepted']);self.assertEqual(len(d.calls),1)
        self.assertEqual(d.calls[0]['scale'],.008)
        np.testing.assert_array_equal(d.calls[0]['initial'],partial[0]['q'])
        self.assertEqual(d.calls[0]['qp'],{'marker':'lower_only'})
        bad=copy.deepcopy(partial[0]);bad['barrier_stage']=0
        with self.assertRaises(ValueError):self.run_fixed(Dynamics(),True,resumed_candidates=kept,resumed_partial_stage=bad)
    def test_near_milestone_does_not_trigger_twice(self):
        just_before=.025-5e-15
        self.assertTrue(check_request(self.plan,.024,just_before)['comparison_required'])
        self.assertFalse(check_request(self.plan,just_before,.026)['comparison_required'])
    def test_resume_rejects_different_original_history_or_activation(self):
        kept=[];self.run_fixed(Dynamics(fail_stage=1),True,sensitivity_candidate_callback=lambda q,p,v,m:kept.append(dict(m,q=q)))
        for key,bad in [('q_previous_sha256','wrong'),('activation',{'muscle':.9}),('dt_s',.001)]:
            broken=copy.deepcopy(kept);broken[0][key]=bad
            with self.assertRaises(ValueError):self.run_fixed(Dynamics(),True,resumed_candidates=broken)
    def test_unresolved_stage_budget_never_commits_or_changes_future_level(self):
        kept=[];q,v,r=self.run_fixed(Dynamics(offsets=(.02,.03,.04,.05)),True,sensitivity_stage_limit=2,
            sensitivity_candidate_callback=lambda q,p,v,m:kept.append(dict(m,q=q)))
        self.assertEqual(r['status'],'fixed_barrier_sensitivity_budget_exhausted');self.assertFalse(r['accepted'])
        self.assertEqual(len(kept),2);np.testing.assert_array_equal(q,self.p)
        self.assertIsNone(r['barrier_qualification']['selected_barrier_scale'])
    def test_final_chord_gate_unchanged_and_rolls_back(self):
        d=Dynamics();d.guard.path_valid=False;q,v,r=self.run_fixed(d)
        self.assertFalse(r['accepted']);self.assertEqual(r['status'],'final_physical_path_failed')
        np.testing.assert_array_equal(q,self.p);np.testing.assert_array_equal(v,self.v)
    def test_positive_slack_gate_unchanged(self):
        d=Dynamics();d.guard.slack=0.;q,v,r=self.run_fixed(d)
        self.assertFalse(r['accepted']);self.assertEqual(len(d.guard.path_calls),0)
    def test_consumer_rejects_false_flags_missing_generalized_or_own_history(self):
        q,v,r=self.run_fixed(Dynamics())
        for key,bad in [('lower_mu_comparison_passed',True),('barrier_continuation_qualified',True),
                        ('contact_generalized_stationarity_qualified',False),('final_physical_path_qualified',False)]:
            broken=copy.deepcopy(r);broken[key]=bad
            with self.assertRaises(ValueError):validate_fixed_frame(broken,self.p,self.v,q)
        with self.assertRaises(ValueError):validate_fixed_frame(r,self.p+1,self.v,q)
    def test_retained_identity_binds_contact_material_and_implementations(self):
        report=dict(coarse_contact_provenance={'query_sha256':'original','setup_seconds':1.,
            'rigid_query':{'build_seconds':2.},'initial_site_state':{'site_query_seconds':3.}},
            source_domain_sha256='d',source_phase_sha256='p',source_support_sha256='s',
            source_support_bundle_sha256={},physical_mass_kg=.1,damping_per_second=0.,
            gravity_mm_s2=[0,-9810,0],activation_ramp_s=.1,stationarity_accuracy_policy='same')
        first=retained_model_sha(report,{'material_sha':'original'},{'mechanics.py':'original'})
        timing=copy.deepcopy(report);timing['coarse_contact_provenance']['setup_seconds']=20.
        self.assertEqual(first,retained_model_sha(timing,{'material_sha':'original'},{'mechanics.py':'original'}))
        changed=copy.deepcopy(report);changed['coarse_contact_provenance']['query_sha256']='changed'
        self.assertNotEqual(first,retained_model_sha(changed,{'material_sha':'original'},{'mechanics.py':'original'}))
        self.assertNotEqual(first,retained_model_sha(report,{'material_sha':'changed'},{'mechanics.py':'original'}))
        self.assertNotEqual(first,retained_model_sha(report,{'material_sha':'original'},{'mechanics.py':'changed'}))
    def test_legacy_history_still_valid(self):
        source=np.zeros((2,3));q=source+.01;v=q/.01
        validate_accepted_history(source,[q],[v],[.01],[dict(accepted=True,dt_s=.01,physical_time_s=.01)])
    def test_crossing_schedule_and_adaptive_resume(self):
        request=check_request(self.plan,.024,.027)
        self.assertTrue(request['comparison_required']);self.assertEqual(request['physical_time_s'],.027)
        self.assertFalse(check_request(self.plan,.027,.03)['comparison_required'])
        prior={'fixed_barrier_check_plan':self.plan,'fixed_barrier_check_plan_sha256':plan_sha(self.plan)}
        changed=make_plan(prior,[.032],'explicit_changed_contact_regime',.027)
        self.assertEqual(changed['predecessor_plan_sha256'],plan_sha(self.plan))
        self.assertTrue(check_request(changed,.031,.033)['comparison_required'])
        resumed=make_plan({'fixed_barrier_check_plan':changed,'fixed_barrier_check_plan_sha256':plan_sha(changed)})
        self.assertEqual(changed,resumed)
        with self.assertRaises(ValueError):make_plan(dict(prior,fixed_barrier_check_plan_sha256='bad'))
        with self.assertRaises(ValueError):make_plan(prior,[.01],'past',.027)
    def test_inconclusive_driver_decision_is_not_dt_refinement(self):
        _,_,r=self.run_fixed(Dynamics(fail_stage=1),True)
        report={'runs':{c:dict(reports=[r],accepted_frames=2) for c in ('passive','zygomaticus_frontalis')}}
        self.assertEqual(driver_decision(report,2,.026),'needs_fixed_barrier_sensitivity_resume')
    def test_future_pair_level_is_minimum_selected_not_last_lower_check(self):
        _,_,a=self.run_fixed(Dynamics(offsets=(.02,.03,.031,.0311)),True)
        _,_,b=self.run_fixed(Dynamics(),True)
        report={'runs':{'passive':{'reports':[a]},'zygomaticus_frontalis':{'reports':[b]}}}
        self.assertEqual(prospective_fixed_scale(report),.008)


if __name__=='__main__':unittest.main()
