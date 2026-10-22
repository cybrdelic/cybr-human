"""Schema-only synthetic fixtures. Never solved, exported or rendered as physics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from accepted_frames import validate_generalized_stationarity
from stationarity_accuracy import validate_stationarity_accuracy
from fixed_barrier_policy import make_plan, check_request, array_sha, stage_scale, MODEL, plan_sha
from fixed_policy_reader import (POLICY, POLICY_SHA, WRAPPER_SHA, FIXED_SOURCES,
    validate_fixed_row, validate_fixed_sources, sensitivity_summary)
from literal_renderer import sha_file


def fixture(checked=False, promoted=False):
    previous=np.arange(12,dtype=float).reshape(4,3)
    velocity=np.ones_like(previous)
    candidate=previous+.001
    end=.026 if checked else .01625
    start=.024 if checked else .015625
    dt=end-start
    winner=dict(contact_generalized_stationarity_qualified=True,
        exterior_contact_stationarity='exact_selected_winner_already_stationary',
        residual_projection_scope='source_origin_support_reactions_only',
        generalized_contact=dict(kind='exact_selected_winner_already_stationary',
            energy_law_unchanged=True,actual_comaximality_satisfied=True,branch_weights_nonnegative=True,
            seam_rows_are_source_support_pins=False,group_weight_sum_error=0.,comaximality_tolerance_N_mm=1e-12,
            maximum_used_branch_energy_gap_N_mm=0.,max_branch_complementarity_N_mm=0.,
            sum_branch_complementarity_N_mm=0.,mixed_groups=[]))
    stages=[]
    count=(3 if promoted else 2) if checked else 1
    for i in range(count):
        scale=stage_scale(.04,i)
        acc=dict(complementarity_bound_satisfied=True,barrier_scale=scale,penalty_energy_N_mm=0.,
            barrier_energy_N_mm=0.,max_lambda_s_N_mm=0.,sum_lambda_s_N_mm=0.,maximum_active_mu_N_mm=0.,
            minimum_slack_mm=.01,per_owner=[],quadratic_contact_force_net_N=[float(i),0.,0.])
        stages.append(dict(copy.deepcopy(winner),not_committed_physical_frame=True,
            inner_stationarity_converged=True,physical_frame_path_deferred=True,final_accounting_qualified=True,
            acceptance_scope='internal_same_timestep_stationarity_only',dynamic_residual_includes_barrier_reaction=True,
            barrier_scale=scale,contact_accounting=acc,dynamic_residual_max_N=(i+1)*1e-9,force_tolerance_N=1e-7,
            dt_s=dt,activation={}))
    selected=count-2 if checked else 0
    comparisons=[]
    for i in range(count-1):
        difference=.002 if i==count-2 else .006
        comparisons.append(dict(upper_scale=stages[i]['barrier_scale'],lower_scale=stages[i+1]['barrier_scale'],
            tolerance_mm=.005,maximum_occupied_site_difference_mm=difference,
            dt_scaled_velocity_difference_mm=difference,passed=i==count-2))
    row=copy.deepcopy(stages[selected])
    row.update(barrier_policy=POLICY,fixed_barrier_qualified=True,barrier_continuation_qualified=False,
        final_inner_stationarity_converged=True,final_physical_path_qualified=True,strict_positive_slack=True,
        complementarity_bound_satisfied=True,physical_frame_path_deferred=False,
        acceptance_scope='physical_frame_with_final_parameter_source_to_endpoint_path',physical_time_s=end,
        lower_mu_comparison_passed=checked,lower_mu_site_difference_mm=.002 if checked else None,
        lower_mu_site_tolerance_mm=.005)
    plan=make_plan()
    row['barrier_qualification']=dict(policy=POLICY,qualified=True,same_physical_time_step=True,
        physical_history_unchanged_during_stages=True,committed_physical_time_steps=1,
        starting_fixed_scale=.04,selected_barrier_scale=stages[selected]['barrier_scale'],selected_stage_index=selected,
        stages=stages,comparisons=comparisons,comparison_performed=checked,comparison_required=checked,
        comparison_time_s=end if checked else None,check_request=check_request(plan,start,end),
        q_previous_sha256=array_sha(previous),velocity_previous_sha256=array_sha(velocity),
        inertial_prediction_sha256=array_sha(previous+dt*velocity),selected_q_sha256=array_sha(candidate),activation={})
    report=dict(fixed_barrier_check_plan=plan,activation_ramp_s=.1)
    return row,report,previous,velocity,candidate,start


def validate(data):
    return validate_fixed_row(*data,'passive',validate_generalized_stationarity,validate_stationarity_accuracy)


class FixedReaderTests(unittest.TestCase):
    def test_ordinary_frame_has_no_fresh_sensitivity(self):
        row=fixture(); validate(row)
        summary=sensitivity_summary([row[0]])
        self.assertEqual(summary['fixed_empirically_checked_frame_indices'],[])
        self.assertEqual(summary['fixed_ordinary_frame_indices_without_fresh_comparison'],[0])

    def test_checked_selects_first_passing_upper_not_lower(self):
        for promoted in (False,True):
            data=fixture(True,promoted); q=validate(data)
            self.assertEqual(q['selected_stage_index'],int(promoted))
            self.assertEqual(q['check_request']['crossed_checks'],[dict(target_time_s=.025,reason='declared_milestone')])
            self.assertEqual(sensitivity_summary([data[0]])['fixed_empirically_checked_physical_times_s'],[.026])

    def test_rejects_fabricated_legacy_flags(self):
        for field,value in [('lower_mu_comparison_passed',True),('lower_mu_site_difference_mm',.001),
                            ('barrier_continuation_qualified',True),('barrier_continuation',{})]:
            data=fixture();data[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate(data)

    def test_rejects_unknown_policy(self):
        data=fixture();data[0]['barrier_policy']='unknown'
        with self.assertRaises(ValueError):validate(data)

    def test_rejects_all_four_own_history_hash_changes(self):
        for key in ('q_previous_sha256','velocity_previous_sha256','inertial_prediction_sha256','selected_q_sha256'):
            data=fixture();data[0]['barrier_qualification'][key]='f'*64
            with self.subTest(key=key),self.assertRaises(ValueError):validate(data)

    def test_rejects_wrong_schedule_actual_time_and_activation(self):
        for change in ('crossed','plan','time','activation','comparison_time','required'):
            data=fixture(True); q=data[0]['barrier_qualification']
            if change=='crossed':q['check_request']['crossed_checks']=[]
            if change=='plan':q['check_request']['plan']['checks'].pop(0)
            if change=='time':q['check_request']['previous_physical_time_s']-=.001
            if change=='activation':q['activation']={'frontalis':.3}
            if change=='comparison_time':q['comparison_time_s']=.025
            if change=='required':q['comparison_required']=False
            with self.subTest(change=change),self.assertRaises(ValueError):validate(data)

    def test_rejects_skipped_first_passing_upper_and_selected_lower(self):
        for change in ('index','earlier_pass','final_fail','nonfinite'):
            data=fixture(True,True);q=data[0]['barrier_qualification']
            if change=='index':q['selected_stage_index']=2
            if change=='earlier_pass':q['comparisons'][0].update(maximum_occupied_site_difference_mm=.001,dt_scaled_velocity_difference_mm=.001,passed=True)
            if change=='final_fail':q['comparisons'][-1]['maximum_occupied_site_difference_mm']=.006
            if change=='nonfinite':q['comparisons'][-1]['dt_scaled_velocity_difference_mm']=float('nan')
            with self.subTest(change=change),self.assertRaises(ValueError):validate(data)

    def test_every_stage_requires_generalized_reactions_and_accuracy(self):
        for index in (0,1):
            for change in ('seam','missing_qp','residual','accounting','nan_accounting','scale'):
                data=fixture(True);stage=data[0]['barrier_qualification']['stages'][index]
                if change=='seam':stage['residual_projection_scope']='seam_projection'
                if change=='missing_qp':stage.pop('generalized_contact')
                if change=='residual':stage['dynamic_residual_max_N']=1e-4
                if change=='accounting':stage['contact_accounting'].pop('per_owner')
                if change=='nan_accounting':stage['contact_accounting']['quadratic_contact_force_net_N'][0]=float('nan')
                if change=='scale':stage['contact_accounting']['barrier_scale']=.0123
                with self.subTest(index=index,change=change),self.assertRaises(ValueError):validate(data)

    def test_selected_upper_stationarity_and_reactions_cannot_be_lower(self):
        for field in ('dynamic_residual_max_N','quadratic_contact_force_net_N'):
            data=fixture(True);row=data[0];lower=row['barrier_qualification']['stages'][1]
            if field=='dynamic_residual_max_N':row[field]=lower[field]
            else:row['contact_accounting'][field]=lower['contact_accounting'][field]
            with self.subTest(field=field),self.assertRaises(ValueError):validate(data)

    def test_rejects_missing_commit_and_nonphysical_stage(self):
        for change in ('commit','stage_count','stage_commit','stationarity','path'):
            data=fixture();q=data[0]['barrier_qualification']
            if change=='commit':q['committed_physical_time_steps']=0
            if change=='stage_count':q['stages']=[]
            if change=='stage_commit':q['stages'][0]['not_committed_physical_frame']=False
            if change=='stationarity':q['stages'][0]['inner_stationarity_converged']=False
            if change=='path':data[0]['physical_frame_path_deferred']=True
            with self.subTest(change=change),self.assertRaises(ValueError):validate(data)

    def test_storage_cursor_bundle_and_full_fixed_frame(self):
        from fixed_policy_reader import DRIVER,CONTROLLER,RAW_HELPER,RAW_HELPER_SHA
        from renderer_report_binding import STORAGE_AUDITOR_SHA,STORAGE_HELPER_SHA
        row,report,previous,velocity,candidate,start=fixture()
        with tempfile.TemporaryDirectory() as td:
            plan=report['fixed_barrier_check_plan'];p=Path(td)/'fixed-barrier-check-plan.json';p.write_text(json.dumps(plan))
            report.update(barrier_policy=POLICY,fixed_barrier_model=MODEL,fixed_barrier_policy_source_sha256=POLICY_SHA,
                barrier_qualification_source_sha256=WRAPPER_SHA,fixed_barrier_check_plan_sha256=plan_sha(plan),
                fixed_barrier_check_plan_path=str(p),fixed_barrier_check_plan_file_sha256=sha_file(p),
                allow_cap_only_gap_search_enabled=True,raw_predictor_geometry_source_sha256=RAW_HELPER_SHA)
            sources={**FIXED_SOURCES,str(p):sha_file(p),DRIVER:'60d90718424d1e5e3acdd8e1238a8b0e6d88ad7b4ba7e432843e743ab6c5427a',
                CONTROLLER:'dabad12cb8ccd14f93bae7d499619bcf044963a04832c36b0ae0249026cff2bb',RAW_HELPER:RAW_HELPER_SHA,
                'contact/reconstruction-020-v1/audit_reconstruction_020_intervals.py':STORAGE_AUDITOR_SHA,
                'mechanics/report_storage.py':STORAGE_HELPER_SHA}
            cursor=dict(document=dict(source_sha256=sources))
            identity=validate_fixed_sources(report,cursor)
            self.assertEqual(identity['report_storage_source_revision'],'lossless_json_gzip_v1_v12')
            self.assertEqual(identity['controller_source_manifest_bindings']['mechanics/report_storage.py'],STORAGE_HELPER_SHA)
            validate((row,report,previous,velocity,candidate,start))
            v14=copy.deepcopy(cursor)
            v14['document']['source_sha256'][CONTROLLER]='728fc856b3b61ab22b3209440a9ec04919fcb636dd3fa26991c9bf1cb55d7b28'
            self.assertEqual(validate_fixed_sources(report,v14)['controller_entrypoint_source_revision'],
                'cap_only_geometry_search_600s_runtime_v14')
            old_auditor_v14=copy.deepcopy(v14)
            old_auditor_v14['document']['source_sha256']['contact/reconstruction-020-v1/audit_reconstruction_020_intervals.py']=FIXED_SOURCES['contact/reconstruction-020-v1/audit_reconstruction_020_intervals.py']
            with self.assertRaises(ValueError):validate_fixed_sources(report,old_auditor_v14)
            v14['document']['source_sha256'][DRIVER]=FIXED_SOURCES[DRIVER]
            with self.assertRaises(ValueError):validate_fixed_sources(report,v14)
            for change in ('missing_helper','wrong_helper','mixed_entrypoints','unknown_auditor'):
                bad=copy.deepcopy(cursor);bs=bad['document']['source_sha256']
                if change=='missing_helper':del bs['mechanics/report_storage.py']
                if change=='wrong_helper':bs['mechanics/report_storage.py']='0'*64
                if change=='mixed_entrypoints':bs[CONTROLLER]='73c4b211fde52cff0fa60874e869d10ed9157b704cc8ac1521b2207e6c8ec847'
                if change=='unknown_auditor':bs['contact/reconstruction-020-v1/audit_reconstruction_020_intervals.py']='0'*64
                with self.subTest(change=change),self.assertRaises(ValueError):validate_fixed_sources(report,bad)

    def test_report_sources_plan_and_paired_manifest_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'plan.json';plan=make_plan();p.write_text(json.dumps(plan))
            report=dict(barrier_policy=POLICY,fixed_barrier_model=MODEL,fixed_barrier_policy_source_sha256=POLICY_SHA,
                barrier_qualification_source_sha256=WRAPPER_SHA,fixed_barrier_check_plan=plan,
                fixed_barrier_check_plan_sha256=plan_sha(plan),fixed_barrier_check_plan_path=str(p),
                fixed_barrier_check_plan_file_sha256=sha_file(p))
            cursor=dict(document=dict(source_sha256={**FIXED_SOURCES,str(p):sha_file(p)}))
            validate_fixed_sources(report,cursor)
            for change in ('policy','wrapper','model','manifest','plan_manifest','plan_hash','missing_cursor'):
                r,c=copy.deepcopy(report),copy.deepcopy(cursor)
                if change=='policy':r['fixed_barrier_policy_source_sha256']='f'*64
                if change=='wrapper':r['barrier_qualification_source_sha256']='f'*64
                if change=='model':r['fixed_barrier_model']['penetration_limit_mm']=.06
                if change=='manifest':c['document']['source_sha256']['mechanics/fixed_barrier_policy.py']='f'*64
                if change=='plan_manifest':c['document']['source_sha256'][str(p)]='f'*64
                if change=='plan_hash':r['fixed_barrier_check_plan_sha256']='f'*64
                if change=='missing_cursor':c=None
                with self.subTest(change=change),self.assertRaises(ValueError):validate_fixed_sources(r,c)
            p.write_text(json.dumps(dict(plan,initial_fixed_scale=.2)))
            with self.assertRaises(ValueError):validate_fixed_sources(report,cursor)

    def test_fixed_external_auditor_is_exactly_bound_without_loosening_geometry(self):
        from test_renderer import external_fixture
        from external_path_audit import (validate_closed_path_audit,CONTACT_SOURCE_FIELDS,
            RECONSTRUCTION_020_CONTRACT,RECONSTRUCTION_020_LIMITS,RECONSTRUCTION_020_PROOF_SOURCES,NEW_INTERVAL_DT_SCOPE)
        from fixed_policy_reader import AUDITOR_SHA
        q0=np.arange(12,dtype=float).reshape(4,3)
        frames=dict(source_initial_positions_mm=q0,positions_mm=np.array([q0+.001,q0+.003]),
            velocities_mm_s=np.ones((2,4,3)),physical_times_s=np.array([.001,.003]))
        policy=dict(reconstruction_contract=RECONSTRUCTION_020_CONTRACT,**RECONSTRUCTION_020_LIMITS,
            reconstruction_contract_schema='fixed_reconstruction_accuracy_v2',trajectory_error_bound_proven=False,
            reconstruction_allowance_scope='Fixed geometric approximation, not a total error bound')
        report=dict(coarse_contact_provenance={key:'a'*64 for key in CONTACT_SOURCE_FIELDS.values()},
            source_domain_sha256='b'*64,source_phase_sha256='c'*64,source_support_sha256='d'*64,
            site_resolution_mm=1.,physical_dt_s=.002,physical_dt_s_scope=NEW_INTERVAL_DT_SCOPE,**policy,
            runs={'fixture':{'reports':[dict(accepted=True,dt_s=dt,endpoint={'maximum_site_penetration_mm':.049}) for dt in (.001,.002)]}})
        report['runs']['fixture']['reports'][1].update(policy,barrier_policy=POLICY)
        for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):
            report['coarse_contact_provenance'][key]='f'*64
        proof=external_fixture(frames,'fixture','1'*64,'2'*64,report)
        proof['contact_sources_sha256'].update(RECONSTRUCTION_020_PROOF_SOURCES)
        proof['contact_sources_sha256']['audit_reconstruction_020_intervals.py']=AUDITOR_SHA
        proof.update(policy,reused_prefix_intervals=1,reused_prefix_audit_sha256='3'*64,
            new_intervals_audited=1,physical_interval_dt_s=[.001,.002],
            fixed_barrier_policy_validation_source_sha256=POLICY_SHA,fixed_barrier_policy_validation_source_unchanged=True)
        proof['intervals'][0].update(reused_without_new_queries=True,reused_from_audit_sha256='3'*64)
        proof['intervals'][1]['site_chord'].update(policy,penetration_cap_mm=.07,
            source_sha256=RECONSTRUCTION_020_PROOF_SOURCES['continuous_closed_site_reconstruction_020.py'],physical_contact_law_changed=False)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic-receipt.json'
            def check(value,rep=report):
                path.write_text(json.dumps(value))
                return validate_closed_path_audit(path,frames,'fixture','1'*64,'2'*64,rep)
            check(proof)
            from renderer_report_binding import STORAGE_AUDITOR_SHA,STORAGE_HELPER_SHA
            storage_proof=copy.deepcopy(proof)
            storage_proof['contact_sources_sha256']['audit_reconstruction_020_intervals.py']=STORAGE_AUDITOR_SHA
            storage_proof.update(report_storage_source_sha256=STORAGE_HELPER_SHA,report_storage_source_unchanged=True)
            check(storage_proof)
            for key,value in (('report_storage_source_sha256','0'*64),('report_storage_source_unchanged',False)):
                bad=copy.deepcopy(storage_proof);bad[key]=value
                with self.subTest(storage_field=key),self.assertRaises(ValueError):check(bad)
            for change in ('policy_sha','source_changed','old_auditor','geometry_helper','source_q','endpoint','positive_J'):
                p,r=copy.deepcopy(proof),copy.deepcopy(report)
                if change=='policy_sha':p['fixed_barrier_policy_validation_source_sha256']='f'*64
                if change=='source_changed':p['fixed_barrier_policy_validation_source_unchanged']=False
                if change=='old_auditor':p['contact_sources_sha256']['audit_reconstruction_020_intervals.py']=RECONSTRUCTION_020_PROOF_SOURCES['audit_reconstruction_020_intervals.py']
                if change=='geometry_helper':p['contact_sources_sha256']['continuous_closed_site_reconstruction_020.py']='f'*64
                if change=='source_q':p['intervals'][1]['q0_sha256']='f'*64
                if change=='endpoint':r['runs']['fixture']['reports'][1]['endpoint']['maximum_site_penetration_mm']=.051
                if change=='positive_J':p['intervals'][1]['positive_J_path']['certified']=False
                with self.subTest(change=change),self.assertRaises(ValueError):check(p,r)

    def test_fixed_case_contrast_does_not_claim_continuation_or_fresh_check(self):
        from measure_latest_pair import numerical_pair_contrast
        row=fixture()[0]
        a=dict(accepted_frame_evidence=[row],barrier_scales=[.04],physical_times_s=[row['physical_time_s']])
        contrast=numerical_pair_contrast(a,copy.deepcopy(a),0,0)
        self.assertIsNone(contrast['both_cases_use_the_same_fixed_continuation_schedule'])
        self.assertTrue(contrast['both_cases_use_the_same_fixed_policy'])
        self.assertFalse(contrast['active_fresh_lower_mu_comparison'])
        self.assertIsNone(contrast['active_lower_mu_sensitivity_mm'])

    def test_controller_restart_plan_alias_requires_identical_bound_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=root/'old'/'fixed-barrier-check-plan.json';alias=root/'new'/'fixed-barrier-check-plan.json'
            original.parent.mkdir();alias.parent.mkdir();plan=make_plan();original.write_text(json.dumps(plan));alias.write_bytes(original.read_bytes())
            report=dict(barrier_policy=POLICY,fixed_barrier_model=MODEL,fixed_barrier_policy_source_sha256=POLICY_SHA,
                barrier_qualification_source_sha256=WRAPPER_SHA,fixed_barrier_check_plan=plan,
                fixed_barrier_check_plan_sha256=plan_sha(plan),fixed_barrier_check_plan_path=str(original),
                fixed_barrier_check_plan_file_sha256=sha_file(original))
            cursor=dict(document=dict(source_sha256={**FIXED_SOURCES,str(alias):sha_file(alias)},
                fixed_barrier_check_plan=plan,fixed_barrier_check_plan_sha256=plan_sha(plan)))
            result=validate_fixed_sources(report,cursor)
            self.assertTrue(result['controller_plan_path_is_exact_byte_alias'])
            self.assertEqual(result['original_report_plan_path'],str(original))
            from fixed_policy_reader import recheck_fixed_plan_bindings
            audit={'fixed_barrier_policy_identity':result}
            recheck_fixed_plan_bindings(audit)
            saved=original.read_bytes();original.write_text('changed after validation')
            with self.assertRaises(RuntimeError):recheck_fixed_plan_bindings(audit)
            original.write_bytes(saved)
            alias.write_text('changed after validation')
            with self.assertRaises(RuntimeError):recheck_fixed_plan_bindings(audit)
            alias.write_bytes(saved)
            recheck_fixed_plan_bindings(audit)
            for change in ('stale_alias','semantic_only','controller_metadata','duplicate'):
                c=copy.deepcopy(cursor);alias.write_bytes(original.read_bytes())
                if change=='stale_alias':alias.write_text('different')
                if change=='semantic_only':alias.write_text(json.dumps(plan,indent=2));c['document']['source_sha256'][str(alias)]=sha_file(alias)
                if change=='controller_metadata':c['document']['fixed_barrier_check_plan_sha256']='f'*64
                if change=='duplicate':c['document']['source_sha256'][str(root/'ambiguous'/'fixed-barrier-check-plan.json')]=sha_file(original)
                with self.subTest(change=change),self.assertRaises(ValueError):validate_fixed_sources(report,c)

    def test_v10_search_optin_entrypoints_must_be_a_reviewed_pair(self):
        from fixed_policy_reader import QUALIFIED_ENTRYPOINT_PAIRS,DRIVER,CONTROLLER,RAW_HELPER,RAW_HELPER_SHA
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'plan.json';plan=make_plan();p.write_text(json.dumps(plan))
            report=dict(barrier_policy=POLICY,fixed_barrier_model=MODEL,fixed_barrier_policy_source_sha256=POLICY_SHA,
                barrier_qualification_source_sha256=WRAPPER_SHA,fixed_barrier_check_plan=plan,
                fixed_barrier_check_plan_sha256=plan_sha(plan),fixed_barrier_check_plan_path=str(p),
                fixed_barrier_check_plan_file_sha256=sha_file(p),allow_cap_only_gap_search_enabled=True,
                raw_predictor_geometry_source_sha256=RAW_HELPER_SHA)
            pair=[pair for pair,name in QUALIFIED_ENTRYPOINT_PAIRS.items() if name.endswith('v10')][0]
            cursor=dict(document=dict(source_sha256={**FIXED_SOURCES,DRIVER:pair[0],CONTROLLER:pair[1],RAW_HELPER:RAW_HELPER_SHA,str(p):sha_file(p)}))
            result=validate_fixed_sources(report,cursor)
            self.assertEqual(result['controller_entrypoint_source_revision'],'cap_only_geometry_search_opt_in_v10')
            v11=copy.deepcopy(cursor)
            v11['document']['source_sha256'][CONTROLLER]='dabad12cb8ccd14f93bae7d499619bcf044963a04832c36b0ae0249026cff2bb'
            self.assertEqual(validate_fixed_sources(report,v11)['controller_entrypoint_source_revision'],
                'cap_only_geometry_search_retry_seed_v11')
            v11['document']['source_sha256'][DRIVER]=FIXED_SOURCES[DRIVER]
            with self.assertRaises(ValueError):validate_fixed_sources(report,v11)
            for change in ('mixed_pair','unknown_pair','helper_manifest','helper_report','flag_type'):
                r,c=copy.deepcopy(report),copy.deepcopy(cursor)
                if change=='mixed_pair':c['document']['source_sha256'][DRIVER]=FIXED_SOURCES[DRIVER]
                if change=='unknown_pair':c['document']['source_sha256'][CONTROLLER]='f'*64
                if change=='helper_manifest':c['document']['source_sha256'][RAW_HELPER]='f'*64
                if change=='helper_report':r['raw_predictor_geometry_source_sha256']='f'*64
                if change=='flag_type':r['allow_cap_only_gap_search_enabled']='true'
                with self.subTest(change=change),self.assertRaises(ValueError):validate_fixed_sources(r,c)


if __name__=='__main__':unittest.main()
