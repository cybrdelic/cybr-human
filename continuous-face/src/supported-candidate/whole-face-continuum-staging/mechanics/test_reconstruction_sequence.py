import copy
import unittest
from paired_reconstruction_sequence import driver_decision,audit_decision,classification_decision,qualified_dt_policy,CASES
from reconstruction_seed import RECONSTRUCTION_FIELDS

class ReconstructionSequenceTests(unittest.TestCase):
    def test_both_cases_same_time_required(self):
        report={'runs':{c:dict(accepted_frames=3,actual_times_s=[.005,.00625,.0075],reports=[dict(accepted=True)]) for c in CASES}}
        self.assertEqual(driver_decision(report,2,.0075),'audit_candidate_pair')
        report['runs'][CASES[1]]['actual_times_s'][-1]=.007
        self.assertEqual(driver_decision(report,2,.0075),'needs_same_stage_solver_recovery')
    def test_new_failure_needs_real_witness(self):
        path=dict(RECONSTRUCTION_FIELDS,site_chord=dict(offenders=[dict(site=3)]))
        report=dict(runs={'passive':dict(reports=[dict(status='final_physical_path_failed',frame_path=path)])})
        self.assertEqual(driver_decision(report,2,.0075),'needs_finite_reconstruction_witness')
        path['certified_interframe_penetration_cap_mm']=.05
        self.assertEqual(driver_decision(report,2,.0075),'integrity_or_query_blocker')
    def test_external_limit_and_integrity_are_bound(self):
        audit=dict(RECONSTRUCTION_FIELDS,qualified=True,audited_snapshots_unchanged_during_audit=True,contact_sources_unchanged_during_audit=True)
        self.assertEqual(audit_decision(0,audit),'qualified')
        audit['certified_interframe_penetration_cap_mm']=.05
        self.assertEqual(audit_decision(0,audit),'integrity_or_query_blocker')
    def test_old_strict_witness_cannot_trigger_new_limit_refinement(self):
        evidence=dict(RECONSTRUCTION_FIELDS,schema='finite_failed_chord_witness_v1',case='passive',previous_audit_sha256='p',
            candidate_npz_sha256='q',classifier_source_sha256='s',classification='genuine_cap_crossing',
            genuine_cap_crossing=True,previous_q_v_unchanged=True,snapshots_unchanged=True,
            physical_force_law_changed=False,accepted_physical_frame=False,
            witnesses=[dict(closed_owner=True,original_containment_valid=True,independent_oblique_containment_valid=True,
                independent_owner_membership_matches=True,finite_penetration_depth_mm=.054,penetration_cap_mm=.05,
                rounding_pad_mm=1e-10,all_owner_facets_enumerated=14094)])
        args=dict(case='passive',prior_sha='p',candidate_sha='q',classifier_sha='s')
        self.assertEqual(classification_decision(0,evidence,**args),'integrity_or_query_blocker')
        evidence['witnesses'][0]['penetration_cap_mm']=.055
        self.assertEqual(classification_decision(0,evidence,**args),'integrity_or_query_blocker')
        evidence['witnesses'][0]['finite_penetration_depth_mm']=.056
        self.assertEqual(classification_decision(0,evidence,**args),'refine_physical_dt')
    def test_dt_growth_and_real_failure_policy_unchanged(self):
        self.assertEqual(qualified_dt_policy(.00125,2,.005,qualified=True),(.0025,0))
        self.assertEqual(qualified_dt_policy(.00125,2,.005,qualified=False),(.000625,0))

if __name__=='__main__':unittest.main()
