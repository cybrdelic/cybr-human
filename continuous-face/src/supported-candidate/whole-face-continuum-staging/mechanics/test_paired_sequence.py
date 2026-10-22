import copy
import unittest
from paired_sequence import driver_decision, audit_decision, CASES


class PairedSequenceTests(unittest.TestCase):
    def test_both_cases_are_required_at_the_same_time(self):
        r={'runs':{c:dict(accepted_frames=2,actual_times_s=[.005,.015],reports=[dict(accepted=True)]) for c in CASES}}
        self.assertEqual(driver_decision(r,1,.015),'audit_candidate_pair')
        wrong=copy.deepcopy(r);wrong['runs'][CASES[1]]['actual_times_s'][-1]=.01
        self.assertEqual(driver_decision(wrong,1,.015),'needs_same_stage_solver_recovery')

    def test_only_physical_chord_failure_requests_new_dt(self):
        row=dict(accepted=False,status='final_physical_path_failed',frame_path=dict(samples=[dict(
            invalid_finite_queries=0,minimum_penetration_slack_mm=-.01,minimum_queried_finite_gap_mm=-.06)]))
        r={'runs':{CASES[0]:dict(reports=[row])}}
        self.assertEqual(driver_decision(r,1,.01),'refine_physical_dt')
        row['frame_path']['samples'][0]['invalid_finite_queries']=1
        self.assertEqual(driver_decision(r,1,.01),'integrity_or_query_blocker')
        row['frame_path']['samples']=[]
        self.assertEqual(driver_decision(r,1,.01),'integrity_or_query_blocker')
        r['runs'][CASES[0]]['reports'][-1]['status']='barrier_stage_failed'
        self.assertEqual(driver_decision(r,1,.01),'needs_same_stage_solver_recovery')

    def test_actual_crossing_and_invalid_query_are_distinguished(self):
        a=dict(original_source_unchanged_during_audit=True,audited_snapshots_unchanged_during_audit=True,
            contact_sources_unchanged_during_audit=True,qualified=False,
            intervals=[dict(qualified=False,site_chord=dict(offenders=[dict(native_query_valid=True,
                clipped_query_gap_mm=-.06,penetration_limit_exceeded=True)]))])
        self.assertEqual(audit_decision(2,a),'refine_physical_dt')
        a['intervals'][0]['site_chord']['offenders'][0]['native_query_valid']=False
        self.assertEqual(audit_decision(2,a),'integrity_or_query_blocker')

    def test_passing_exit_alone_cannot_commit_or_hide_changed_sources(self):
        self.assertEqual(audit_decision(0,None),'integrity_or_query_blocker')
        a=dict(qualified=True,original_source_unchanged_during_audit=True,
            audited_snapshots_unchanged_during_audit=True,contact_sources_unchanged_during_audit=True)
        self.assertEqual(audit_decision(0,a),'qualified')
        a['original_source_unchanged_during_audit']=False
        self.assertEqual(audit_decision(0,a),'qualified')
        a['contact_sources_unchanged_during_audit']=False
        self.assertEqual(audit_decision(0,a),'integrity_or_query_blocker')


if __name__=='__main__':
    unittest.main()

class WitnessDecisionTests(unittest.TestCase):
    def test_independent_witness_does_not_hide_integrity_or_unknown_query(self):
        from paired_sequence import classification_decision
        evidence=dict(schema='finite_failed_chord_witness_v1',case='passive',previous_audit_sha256='p',
            candidate_npz_sha256='q',classifier_source_sha256='s',classification='genuine_cap_crossing',
            genuine_cap_crossing=True,previous_q_v_unchanged=True,snapshots_unchanged=True,
            physical_force_law_changed=False,accepted_physical_frame=False,
            witnesses=[dict(closed_owner=True,original_containment_valid=True,independent_oblique_containment_valid=True,
                independent_owner_membership_matches=True,finite_penetration_depth_mm=.053,penetration_cap_mm=.05,
                rounding_pad_mm=1e-10,all_owner_facets_enumerated=14094)])
        args=dict(case='passive',prior_sha='p',candidate_sha='q',classifier_sha='s')
        self.assertEqual(classification_decision(0,evidence,**args),'refine_physical_dt')
        evidence['previous_q_v_unchanged']=False
        self.assertEqual(classification_decision(0,evidence,**args),'integrity_or_query_blocker')
        evidence['previous_q_v_unchanged']=True;evidence['witnesses'][0]['independent_oblique_containment_valid']=False
        self.assertEqual(classification_decision(0,evidence,**args),'integrity_or_query_blocker')

class TimePolicyTests(unittest.TestCase):
    def test_growth_requires_three_qualified_pairs_and_failure_resets(self):
        from paired_sequence import qualified_dt_policy
        dt,streak=.00125,0
        for expected_streak in (1,2):
            dt,streak=qualified_dt_policy(dt,streak,.005,qualified=True)
            self.assertEqual(dt,.00125);self.assertEqual(streak,expected_streak)
        dt,streak=qualified_dt_policy(dt,streak,.005,qualified=True)
        self.assertEqual((dt,streak),(.0025,0))
        self.assertEqual(qualified_dt_policy(dt,2,.005,qualified=False),(.00125,0))
        self.assertEqual(qualified_dt_policy(.005,2,.005,qualified=True),(.005,0))
