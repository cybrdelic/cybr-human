import copy
import unittest
from reconstruction_020_seed import RECONSTRUCTION_FIELDS
from requalify_020_stationary_endpoint import requalified_row, bind_stationary_metadata

class RequalificationTests(unittest.TestCase):
    def original(self):
        row=dict(status='final_physical_path_failed',accepted=False,barrier_continuation=dict(committed_physical_time_steps=0),
            dynamic_residual_max_N=1.3e-7,stationarity_accuracy=dict(route='full_qp_force_state_fallback'),
            frame_path=dict(feasible=False))
        for k in ('final_inner_stationarity_converged','barrier_continuation_qualified','lower_mu_comparison_passed',
                  'strict_positive_slack','complementarity_bound_satisfied','contact_generalized_stationarity_qualified'):
            row[k]=True
        return row
    def path(self):return dict(RECONSTRUCTION_FIELDS,feasible=True,coarse_whole_segment_checked=True)
    def test_reuses_unchanged_numerical_report_without_mutating_old(self):
        old=self.original();snapshot=copy.deepcopy(old);new=requalified_row(old,self.path())
        self.assertEqual(old,snapshot);self.assertTrue(new['accepted'])
        self.assertEqual(new['dynamic_residual_max_N'],old['dynamic_residual_max_N'])
        self.assertEqual(new['stationarity_accuracy'],old['stationarity_accuracy'])
        self.assertEqual(new['barrier_continuation']['committed_physical_time_steps'],1)
    def test_failed_or_wrong_limit_path_not_admitted(self):
        for update in (dict(feasible=False),dict(certified_interframe_penetration_cap_mm=.055),dict(coarse_whole_segment_checked=False)):
            path=self.path();path.update(update)
            with self.assertRaises(ValueError):requalified_row(self.original(),path)
    def test_nonstationary_or_infeasible_reaction_report_not_reused(self):
        for k in ('final_inner_stationarity_converged','strict_positive_slack','contact_generalized_stationarity_qualified','lower_mu_comparison_passed'):
            row=self.original();row[k]=False
            with self.assertRaises(ValueError):requalified_row(row,self.path())
    def test_actual_stationary_parameters_are_bound(self):
        row=dict(dt_s=.00125,activation={'frontalis':.2},barrier_continuation=dict(accepted_barrier_scale=.008))
        meta=dict(dt_s=.00125,activation={'frontalis':.2},barrier_scale=.008)
        bind_stationary_metadata(row,meta)
        for key,value in [('dt_s',.000625),('activation',{}),('barrier_scale',.04)]:
            bad=dict(meta);bad[key]=value
            with self.assertRaises(ValueError):bind_stationary_metadata(row,bad)
if __name__=='__main__':unittest.main()
