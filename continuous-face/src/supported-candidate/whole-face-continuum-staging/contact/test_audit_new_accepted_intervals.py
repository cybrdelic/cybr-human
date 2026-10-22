"""Prefix identity and adaptive-history regressions; no native queries or solve."""
from pathlib import Path
import json,unittest
import numpy as np
from audit_new_accepted_intervals import reusable_prefix,validate_timing_and_velocity

HERE=Path(__file__).resolve().parent


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.path=HERE/'temporal-audits/v8-passive-step1.json'
        prior=json.loads(self.path.read_text())
        with np.load(prior['accepted_npz_snapshot_path']) as z:
            self.kw=dict(case='passive',q=z['positions_mm'],v=z['velocities_mm_s'],times=z['physical_times_s'],initial=z['source_initial_positions_mm'])
        self.kw.update(sources=prior['contact_sources_sha256'],physical_query_hashes=prior['physical_query_data_sha256'],
            physical_sources={k:prior[k] for k in ('source_domain_sha256','source_phase_sha256','source_support_sha256')})

    def test_actual_qualified_prefix_reuses_without_queries(self):
        rows,digest=reusable_prefix(self.path,**self.kw)
        self.assertEqual(len(rows),1);self.assertTrue(rows[0]['reused_without_new_queries'])
        self.assertEqual(rows[0]['reused_from_audit_sha256'],digest)

    def test_changed_coordinate_velocity_time_or_case_rejected(self):
        for key in ('q','v','times'):
            with self.subTest(key=key):
                kw=dict(self.kw);kw[key]=kw[key].copy();kw[key].flat[0]+=1e-8
                with self.assertRaises(ValueError):reusable_prefix(self.path,**kw)
        with self.assertRaises(ValueError):reusable_prefix(self.path,**dict(self.kw,case='zygomaticus_frontalis'))

    def test_changed_proof_source_rejected(self):
        sources=dict(self.kw['sources']);sources['volume_guard.py']='changed'
        with self.assertRaises(ValueError):reusable_prefix(self.path,**dict(self.kw,sources=sources))


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.initial=np.zeros((2,3));self.q=np.array([np.full((2,3),.01),np.full((2,3),.03)])
        self.times=np.array([.005,.015]);self.v=np.full_like(self.q,2.)
        self.rows=[dict(dt_s=.005,physical_time_s=.005),dict(dt_s=.01,physical_time_s=.015)]
        self.report=dict(physical_dt_s=.01,physical_dt_s_scope='Newly requested intervals only; prior actual dt is retained per frame')

    def test_adaptive_actual_intervals_and_velocity_pass(self):
        validate_timing_and_velocity(self.report,self.rows,self.q,self.v,self.times,self.initial)

    def test_bad_row_dt_and_velocity_rejected(self):
        rows=[dict(x) for x in self.rows];rows[0]['dt_s']=.01
        with self.assertRaises(ValueError):validate_timing_and_velocity(self.report,rows,self.q,self.v,self.times,self.initial)
        v=self.v.copy();v[1,0,0]+=1e-4
        with self.assertRaises(ValueError):validate_timing_and_velocity(self.report,self.rows,self.q,v,self.times,self.initial)

    def test_legacy_global_dt_still_checked(self):
        with self.assertRaises(ValueError):validate_timing_and_velocity(dict(physical_dt_s=.01),self.rows,self.q,self.v,self.times,self.initial)


if __name__=='__main__':unittest.main()
