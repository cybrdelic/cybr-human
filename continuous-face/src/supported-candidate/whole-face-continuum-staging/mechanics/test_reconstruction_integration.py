import json
from pathlib import Path
import unittest
import numpy as np
from barrier_reconstruction_continuation import barrier_step
from test_barrier_continuation import ContinuationTests
from reconstruction_seed import audited_passive_endpoints

class ReconstructionIntegrationTests(unittest.TestCase):
    def test_final_callback_only_and_own_history(self):
        d=ContinuationTests.Dynamics();old_path=d.guard.path; calls=[]
        previous=np.zeros((4,3));velocity=np.ones((4,3))
        def final(a,b):
            calls.append((a.copy(),b.copy()));return dict(feasible=True,whole_segment_certified=True)
        q,v,row=barrier_step(d,previous,velocity,dt=.01,activation={},final_physical_path=final)
        self.assertTrue(row['accepted']);self.assertEqual(len(calls),1)
        self.assertEqual(d.guard.path,old_path)
        for a,b,dt,warm in d.calls:
            np.testing.assert_array_equal(a,previous);np.testing.assert_array_equal(b,velocity)
        np.testing.assert_allclose(v,(q-previous)/.01)
    def test_bad_reconstruction_rolls_back(self):
        d=ContinuationTests.Dynamics();previous=np.zeros((4,3));velocity=np.ones((4,3))
        q,v,row=barrier_step(d,previous,velocity,dt=.01,activation={},
            final_physical_path=lambda a,b:dict(feasible=False,whole_segment_certified=False))
        self.assertFalse(row['accepted']);np.testing.assert_array_equal(q,previous);np.testing.assert_array_equal(v,velocity)
    def test_actual_frozen_passive_seed_only_copies_q(self):
        p=Path(__file__).parent.parent/'contact/temporal-audits/reconstruction-001-passive.json'
        audit=json.loads(p.read_text());prior=json.loads(Path(audit['mechanics_report_path']).read_text())
        with np.load(audit['accepted_npz_snapshot_path']) as z: X=z['source_initial_positions_mm'].copy()
        seeds,evidence=audited_passive_endpoints(p,prior,X)
        self.assertEqual(list(seeds),[.005,.00625,.007500000000000001]);self.assertIn('q-only',evidence['use'])
        wrong=dict(prior,physical_mass_kg=2)
        with self.assertRaises(ValueError):audited_passive_endpoints(p,wrong,X)

if __name__=='__main__':unittest.main()
