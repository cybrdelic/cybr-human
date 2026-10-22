import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from accepted_history import next_physical_time, validate_accepted_history, validate_resume_interval_audits


class AcceptedHistoryTests(unittest.TestCase):
    def test_changed_dt_uses_actual_cumulative_time(self):
        self.assertAlmostEqual(next_physical_time([.005], .01), .015)
        self.assertAlmostEqual(next_physical_time([.005, .015], .0025), .0175)
        for times in ([.005, .005], [.005, .004], [np.nan], [-.1]):
            with self.assertRaises(ValueError):
                next_physical_time(times, .01)

    def test_each_accepted_interval_keeps_its_own_velocity_and_dt(self):
        x = np.zeros((2, 3)); q = [x+.01, x+.04]
        v = [(q[0]-x)/.005, (q[1]-q[0])/.01]
        rows = [dict(accepted=True, dt_s=dt, physical_time_s=t) for dt, t in ((.005, .005), (.01, .015))]
        validate_accepted_history(x, q, v, [.005, .015], rows)
        wrong = copy.deepcopy(rows); wrong[1]['dt_s'] = .005
        with self.assertRaises(ValueError):
            validate_accepted_history(x, q, v, [.005, .015], wrong)
        with self.assertRaises(ValueError):
            validate_accepted_history(x, q, [v[0], v[0]], [.005, .015], rows)

    def test_external_audit_must_bind_the_exact_accepted_history(self):
        prior = dict(source_domain_sha256='domain', source_phase_sha256='phase', source_support_sha256='support',
            runs={'passive': dict(accepted_frames=1, accepted_npz_sha256='actual-file-hash', actual_times_s=[.005])})
        audit = {k: prior[k] for k in ('source_domain_sha256', 'source_phase_sha256', 'source_support_sha256')}
        audit.update(schema='continuous_closed_site_external_audit_v1', qualified=True, case='passive',
            accepted_npz_sha256='actual-file-hash', physical_times_s=[.005], accepted_frames_audited=1,
            saved_velocity_history_consistent=True, intervals=[dict(qualified=True,
                site_chord=dict(continuous_closed_owner_site_cap_checked=True, feasible=True),
                positive_J_path=dict(certified=True))])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'audit.json'; p.write_text(json.dumps(audit))
            self.assertIn('passive', validate_resume_interval_audits(prior, [p], ['passive']))
            for change in (dict(qualified=False), dict(accepted_npz_sha256='different'), dict(physical_times_s=[.01])):
                p.write_text(json.dumps(dict(audit, **change)))
                with self.assertRaises(ValueError):
                    validate_resume_interval_audits(prior, [p], ['passive'])
            with self.assertRaises(ValueError):
                validate_resume_interval_audits(prior, [], ['passive'])


if __name__ == '__main__':
    unittest.main()
