"""Warm starts must not turn barrier continuation into extra physical time."""
import unittest
import numpy as np
from implicit_dynamics import ImplicitDynamics
from test_mechanics import fixture, ObstacleFreeFixture
from barrier_continuation import barrier_step


class WarmStartTests(unittest.TestCase):
    def test_same_inertial_prediction(self):
        s = fixture()
        solver = ImplicitDynamics(s, ObstacleFreeFixture())
        previous = s.X + [2., 1., -3.]
        velocity = np.tile([.3, -.2, .5], (len(s.X), 1))
        dt = .001
        expected = previous + dt * velocity + dt**2 * solver.gravity
        guess = previous + [.001, -.002, .003]
        q, v, report = solver.step(previous, velocity, dt=dt,
            initial_iterate=guess, force_tolerance_N=1e-12)
        self.assertTrue(report['accepted'])
        self.assertTrue(report['warm_start_used'])
        np.testing.assert_allclose(q, expected, atol=1e-10, rtol=0)
        np.testing.assert_allclose(v, (expected-previous)/dt, atol=1e-8, rtol=0)

    def test_failed_warm_start_does_not_commit(self):
        s = fixture()
        solver = ImplicitDynamics(s, ObstacleFreeFixture())
        velocity = np.tile([.3, -.2, .5], (len(s.X), 1))
        q, v, report = solver.step(s.X, velocity, dt=.01,
            initial_iterate=s.X+[0., -.001, 0.], max_iterations=0)
        self.assertFalse(report['accepted'])
        np.testing.assert_array_equal(q, s.X)
        np.testing.assert_array_equal(v, velocity)


class ContinuationTests(unittest.TestCase):
    class Guard:
        def set_barrier_scale(self, scale):
            self.scale = scale
        def max_physical_site_difference(self, a, b):
            return np.linalg.norm(a-b, axis=1).max()
        def path(self, a, b):
            return dict(feasible=True, whole_segment_certified=True)

    class Dynamics:
        def __init__(self, fail_at=None):
            self.guard = ContinuationTests.Guard()
            self.calls = []
            self.deferred_paths = []
            self.fail_at = fail_at
        def path_is_valid(self, evidence):
            return evidence.get('feasible') is True and evidence.get('whole_segment_certified') is True
        def step(self, previous, velocity, *, dt, initial_iterate, **kwargs):
            self.deferred_paths.append(kwargs.get('defer_physical_frame_path'))
            self.calls.append((previous.copy(), velocity.copy(), dt,
                None if initial_iterate is None else initial_iterate.copy()))
            accepted = len(self.calls) != self.fail_at
            q = previous + [0., .02 + .01*self.guard.scale, 0.]
            return q, (q-previous)/dt, dict(accepted=accepted, status='converged' if accepted else 'iteration_limit',
                contact_accounting=dict(minimum_slack_mm=None, active_barrier_owner_terms=0, complementarity_bound_satisfied=True))

    def test_one_commit_after_sensitivity(self):
        d = self.Dynamics()
        previous, velocity = np.zeros((4, 3)), np.ones((4, 3))
        q, v, report = barrier_step(d, previous, velocity, dt=.01, activation={})
        self.assertTrue(report['accepted'])
        self.assertEqual(len(d.calls), 3)
        for p, v0, dt, warm in d.calls:
            np.testing.assert_array_equal(p, previous)
            np.testing.assert_array_equal(v0, velocity)
            self.assertEqual(dt, .01)
        np.testing.assert_allclose(q[:, 1], .0204)
        np.testing.assert_allclose(v, (q-previous)/.01)
        self.assertLess(report['lower_mu_site_difference_mm'], .005)
        self.assertEqual(report['barrier_continuation']['committed_physical_time_steps'], 1)
        self.assertTrue(all(d.deferred_paths))

    def test_only_final_parameter_state_checks_physical_chord(self):
        d = self.Dynamics(); paths = []
        d.guard.path = lambda a, b: (paths.append((a.copy(), b.copy())) or dict(feasible=True, whole_segment_certified=True))
        q, v, report = barrier_step(d, np.zeros((4, 3)), np.zeros((4, 3)), dt=.01, activation={})
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(d.calls), 3)
        self.assertTrue(report['final_physical_path_qualified'])
        np.testing.assert_array_equal(paths[0][1], q)

    def test_failed_final_chord_never_commits_stationary_endpoints(self):
        d = self.Dynamics(); checkpoints = []
        d.guard.path = lambda a, b: dict(feasible=False, whole_segment_certified=False)
        previous, velocity = np.zeros((4, 3)), np.ones((4, 3))
        q, v, report = barrier_step(d, previous, velocity, dt=.01, activation={},
            checkpoint_callback=lambda *args: checkpoints.append(args))
        self.assertFalse(report['accepted'])
        self.assertTrue(report['barrier_continuation_qualified'])
        self.assertEqual(report['status'], 'final_physical_path_failed')
        self.assertEqual(report['barrier_continuation']['committed_physical_time_steps'], 0)
        self.assertTrue(checkpoints[-1][-1]['final_parameter_stationary_but_physical_path_failed'])
        np.testing.assert_array_equal(q, previous)
        np.testing.assert_array_equal(v, velocity)

    def test_failed_lower_mu_rolls_back_every_stage(self):
        d = self.Dynamics(fail_at=2)
        previous, velocity = np.zeros((4, 3)), np.ones((4, 3))
        q, v, report = barrier_step(d, previous, velocity, dt=.01, activation={})
        self.assertFalse(report['accepted'])
        np.testing.assert_array_equal(q, previous)
        np.testing.assert_array_equal(v, velocity)
        self.assertEqual(report['barrier_continuation']['committed_physical_time_steps'], 0)

    def test_operation_checkpoint_preserves_same_mu_and_history(self):
        class BudgetDynamics(self.Dynamics):
            def step(self, previous, velocity, *, dt, initial_iterate, **kwargs):
                q, v, row = super().step(previous, velocity, dt=dt,
                    initial_iterate=initial_iterate, **kwargs)
                if not row['accepted']:
                    row['history'] = [dict(implicit_objective_N_mm=2.), dict(implicit_objective_N_mm=1.)]
                    kwargs['diagnostic_iterate_callback'](q.copy(), row)
                return q, v, row
        d = BudgetDynamics(fail_at=2)
        checkpoints = []
        previous, velocity = np.zeros((4, 3)), np.ones((4, 3))
        q, v, row = barrier_step(d, previous, velocity, dt=.01, activation={},
            max_operation_chunks=2, checkpoint_callback=lambda *data: checkpoints.append(data))
        self.assertTrue(row['accepted'])
        self.assertEqual(len(d.calls), 4)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0][3]['barrier_scale'], .2)
        np.testing.assert_allclose(d.calls[2][3], previous+[0., .022, 0.])
        for p, v0, dt, warm in d.calls:
            np.testing.assert_array_equal(p, previous)
            np.testing.assert_array_equal(v0, velocity)
            self.assertEqual(dt, .01)
        self.assertEqual(row['barrier_continuation']['committed_physical_time_steps'], 1)
        self.assertEqual(row['barrier_continuation']['stages'][1]['operation_chunk_count'], 2)


if __name__ == '__main__':
    unittest.main()
