import unittest
import numpy as np
from accepted_history import matched_passive_numerical_guess, inertia_scaled_numerical_guess
from implicit_dynamics import ImplicitDynamics
from test_mechanics import fixture, ObstacleFreeFixture


class NumericalGuessTests(unittest.TestCase):
    def test_shorter_dt_scales_acceleration_not_previous_velocity(self):
        q=np.array([[1.,2.,3.]]);v=np.array([[10.,20.,30.]]);old_dt=.005;new_dt=.0025
        acceleration_correction=np.array([[.1,-.2,.3]])
        failed=q+old_dt*v+acceleration_correction
        guess=inertia_scaled_numerical_guess(q,v,failed,old_dt,new_dt)
        np.testing.assert_allclose(guess,q+new_dt*v+.25*acceleration_correction)
        self.assertFalse(np.shares_memory(guess,q));self.assertFalse(np.shares_memory(guess,v))
        with self.assertRaises(ValueError):inertia_scaled_numerical_guess(q,v,failed,old_dt,old_dt)

    def test_later_passive_guess_does_not_replace_active_q_velocity_or_prediction(self):
        s=fixture();guard=ObstacleFreeFixture();solver=ImplicitDynamics(s,guard)
        previous=s.X+[2.,1.,-3.];velocity=np.tile([.3,-.2,.5],(len(s.X),1));dt=.001
        p0,v0=previous.copy(),velocity.copy()
        passive=previous+[.001,-.002,.003]
        guess=matched_passive_numerical_guess('zygomaticus_frontalis',.01,{.01:passive},previous,guard,None)
        self.assertIsNotNone(guess);self.assertFalse(np.shares_memory(guess,passive))
        q,v,row=solver.step(previous,velocity,dt=dt,initial_iterate=guess,force_tolerance_N=1e-12)
        self.assertTrue(row['accepted'])
        expected=previous+dt*velocity+dt**2*solver.gravity
        np.testing.assert_allclose(q,expected,rtol=0,atol=1e-10)
        np.testing.assert_allclose(v,(expected-previous)/dt,rtol=0,atol=1e-8)
        np.testing.assert_array_equal(previous,p0);np.testing.assert_array_equal(velocity,v0)

    def test_unmatched_or_infeasible_guess_falls_back_to_own_history(self):
        class Guard:
            def endpoint(self,q):return dict(feasible=False)
        q=np.zeros((4,3));guard=Guard()
        self.assertIsNone(matched_passive_numerical_guess('passive',.01,{.01:q},q,guard,None))
        self.assertIsNone(matched_passive_numerical_guess('zygomaticus_frontalis',.01,{.005:q},q,guard,None))
        self.assertIsNone(matched_passive_numerical_guess('zygomaticus_frontalis',.01,{.01:q},q,guard,None))


if __name__=='__main__':unittest.main()
