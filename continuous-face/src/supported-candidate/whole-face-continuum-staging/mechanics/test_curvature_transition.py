import copy
import tempfile
import unittest
from pathlib import Path
from curvature_transition import (validate_curvature_transition, checkpoint_guard_sha,
                                  EPIGRAPH_SHA256, REVISION)
from source_transfer import sha


class CurvatureTransitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'factory.py'
        self.path.write_text('# isolated fixture\n')
        self.old = dict(exterior_branch_provider_sha256=EPIGRAPH_SHA256,
                        penalty_stiffness=.01, nested=dict(cap=.05))
        self.prior = dict(guard_module_sha256=EPIGRAPH_SHA256,
                          coarse_contact_provenance=self.old)
        self.new = dict(self.old, interior_feature_tangent_revision=REVISION,
            interior_feature_tangent_source_sha256=sha(self.path),
            base_epigraph_factory_sha256=EPIGRAPH_SHA256,
            unchanged_energy_gradient_guards=True)

    def transition(self, **kw):
        return validate_curvature_transition(self.prior, self.new, self.path,
                                             EPIGRAPH_SHA256, **kw)

    def test_only_q_guess_crosses_tangent_transition(self):
        t = self.transition()
        self.assertFalse(t['old_qp_state_reused'])
        self.assertEqual(checkpoint_guard_sha({}, t), EPIGRAPH_SHA256)
        with self.assertRaises(ValueError):
            self.transition(old_qp_import_requested=True)

    def test_physical_and_query_parameters_must_match(self):
        self.new['nested'] = dict(cap=.07)
        with self.assertRaises(ValueError): self.transition()

    def test_factory_hash_must_match_saved_identity(self):
        self.path.write_text('# changed source\n')
        with self.assertRaises(ValueError): self.transition()

    def test_only_native_build_walltime_is_nonphysical(self):
        self.old['rigid_query'] = dict(build_seconds=1., source='same')
        self.new['rigid_query'] = dict(build_seconds=2., source='same')
        self.old['initial_site_state'] = dict(site_query_seconds=1., feasible=True)
        self.new['initial_site_state'] = dict(site_query_seconds=2., feasible=True)
        self.old['setup_seconds'] = 1.
        self.new['setup_seconds'] = 2.
        self.assertEqual(self.transition()['current_native_setup_seconds'], 2.)
        same = dict(guard_module_sha256=sha(self.path), coarse_contact_provenance=copy.deepcopy(self.new))
        self.new['rigid_query']['build_seconds'] = 3.
        self.assertIsNone(validate_curvature_transition(same, self.new, self.path))
        self.new['rigid_query']['source'] = 'different'
        with self.assertRaises(ValueError): self.transition()
        self.new['rigid_query']['source'] = 'same'
        self.new['rigid_query']['build_seconds'] = float('nan')
        with self.assertRaises(ValueError): self.transition()

    def test_old_provider_identity_cannot_be_relabelled(self):
        self.new['exterior_branch_provider_sha256'] = sha(self.path)
        with self.assertRaises(ValueError): self.transition()

    def test_later_unchanged_factory_keeps_current_checkpoint_identity(self):
        prior = dict(guard_module_sha256=sha(self.path), coarse_contact_provenance=copy.deepcopy(self.new))
        self.assertIsNone(validate_curvature_transition(prior, self.new, self.path))
        self.assertEqual(checkpoint_guard_sha(prior, None), sha(self.path))
        self.new['penalty_stiffness'] = .02
        with self.assertRaises(ValueError): validate_curvature_transition(prior, self.new, self.path)


if __name__ == '__main__': unittest.main()
