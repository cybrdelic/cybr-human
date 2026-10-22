import unittest
from pathlib import Path
from coarse_physical_site_contact import sha
from classify_failed_site_chord import finite_cap_witnesses
from classify_failed_reconstruction_020_chord import reconstruction_offenders
from test_reconstruction_020_allowance import slab


class ReconstructionWitnessTests(unittest.TestCase):
    def test_witness_must_exceed_reconstruction_not_endpoint_limit(self):
        for depth,expected in [(.053,False),(.054,False),(.07,False),(.071,True),(.077,True),(.105,True)]:
            result=finite_cap_witnesses(slab(depth).rigid,[[.13,.07,0.]],[1],[.5],.07)
            self.assertEqual(result['genuine_cap_crossing'],expected,depth)

    def test_exact_contract_and_offender_schema(self):
        chord=dict(reconstruction_contract='fixed_0p02mm_interframe_allowance_v1',reconstruction_contract_schema='fixed_reconstruction_accuracy_v2',physical_endpoint_penetration_cap_mm=.05,interframe_reconstruction_allowance_mm=.02,
            certified_interframe_penetration_cap_mm=.07,penetration_cap_mm=.07,
            source_sha256=sha(Path(__file__).with_name('continuous_closed_site_reconstruction_020.py')),
            offenders=[dict(site=1,alpha=.5,point_mm=[0,0,0])])
        cap,offenders=reconstruction_offenders(chord)
        self.assertEqual(cap,.07);self.assertEqual(offenders,chord['offenders'])
        for key in ('reconstruction_contract','reconstruction_contract_schema','physical_endpoint_penetration_cap_mm','interframe_reconstruction_allowance_mm','certified_interframe_penetration_cap_mm','penetration_cap_mm','source_sha256'):
            changed=dict(chord);changed[key]='changed'
            with self.assertRaises(ValueError):reconstruction_offenders(changed)
        self.assertEqual(reconstruction_offenders(dict(chord,offenders=[]))[1],[])


    def test_old_055_contract_cannot_trigger_new_model_refinement(self):
        chord=dict(physical_endpoint_penetration_cap_mm=.05,interframe_reconstruction_allowance_mm=.005,
            certified_interframe_penetration_cap_mm=.055,penetration_cap_mm=.055,
            source_sha256=sha(Path(__file__).resolve().parent.parent/'continuous_closed_site_reconstruction.py'),
            offenders=[dict(site=1,alpha=.5,point_mm=[.13,.07,0.])])
        with self.assertRaises(ValueError):reconstruction_offenders(chord)
        result=finite_cap_witnesses(slab(.06).rigid,[[.13,.07,0.]],[1],[.5],.07)
        self.assertFalse(result['genuine_cap_crossing'])

    def test_missing_ambiguous_nonfinite_witness_not_genuine(self):
        import numpy as np
        native=slab(.077).rigid
        r=finite_cap_witnesses(native,np.empty((0,3)),[],[],.07)
        self.assertFalse(r['genuine_cap_crossing']);self.assertEqual(r['classification'],'inconclusive')
        old=native.containment
        def ambiguous(points):
            inside,valid=old(points);valid[:]=False;return inside,valid
        native.containment=ambiguous
        self.assertFalse(finite_cap_witnesses(native,[[.13,.07,0.]],[1],[.5],.07)['genuine_cap_crossing'])
        with self.assertRaises(ValueError):finite_cap_witnesses(native,[[np.nan,0,0]],[1],[.5],.07)
        with self.assertRaises(ValueError):finite_cap_witnesses(native,[[.13,.07,0.]],[1],[np.nan],.07)


if __name__=='__main__':unittest.main()
