import unittest
from pathlib import Path
from coarse_physical_site_contact import sha
from classify_failed_site_chord import finite_cap_witnesses
from classify_failed_reconstruction_chord import reconstruction_offenders
from test_reconstruction_allowance import slab


class ReconstructionWitnessTests(unittest.TestCase):
    def test_witness_must_exceed_reconstruction_not_endpoint_limit(self):
        for depth,expected in [(.053,False),(.054,False),(.055,False),(.056,True),(.077,True),(.105,True)]:
            result=finite_cap_witnesses(slab(depth).rigid,[[.13,.07,0.]],[1],[.5],.055)
            self.assertEqual(result['genuine_cap_crossing'],expected,depth)

    def test_exact_contract_and_offender_schema(self):
        chord=dict(physical_endpoint_penetration_cap_mm=.05,interframe_reconstruction_allowance_mm=.005,
            certified_interframe_penetration_cap_mm=.055,penetration_cap_mm=.055,
            source_sha256=sha(Path(__file__).with_name('continuous_closed_site_reconstruction.py')),
            offenders=[dict(site=1,alpha=.5,point_mm=[0,0,0])])
        cap,offenders=reconstruction_offenders(chord)
        self.assertEqual(cap,.055);self.assertEqual(offenders,chord['offenders'])
        for key in ('physical_endpoint_penetration_cap_mm','interframe_reconstruction_allowance_mm','certified_interframe_penetration_cap_mm','penetration_cap_mm','source_sha256'):
            changed=dict(chord);changed[key]='changed'
            with self.assertRaises(ValueError):reconstruction_offenders(changed)
        self.assertEqual(reconstruction_offenders(dict(chord,offenders=[]))[1],[])


if __name__=='__main__':unittest.main()
