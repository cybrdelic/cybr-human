import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from coarse_rigid_query import NativeRigidQuery
from coarse_physical_site_contact import sha
from continuous_closed_site_path import certify_closed_owner_site_chord
from continuous_closed_site_reconstruction import certify_closed_owner_reconstruction_chord
from audit_reconstruction_intervals import reusable_prefix
from test_continuous_closed_site_path import point_guard

HERE=Path(__file__).resolve().parent

def slab(halfwidth):
    V=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    V[:,2]*=halfwidth
    F=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return point_guard(NativeRigidQuery(V,F,np.ones(len(F),int)))


class ReconstructionTests(unittest.TestCase):
    def test_053_intermediate_only_under_declared_allowance(self):
        g=slab(.053);q0=np.array([[.13,.07,.11]]);q1=np.array([[.23,.07,-.11]])
        self.assertFalse(certify_closed_owner_site_chord(g,q0,q1)['feasible'])
        seen=[];old=g.rigid.query_band
        def query(points,**kw):
            seen.append(kw.get('penetration_limit',.05));return old(points,**kw)
        g.rigid.query_band=query
        result=certify_closed_owner_reconstruction_chord(g,q0,q1)
        self.assertTrue(result['feasible']);self.assertIn(.055,seen)
        self.assertEqual(result['physical_endpoint_penetration_cap_mm'],.05)
        self.assertEqual(result['interframe_reconstruction_allowance_mm'],.005)
        self.assertEqual(result['certified_interframe_penetration_cap_mm'],.055)
        self.assertFalse(result['global_surface_collision_certified'])

    def test_053_endpoint_still_rejected(self):
        g=slab(.053);q=np.array([[.13,.07,0.]])
        self.assertFalse(certify_closed_owner_reconstruction_chord(g,q,q)['feasible'])

    def test_larger_crossings_and_exact_bound_rejected(self):
        for depth in (.055,.056,.077,.105):
            g=slab(depth)
            r=certify_closed_owner_reconstruction_chord(g,np.array([[.13,.07,.2]]),np.array([[.23,.07,-.2]]))
            self.assertFalse(r['feasible'],depth)

    def test_ray_ambiguity_and_nonfinite_midpoints_not_ignored(self):
        for bad in ('valid','signed_gap'):
            g=slab(.053);old=g.rigid.query_band
            def query(points,**kw):
                r=old(points,**kw)
                if kw.get('penetration_limit')==.055:r[bad][:]=False if bad=='valid' else np.nan
                return r
            g.rigid.query_band=query
            self.assertFalse(certify_closed_owner_reconstruction_chord(g,np.array([[.13,.07,.11]]),np.array([[.23,.07,-.11]]))['feasible'])

    def test_no_opportunistic_cap_parameter(self):
        g=slab(.053);g.penetration_limit=.06
        with self.assertRaises(ValueError):certify_closed_owner_reconstruction_chord(g,np.zeros((1,3)),np.zeros((1,3)))

    def test_existing_strict_prefix_reused_without_geometry_queries(self):
        path=HERE/'temporal-audits/step2-quarter-passive.json';prior=json.loads(path.read_text())
        with np.load(prior['accepted_npz_snapshot_path']) as z:
            inputs=dict(case='passive',q=z['positions_mm'],v=z['velocities_mm_s'],times=z['physical_times_s'],initial=z['source_initial_positions_mm'])
        sources=copy.deepcopy(prior['contact_sources_sha256'])
        sources['continuous_closed_site_reconstruction.py']=sha(HERE/'continuous_closed_site_reconstruction.py')
        sources['audit_reconstruction_intervals.py']=sha(HERE/'audit_reconstruction_intervals.py')
        inputs.update(sources=sources,physical_query_hashes=prior['physical_query_data_sha256'],physical_sources={k:prior[k] for k in ('source_domain_sha256','source_phase_sha256','source_support_sha256')})
        rows,digest=reusable_prefix(path,**inputs)
        self.assertEqual(len(rows),2);self.assertTrue(all(r['reused_without_new_queries'] for r in rows))
        with tempfile.TemporaryDirectory() as folder:
            altered=copy.deepcopy(prior);altered['intervals'][-1]['site_chord']['penetration_cap_mm']=.056
            changed=Path(folder)/'changed.json';changed.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):reusable_prefix(changed,**inputs)


if __name__=='__main__':unittest.main()
