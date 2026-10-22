import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from coarse_rigid_query import NativeRigidQuery
from coarse_physical_site_contact import sha
from continuous_closed_site_path import certify_closed_owner_site_chord
from continuous_closed_site_reconstruction import certify_closed_owner_reconstruction_chord as old_reconstruction
from continuous_closed_site_reconstruction_020 import certify_closed_owner_reconstruction_chord
from audit_reconstruction_020_intervals import reusable_prefix
from test_continuous_closed_site_path import point_guard

HERE=Path(__file__).resolve().parent
CONTACT=HERE.parent

def slab(halfwidth):
    V=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    V[:,2]*=halfwidth
    F=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return point_guard(NativeRigidQuery(V,F,np.ones(len(F),int)))


class ReconstructionTests(unittest.TestCase):
    def test_060_intermediate_only_under_new_declared_allowance(self):
        g=slab(.06);q0=np.array([[.13,.07,.11]]);q1=np.array([[.23,.07,-.11]])
        self.assertFalse(certify_closed_owner_site_chord(g,q0,q1)['feasible'])
        self.assertFalse(old_reconstruction(g,q0,q1)['feasible'])
        seen=[];old=g.rigid.query_band
        def query(points,**kw):
            seen.append(kw.get('penetration_limit',.05));return old(points,**kw)
        g.rigid.query_band=query
        result=certify_closed_owner_reconstruction_chord(g,q0,q1)
        self.assertTrue(result['feasible']);self.assertIn(.07,seen)
        self.assertEqual(result['physical_endpoint_penetration_cap_mm'],.05)
        self.assertEqual(result['interframe_reconstruction_allowance_mm'],.02)
        self.assertEqual(result['certified_interframe_penetration_cap_mm'],.07)
        self.assertFalse(result['global_surface_collision_certified'])

    def test_053_endpoint_still_rejected(self):
        g=slab(.053);q=np.array([[.13,.07,0.]])
        self.assertFalse(certify_closed_owner_reconstruction_chord(g,q,q)['feasible'])

    def test_larger_crossings_and_exact_bound_rejected(self):
        for depth in (.07,.071,.077,.105):
            g=slab(depth)
            r=certify_closed_owner_reconstruction_chord(g,np.array([[.13,.07,.2]]),np.array([[.23,.07,-.2]]))
            self.assertFalse(r['feasible'],depth)

    def test_ray_ambiguity_and_nonfinite_midpoints_not_ignored(self):
        for bad in ('valid','signed_gap'):
            g=slab(.06);old=g.rigid.query_band
            def query(points,**kw):
                r=old(points,**kw)
                if kw.get('penetration_limit')==.07:r[bad][:]=False if bad=='valid' else np.nan
                return r
            g.rigid.query_band=query
            self.assertFalse(certify_closed_owner_reconstruction_chord(g,np.array([[.13,.07,.11]]),np.array([[.23,.07,-.11]]))['feasible'])

    def test_no_opportunistic_cap_parameter(self):
        g=slab(.053);g.penetration_limit=.06
        with self.assertRaises(ValueError):certify_closed_owner_reconstruction_chord(g,np.zeros((1,3)),np.zeros((1,3)))

    def test_existing_strict_prefix_reused_without_geometry_queries(self):
        path=CONTACT/'temporal-audits/step2-quarter-passive.json';prior=json.loads(path.read_text())
        with np.load(prior['accepted_npz_snapshot_path']) as z:
            inputs=dict(case='passive',q=z['positions_mm'],v=z['velocities_mm_s'],times=z['physical_times_s'],initial=z['source_initial_positions_mm'])
        sources=copy.deepcopy(prior['contact_sources_sha256'])
        sources['continuous_closed_site_reconstruction.py']=sha(CONTACT/'continuous_closed_site_reconstruction.py')
        sources['continuous_closed_site_reconstruction_020.py']=sha(HERE/'continuous_closed_site_reconstruction_020.py')
        sources['audit_reconstruction_020_intervals.py']=sha(HERE/'audit_reconstruction_020_intervals.py')
        inputs.update(sources=sources,physical_query_hashes=prior['physical_query_data_sha256'],physical_sources={k:prior[k] for k in ('source_domain_sha256','source_phase_sha256','source_support_sha256')})
        rows,digest=reusable_prefix(path,**inputs)
        self.assertEqual(len(rows),2);self.assertTrue(all(r['reused_without_new_queries'] for r in rows))
        with tempfile.TemporaryDirectory() as folder:
            altered=copy.deepcopy(prior);altered['intervals'][-1]['site_chord']['penetration_cap_mm']=.056
            changed=Path(folder)/'changed.json';changed.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):reusable_prefix(changed,**inputs)


    def prefix_inputs(self,path):
        prior=json.loads(path.read_text())
        with np.load(prior['accepted_npz_snapshot_path']) as z:
            inputs=dict(case='passive',q=z['positions_mm'],v=z['velocities_mm_s'],times=z['physical_times_s'],initial=z['source_initial_positions_mm'])
        sources=copy.deepcopy(prior['contact_sources_sha256'])
        for name,folder in [('continuous_closed_site_reconstruction.py',CONTACT),('continuous_closed_site_reconstruction_020.py',HERE),('audit_reconstruction_020_intervals.py',HERE)]:
            sources[name]=sha(folder/name)
        inputs.update(sources=sources,physical_query_hashes=prior['physical_query_data_sha256'],physical_sources={k:prior[k] for k in ('source_domain_sha256','source_phase_sha256','source_support_sha256')})
        return prior,inputs

    def test_old_055_prefix_retained_exactly_not_relabelled(self):
        path=CONTACT/'temporal-audits/reconstruction25-v1-002-passive.json'
        old_bytes=path.read_bytes();prior,inputs=self.prefix_inputs(path)
        rows,digest=reusable_prefix(path,**inputs)
        self.assertEqual(len(rows),5);self.assertEqual(rows[-1]['t1_s'],.01)
        self.assertEqual(rows[-1]['site_chord'],prior['intervals'][-1]['site_chord'])
        self.assertEqual(rows[-1]['site_chord']['penetration_cap_mm'],.055)
        self.assertEqual(path.read_bytes(),old_bytes)

    def test_prefix_hash_history_query_and_positive_j_fail_closed(self):
        path=CONTACT/'temporal-audits/reconstruction25-v1-002-passive.json';prior,inputs=self.prefix_inputs(path)
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'bad.json'
            for mutate in (
                lambda x:x.update(accepted_npz_sha256='changed'),
                lambda x:x.update(mechanics_report_sha256='changed'),
                lambda x:x['contact_sources_sha256'].update(continuous_closed_site_reconstruction_py='changed'),
                lambda x:x['intervals'][-1].update(q0_sha256='changed'),
                lambda x:x['intervals'][-1]['positive_J_path'].update(certified=False),
                lambda x:x['intervals'][-1]['site_chord'].update(source_sha256='changed'),
                lambda x:x['physical_query_data_sha256'].update(native_source_sha256='changed')):
                changed=copy.deepcopy(prior);mutate(changed)
                # Explicit real-key source mutation, not an unrelated extra key.
                if 'continuous_closed_site_reconstruction_py' in changed['contact_sources_sha256']:
                    changed['contact_sources_sha256']['continuous_closed_site_reconstruction.py']='changed'
                target.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):reusable_prefix(target,**inputs)
            changed=copy.deepcopy(inputs);changed['v'][-1,0,0]+=1
            with self.assertRaises(ValueError):reusable_prefix(path,**changed)

    def test_stricter_invalid_native_query_is_never_ignored(self):
        g=slab(.06);old=g.rigid.query_band
        def query(points,**kw):
            if kw.get('penetration_limit')==.07:
                return old(points,**dict(kw,penetration_limit=.05))
            return old(points,**kw)
        g.rigid.query_band=query
        r=certify_closed_owner_reconstruction_chord(g,np.array([[.13,.07,.11]]),np.array([[.23,.07,-.11]]))
        self.assertFalse(r['feasible']);self.assertFalse(r['offenders'][0]['native_query_valid'])

    def test_bounded_missing_certificate_is_not_acceptance(self):
        g=slab(.06)
        r=certify_closed_owner_reconstruction_chord(g,np.array([[.13,.07,.11]]),np.array([[.23,.07,-.11]]),max_midpoint_queries=0,enable_triangle_tubes=False)
        self.assertFalse(r['feasible']);self.assertIn('Inconclusive',r['reason'])


if __name__=='__main__':unittest.main()
