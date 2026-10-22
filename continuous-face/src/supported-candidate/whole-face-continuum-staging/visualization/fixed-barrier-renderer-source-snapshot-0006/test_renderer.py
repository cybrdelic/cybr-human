"""Synthetic pixel/math tests only. These fixtures are never presented as physics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from literal_renderer import Rasterizer, Camera, project, reconstruct_boundary, make_cameras, sha_file
from accepted_frames import load_accepted_frames, boundary_frame, validate_generalized_stationarity
from render_actual import movie_timing, verify_encoded_movie
from snapshot_actual import snapshot
from compare_actual import match_times, summary
from external_path_audit import validate_closed_path_audit, bound_array_sha, ARRAY_HASH_CONVENTION, CONTACT_SOURCE_FIELDS, NEW_INTERVAL_DT_SCOPE, RECONSTRUCTION_CONTRACT, RECONSTRUCTION_LIMITS
from render_mobile_actual import mobile_camera
from variable_movie import variable_movie_timing, encode_variable_movie, verify_variable_movie
from PIL import Image
from stationarity_accuracy import POLICY, validate_stationarity_accuracy
from external_path_audit import RECONSTRUCTION_020_CONTRACT,RECONSTRUCTION_020_LIMITS,RECONSTRUCTION_020_PROOF_SOURCES
from paired_cursor_binding import read_paired_cursor,validate_paired_cursor_binding


def external_fixture(frames,case,frame_sha,report_sha,report):
    """Input-validation JSON only; this never certifies or renders physics."""
    provenance=report.get('coarse_contact_provenance',{})
    sources={file:provenance.get(key,'a'*64) for file,key in CONTACT_SOURCE_FIELDS.items()}
    sources.update({file:'f'*64 for file in ('continuous_closed_site_path.py','audit_accepted_site_chords.py','volume_guard.py')})
    times=frames['physical_times_s']
    intervals=[]
    for i,t in enumerate(times):
        q0=frames['source_initial_positions_mm'] if i==0 else frames['positions_mm'][i-1]
        v0=np.zeros_like(q0) if i==0 else frames['velocities_mm_s'][i-1]
        intervals.append({'frame_index':i,'t0_s':0. if i==0 else float(times[i-1]),'t1_s':float(t),
            'q0_sha256':bound_array_sha(q0),'q1_sha256':bound_array_sha(frames['positions_mm'][i]),
            'previous_velocity_sha256':bound_array_sha(v0),'qualified':True,
            'site_chord':{'continuous_closed_owner_site_cap_checked':True,'penetration_cap_mm':.05,
                'source_sha256':sources['continuous_closed_site_path.py']},
            'positive_J_path':{'certified':True}})
    return {'schema':'continuous_closed_site_external_audit_v1','qualified':True,'case':case,
        'accepted_npz_sha256':frame_sha,'mechanics_report_sha256':report_sha,
        'array_hash_convention':ARRAY_HASH_CONVENTION,
        'source_initial_positions_sha256':bound_array_sha(frames['source_initial_positions_mm']),
        'original_source_unchanged_during_audit':False,'audited_snapshots_unchanged_during_audit':True,
        'contact_sources_unchanged_during_audit':True,'physical_force_law_changed':False,'physical_history_changed':False,
        'physical_query_data_sha256':{key:provenance.get(key) for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256')},
        'saved_velocity_history_consistent':True,
        'global_surface_collision_certified':False,'unsigned_open_sheet_non_crossing_certified':False,
        'contact_sources_sha256':sources,'accepted_frames_audited':len(times),'physical_times_s':times.tolist(),
        'intervals':intervals,**{key:report.get(key) for key in
            ('source_domain_sha256','source_phase_sha256','source_support_sha256','site_resolution_mm')}}

class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r=Rasterizer()

    def camera(self): return Camera(0.,64,64,0.,0.,20.,-2.,2.)

    def draw(self,p,f,classes=None):
        f=np.asarray(f,np.int32); c=np.zeros(len(f),np.uint8) if classes is None else np.asarray(classes,np.uint8)
        buffers=self.r.buffers(self.camera())
        stats=self.r.draw(buffers,np.asarray(p,float),f,c,self.camera(),
            palette=np.array([[255.,0.,0.],[0.,0.,255.]]),lighting=False,depth_cue=False)
        return buffers,stats

    def test_finite_projection_and_rotation(self):
        v=np.array([[1.,2.,3.],[-4.,5.,6.]])
        np.testing.assert_array_equal(project(v,0),v)
        np.testing.assert_allclose(project(v,90),np.c_[v[:,2],v[:,1],-v[:,0]],atol=1e-14)
        self.assertTrue(np.isfinite(project(v,32)).all())
        with self.assertRaises(ValueError): project([[np.nan,0,0]],0)
        with self.assertRaises(ValueError): project(v,np.inf)

    def test_winding_no_backface_culling(self):
        p=[[-1,-1,0],[1,-1,0],[0,1,0]]
        a,sa=self.draw(p,[[0,1,2]]); b,sb=self.draw(p,[[2,1,0]])
        np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_allclose(a[1],b[1])
        self.assertGreater(sa['covered_samples'],0); self.assertEqual(sb['submitted'],1)

    def test_occlusion_independent_submission_order(self):
        p=[[-1,-1,-1],[1,-1,-1],[0,1,-1],[-1,-1,1],[1,-1,1],[0,1,1]]
        a,_=self.draw(p,[[0,1,2],[3,4,5]],[0,1]); b,_=self.draw(p,[[3,4,5],[0,1,2]],[1,0])
        np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[0][32,32],[0,0,255])

    def test_intersecting_triangles_use_per_pixel_depth(self):
        p=[[-1,-1,-1],[1,-1,1],[0,1,0],[-1,-1,1],[1,-1,-1],[0,1,0]]
        (rgb,z,owner),_=self.draw(p,[[0,1,2],[3,4,5]],[0,1])
        np.testing.assert_array_equal(rgb[38,24],[0,0,255]); np.testing.assert_array_equal(rgb[38,40],[255,0,0])
        self.assertAlmostEqual(z[38,40],(40.5-32)/20)

    def test_shared_triangle_edge_has_no_holes(self):
        p=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]
        (rgb,z,owner),_=self.draw(p,[[0,1,2],[0,2,3]])
        self.assertTrue(np.isfinite(z[12:52,12:52]).all())

    def test_degenerate_and_outside_finite_triangles(self):
        p=[[0,0,0],[0,0,1],[0,0,2],[9,9,0],[10,9,0],[9,10,0]]
        _,s=self.draw(p,[[0,1,2],[3,4,5]])
        self.assertEqual(s['submitted'],2); self.assertEqual(s['edge_on'],1); self.assertEqual(s['outside'],1)
        with self.assertRaises(ValueError): self.draw(p,[[0,1,100]])
        with self.assertRaises(ValueError): self.r.draw(self.r.buffers(self.camera()),np.asarray(p,float),
            np.array([[0,1,2**33]],np.int64),np.array([0],np.uint8),self.camera())

    def test_exact_shared_coordinate_reconstruction(self):
        q=np.array([[3,4,5],[7,4,5],[3,8,5],[3,4,9]],float)
        t=np.array([[0,1,2,3]],np.int32); parent=np.array([0,0,0],np.int32)
        bary=np.array([[.25,.25,.25,.25],[.25,.25,.25,.25],[0,1,0,0]])
        actual=reconstruct_boundary(q,t,parent,bary)
        expected=np.einsum('vi,vij->vj',bary,q[t[parent]],optimize=False)
        np.testing.assert_array_equal(actual,expected); np.testing.assert_array_equal(actual[0],actual[1])
        np.testing.assert_array_equal(actual[2],q[1])
        with self.assertRaises(ValueError): reconstruct_boundary(q,t,np.array([1,0,0]),bary)

    def test_fixed_aspect_common_scale_and_no_recenter(self):
        v=np.array([[-1,-3,-2],[1,3,2]],float); cameras=make_cameras(v)
        self.assertEqual(len(set(c.pixels_per_unit for c in cameras)),1)
        for c in cameras: c.check_boundary(v)
        with self.assertRaises(ValueError): cameras[0].check_boundary(v+100)

    def test_mobile_camera_has_fixed_uniform_whole_face_scale(self):
        reference=np.array([[-80.,-100.,-50.],[80.,100.,60.]])
        camera=mobile_camera(reference)
        self.assertEqual(camera.degrees,0.)
        self.assertEqual((camera.width,camera.height),(1000,1160))
        camera.check_boundary(reference)
        # A physical unit has the same pixel scale vertically/horizontally.
        self.assertAlmostEqual(camera.pixels_per_unit,1160/232)
        with self.assertRaises(ValueError): camera.check_boundary(reference+1000.)

    def test_actual_loader_acceptance_identity_and_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); nodes=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
            g={'sha256':'identity-A','nodes':nodes,'tets':np.array([[0,1,2,3]]),
               'exterior_tet':np.array([0]),'exterior_barycentric':np.array([[.25,.25,.25,.25]])}
            q=np.array([nodes+.01,nodes+.02]); ts=np.array([.001,.002])
            vv=(q-np.array([nodes,q[0]]))/.001
            # This temporary file exists only to test input validation. It is
            # never handed to the image renderer and not retained as an output.
            np.savez(p/'fixture.npz',positions_mm=q,velocities_mm_s=vv,
                physical_times_s=ts,source_initial_positions_mm=nodes,scale_mm_per_L0=1.)
            rows=[{'accepted':True,'status':'converged','fixture_only':False,'physical_time_s':float(t),
                'feasible':True,'endpoint':{'feasible':True},'minimum_J':1.,'dynamic_residual_max_N':1e-8,
                'force_tolerance_N':1e-7,'dt_s':.001,'frame_path':{'feasible':True,'whole_segment_certified':True}} for t in ts]
            report={'source_domain_sha256':'identity-A','runs':{'fixture':{'accepted_frames':2,
                'actual_times_s':ts.tolist(),'reports':rows,'accepted_npz_sha256':sha_file(p/'fixture.npz')}}}
            def load(r):
                (p/'report.json').write_text(json.dumps(r))
                audit_path=None
                if r.get('coarse_contact_provenance',{}).get('exterior_branch_representation'):
                    audit_path=p/'external-validation-fixture.json'
                    fixture_frames={'positions_mm':q,'velocities_mm_s':vv,
                        'physical_times_s':ts,'source_initial_positions_mm':nodes}
                    audit_path.write_text(json.dumps(external_fixture(fixture_frames,'fixture',
                        sha_file(p/'fixture.npz'),sha_file(p/'report.json'),r)))
                return load_accepted_frames(p/'fixture.npz',p/'report.json','fixture',g,audit_path)
            frames,audit=load(report)
            np.testing.assert_array_equal(boundary_frame(frames,0,g),np.einsum('vi,vij->vj',g['exterior_barycentric'],q[0][g['tets'][g['exterior_tet']]]))
            coarse=copy.deepcopy(report)
            coarse.update(contact_mode='coarse_physical_sites',global_surface_collision_certified=False,site_resolution_mm=1.)
            for row in coarse['runs']['fixture']['reports']:
                row['frame_path'].update(whole_segment_certified=False,coarse_whole_segment_checked=True)
            _,coarse_audit=load(coarse)
            self.assertEqual(coarse_audit['contact_mode'],'coarse_physical_sites')
            self.assertIs(coarse_audit['global_surface_collision_certified'],False)
            barrier=copy.deepcopy(coarse)
            barrier['contact_enforcement']='compact_cap_barrier_v1'
            for row in barrier['runs']['fixture']['reports']:
                row.update(contact_enforcement='compact_cap_barrier_v1',
                    barrier_continuation_qualified=True,lower_mu_comparison_passed=True,
                    final_inner_stationarity_converged=True,lower_mu_site_difference_mm=.004,
                    final_physical_path_qualified=True,physical_frame_path_deferred=False,
                    acceptance_scope='physical_frame_with_final_parameter_source_to_endpoint_path',
                    strict_positive_slack=True,complementarity_bound_satisfied=True,
                    dynamic_residual_includes_barrier_reaction=True,
                    lower_mu_site_tolerance_mm=.005,
                    contact_accounting={'validation_fixture_only':True,
                        'complementarity_bound_satisfied':True,'barrier_scale':.2,
                        'penalty_energy_N_mm':1e-5,'barrier_energy_N_mm':1e-7,
                        'max_lambda_s_N_mm':1e-8,'sum_lambda_s_N_mm':1e-7,
                        'maximum_active_mu_N_mm':1e-8,'per_owner':[]},
                    barrier_continuation={'validation_fixture_only':True,
                        'qualified':True,'same_physical_time_step':True,
                        'physical_history_unchanged_during_stages':True,
                        'committed_physical_time_steps':1,
                        'stages':[{'not_committed_physical_frame':True,
                                   'inner_stationarity_converged':True,
                                   'physical_frame_path_deferred':True,
                                   'acceptance_scope':'internal_same_timestep_stationarity_only'} for _ in range(2)]})
            _,barrier_audit=load(barrier)
            self.assertFalse(barrier_audit['complementarity_is_exact'])
            self.assertEqual(barrier_audit['accepted_frame_evidence'][0]['lower_mu_site_difference_mm'],.004)
            epigraph=copy.deepcopy(barrier)
            epigraph.update(exterior_contact_stationarity='finite_exterior_energy_epigraph_v1',
                source_phase_sha256='e'*64,source_support_sha256='d'*64,physical_dt_s=.001,
                epigraph_mechanics_source_sha256='a'*64,guard_module_sha256='b'*64,
                coarse_contact_provenance={'exterior_branch_representation':'complete_exterior_finite_energy_branches_v1',
                    'original_minimum_distance_energy_unchanged':True,
                    'exterior_branch_provider_sha256':'b'*64,'exterior_branch_source_sha256':'c'*64})
            for key in CONTACT_SOURCE_FIELDS.values(): epigraph['coarse_contact_provenance'].setdefault(key,'a'*64)
            for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):
                epigraph['coarse_contact_provenance'][key]='f'*64
            winner={'contact_generalized_stationarity_qualified':True,
                'exterior_contact_stationarity':'exact_selected_winner_already_stationary',
                'residual_projection_scope':'source_origin_support_reactions_only',
                'generalized_contact':{'kind':'exact_selected_winner_already_stationary',
                    'energy_law_unchanged':True,'actual_comaximality_satisfied':True,
                    'branch_weights_nonnegative':True,'seam_rows_are_source_support_pins':False,
                    'group_weight_sum_error':0.,'comaximality_tolerance_N_mm':1e-12,
                    'maximum_used_branch_energy_gap_N_mm':0.,'max_branch_complementarity_N_mm':0.,
                    'sum_branch_complementarity_N_mm':0.,'mixed_groups':[]}}
            for row in epigraph['runs']['fixture']['reports']:
                row.update(copy.deepcopy(winner))
                for stage in row['barrier_continuation']['stages']: stage.update(copy.deepcopy(winner))
            _,epigraph_audit=load(epigraph)
            self.assertEqual(epigraph_audit['epigraph_mechanics_source_sha256'],'a'*64)
            for change in ('missing_top_mode','factory_hash','missing_stage_evidence','projection'):
                broken=copy.deepcopy(epigraph)
                if change=='missing_top_mode': broken.pop('exterior_contact_stationarity')
                if change=='factory_hash': broken['guard_module_sha256']='d'*64
                if change=='missing_stage_evidence': broken['runs']['fixture']['reports'][0]['barrier_continuation']['stages'][0].pop('generalized_contact')
                if change=='projection': broken['runs']['fixture']['reports'][0]['residual_projection_scope']='projected_seam'
                with self.assertRaises(ValueError,msg='epigraph '+change): load(broken)
            for change in ('barrier_continuation_qualified','lower_mu_comparison_passed',
                           'final_inner_stationarity_converged','missing_contact_accounting',
                           'missing_continuation','budget','loosened_budget','unknown_contract',
                           'missing_contract','row_contract','complementarity','accounting_nan',
                           'fake_time','wrong_commits','unconverged_stage','deferred_final_path',
                           'internal_scope','final_path_failed','reaction_excluded'):
                broken=copy.deepcopy(barrier)
                row=broken['runs']['fixture']['reports'][0]
                if change in ('barrier_continuation_qualified','lower_mu_comparison_passed',
                              'final_inner_stationarity_converged'): row[change]=False
                if change=='missing_contact_accounting': row.pop('contact_accounting')
                if change=='missing_continuation': row.pop('barrier_continuation')
                if change=='budget': row['lower_mu_site_difference_mm']=.005
                if change=='loosened_budget': row['lower_mu_site_tolerance_mm']=.006
                if change=='unknown_contract': broken['contact_enforcement']='unrecognized'
                if change=='missing_contract': broken.pop('contact_enforcement')
                if change=='row_contract': row.pop('contact_enforcement')
                if change=='complementarity': row['contact_accounting']['complementarity_bound_satisfied']=False
                if change=='accounting_nan': row['contact_accounting']['max_lambda_s_N_mm']=float('nan')
                if change=='fake_time': row['barrier_continuation']['same_physical_time_step']=False
                if change=='wrong_commits': row['barrier_continuation']['committed_physical_time_steps']=2
                if change=='unconverged_stage': row['barrier_continuation']['stages'][0]['inner_stationarity_converged']=False
                if change=='deferred_final_path': row['physical_frame_path_deferred']=True
                if change=='internal_scope': row['acceptance_scope']='internal_same_timestep_stationarity_only'
                if change=='final_path_failed': row['final_physical_path_qualified']=False
                if change=='reaction_excluded': row['dynamic_residual_includes_barrier_reaction']=False
                with self.assertRaises(ValueError,msg='barrier '+change): load(broken)
            for change in ('claim','resolution','site_check','missing_mode'):
                broken=copy.deepcopy(coarse)
                if change=='claim': broken['global_surface_collision_certified']=True
                if change=='resolution': broken.pop('site_resolution_mm')
                if change=='site_check': broken['runs']['fixture']['reports'][0]['frame_path'].pop('coarse_whole_segment_checked')
                if change=='missing_mode': broken.pop('contact_mode')
                with self.assertRaises(ValueError,msg='coarse '+change): load(broken)
            for change in ('geometry','fixture','path','time','accepted','endpoint','residual','volume','file_hash','missing_file_hash'):
                broken=copy.deepcopy(report)
                if change=='geometry': broken['source_domain_sha256']='identity-B'
                if change=='fixture': broken['runs']['fixture']['reports'][0]['fixture_only']=True
                if change=='path': broken['runs']['fixture']['reports'][0]['frame_path']['whole_segment_certified']=False
                if change=='time': broken['runs']['fixture']['actual_times_s'][0]=.003
                if change=='accepted': broken['runs']['fixture']['reports'][0]['accepted']=False
                if change=='endpoint': broken['runs']['fixture']['reports'][0]['feasible']=False
                if change=='residual': broken['runs']['fixture']['reports'][0]['dynamic_residual_max_N']=1e-4
                if change=='volume': broken['runs']['fixture']['reports'][0]['minimum_J']=-1
                if change=='file_hash': broken['runs']['fixture']['accepted_npz_sha256']='0'*64
                if change=='missing_file_hash': broken['runs']['fixture'].pop('accepted_npz_sha256')
                with self.assertRaises(ValueError,msg=change): load(broken)
            np.savez(p/'fixture.npz',positions_mm=q,velocities_mm_s=np.zeros_like(q),
                physical_times_s=ts,source_initial_positions_mm=nodes[::-1],scale_mm_per_L0=1.)
            with self.assertRaises(ValueError): load(report)

    def test_movie_timing_rejects_stills_and_invented_cadence(self):
        q=np.zeros((3,4,3)); q[1]+=1; q[2]+=2
        self.assertAlmostEqual(movie_timing(np.array([.001,.002,.003]),q,24),.024)
        for ts,positions in [(np.array([.001]),q[:1]),
                             (np.array([.001,.002]),np.zeros((2,4,3))),
                             (np.array([.001,.002,.004]),q)]:
            with self.assertRaises(ValueError): movie_timing(ts,positions,24)

    def test_comparison_uses_exact_time_matches_and_unscaled_metrics(self):
        times,ia,ip=match_times(np.array([.001,.002,.003]),np.array([.001,.003,.004]))
        np.testing.assert_array_equal(times,[.001,.003])
        np.testing.assert_array_equal(ia,[0,2]); np.testing.assert_array_equal(ip,[0,1])
        with self.assertRaises(ValueError): match_times(np.array([.001]),np.array([.00100000001]))
        result=summary(np.array([0.,3.,4.]))
        self.assertEqual(result['maximum'],4.)
        self.assertAlmostEqual(result['rms'],np.sqrt(25/3))
        self.assertEqual(summary([])['count'],0)
        with self.assertRaises(ValueError): summary([np.nan])

    def test_movie_verification_checks_encoded_count_size_and_rate(self):
        # This mocks only ffprobe metadata. No synthetic motion movie is made.
        stream={'nb_read_frames':'3','width':2250,'height':1200,'avg_frame_rate':'24/1'}
        with patch('render_actual.subprocess.run') as run:
            run.return_value.stdout=json.dumps({'streams':[stream]})
            self.assertTrue(verify_encoded_movie('unused',3,24,(2250,1200))['decoded_frame_count_verified'])
            for key,value in (('nb_read_frames','4'),('height',1198),('avg_frame_rate','25/1')):
                broken=dict(stream); broken[key]=value
                run.return_value.stdout=json.dumps({'streams':[broken]})
                with self.assertRaises(RuntimeError): verify_encoded_movie('unused',3,24,(2250,1200))

    def test_generalized_contact_rejects_seam_projection_and_off_seam_mixtures(self):
        evidence={'kind':'finite_exterior_energy_epigraph_v1','energy_law_unchanged':True,
            'actual_comaximality_satisfied':True,'branch_weights_nonnegative':True,
            'seam_rows_are_source_support_pins':False,'group_weight_sum_error':0.,
            'comaximality_tolerance_N_mm':1e-12,'maximum_used_branch_energy_gap_N_mm':1e-14,
            'max_branch_complementarity_N_mm':5e-15,'sum_branch_complementarity_N_mm':5e-15,
            'mixed_groups':[{'group':0,'weights':[.5,.5],'branch_energy_gaps_N_mm':[0.,1e-14],
                'branch_indices':[0,1],'branch_site':[4,4],'branch_owner':[2,2],'branch_triangle':[3,7]}]}
        row={'contact_generalized_stationarity_qualified':True,
            'exterior_contact_stationarity':'finite_exterior_energy_epigraph_v1',
            'residual_projection_scope':'source_origin_support_reactions_only',
            'smooth_selected_branch_dynamic_residual_max_N':.003,'generalized_contact':evidence}
        validate_generalized_stationarity(row)
        for change in ('projected_seam','off_seam','missing_qualification','negative_weight','weight_sum',
                       'wrong_projection','missing_native_face','bad_gap','nan_complementarity'):
            broken=copy.deepcopy(row); e=broken['generalized_contact']
            if change=='projected_seam': e['seam_rows_are_source_support_pins']=True
            if change=='off_seam': e['actual_comaximality_satisfied']=False
            if change=='missing_qualification': broken.pop('contact_generalized_stationarity_qualified')
            if change=='negative_weight': e['mixed_groups'][0]['weights']=[-1.,2.]
            if change=='weight_sum': e['mixed_groups'][0]['weights']=[.4,.4]
            if change=='wrong_projection': broken['residual_projection_scope']='source_and_seam_rows'
            if change=='missing_native_face': e['mixed_groups'][0].pop('branch_triangle')
            if change=='bad_gap': e['mixed_groups'][0]['branch_energy_gaps_N_mm'][1]=2e-12
            if change=='nan_complementarity': e['sum_branch_complementarity_N_mm']=float('nan')
            with self.assertRaises(ValueError,msg=change): validate_generalized_stationarity(broken)
        winner=copy.deepcopy(row)
        winner['exterior_contact_stationarity']='exact_selected_winner_already_stationary'
        winner.pop('smooth_selected_branch_dynamic_residual_max_N')
        winner['generalized_contact'].update(kind='exact_selected_winner_already_stationary',
            mixed_groups=[],maximum_used_branch_energy_gap_N_mm=0.,max_branch_complementarity_N_mm=0.,
            sum_branch_complementarity_N_mm=0.)
        validate_generalized_stationarity(winner)
        winner['generalized_contact']['sum_branch_complementarity_N_mm']=1e-20
        with self.assertRaises(ValueError): validate_generalized_stationarity(winner)

    def test_external_continuous_audit_binds_every_actual_interval(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'external-input-validation-only.json'
            q0=np.arange(12,dtype=np.float64).reshape(4,3)
            frames={'source_initial_positions_mm':q0,'positions_mm':np.array([q0+.001,q0+.002]),
                'velocities_mm_s':np.ones((2,4,3)),'physical_times_s':np.array([.001,.002])}
            report={'coarse_contact_provenance':{key:'a'*64 for key in CONTACT_SOURCE_FIELDS.values()},
                'source_domain_sha256':'b'*64,'source_phase_sha256':'c'*64,
                'source_support_sha256':'d'*64,'site_resolution_mm':1.,'physical_dt_s':.001,
                'runs':{'fixture':{'reports':[{'accepted':True,'dt_s':.001} for _ in range(2)]}}}
            for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):
                report['coarse_contact_provenance'][key]='f'*64
            original=external_fixture(frames,'fixture','1'*64,'2'*64,report)
            def check(value):
                path.write_text(json.dumps(value))
                return validate_closed_path_audit(path,frames,'fixture','1'*64,'2'*64,report)
            self.assertFalse(check(original)['global_surface_collision_certified'])
            for change in ('qualified','npz','report','q1','velocity','time','missing_interval',
                           'site_path','volume','source','law','changed_during_audit','global_claim'):
                broken=copy.deepcopy(original)
                if change=='qualified': broken['qualified']=False
                if change=='npz': broken['accepted_npz_sha256']='3'*64
                if change=='report': broken['mechanics_report_sha256']='3'*64
                if change=='q1': broken['intervals'][1]['q1_sha256']='3'*64
                if change=='velocity': broken['intervals'][1]['previous_velocity_sha256']=bound_array_sha(np.zeros_like(q0))
                if change=='time': broken['intervals'][1]['t0_s']=0.
                if change=='missing_interval': broken['intervals'].pop()
                if change=='site_path': broken['intervals'][0]['site_chord']['continuous_closed_owner_site_cap_checked']=False
                if change=='volume': broken['intervals'][0]['positive_J_path']['certified']=False
                if change=='source': broken['contact_sources_sha256']['coarse_rigid_query.py']='3'*64
                if change=='law': broken['physical_force_law_changed']=True
                if change=='changed_during_audit': broken['audited_snapshots_unchanged_during_audit']=False
                if change=='global_claim': broken['global_surface_collision_certified']=True
                with self.assertRaises(ValueError,msg=change): check(broken)

    def test_incremental_audit_preserves_actual_adaptive_history(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'incremental-input-validation-only.json'
            q0=np.arange(12,dtype=float).reshape(4,3)
            frames={'source_initial_positions_mm':q0,'positions_mm':np.array([q0+.001,q0+.003]),
                'velocities_mm_s':np.ones((2,4,3)),'physical_times_s':np.array([.001,.003])}
            report={'coarse_contact_provenance':{key:'a'*64 for key in CONTACT_SOURCE_FIELDS.values()},
                'source_domain_sha256':'b'*64,'source_phase_sha256':'c'*64,'source_support_sha256':'d'*64,
                'site_resolution_mm':1.,'physical_dt_s':.002,'physical_dt_s_scope':NEW_INTERVAL_DT_SCOPE,
                'runs':{'fixture':{'reports':[{'accepted':True,'dt_s':dt} for dt in (.001,.002)]}}}
            for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):
                report['coarse_contact_provenance'][key]='f'*64
            original=external_fixture(frames,'fixture','1'*64,'2'*64,report)
            del original['contact_sources_sha256']['audit_accepted_site_chords.py']
            original['contact_sources_sha256']['audit_new_accepted_intervals.py']='e'*64
            original.update(reused_prefix_intervals=1,reused_prefix_audit_sha256='3'*64,
                new_intervals_audited=1,physical_interval_dt_s=[.001,.002])
            original['intervals'][0].update(reused_without_new_queries=True,reused_from_audit_sha256='3'*64)
            def check(value,rep=report):
                path.write_text(json.dumps(value))
                return validate_closed_path_audit(path,frames,'fixture','1'*64,'2'*64,rep)
            self.assertFalse(check(original)['global_surface_collision_certified'])
            for change in ('prefix_count','prefix_hash','reuse_flag','new_count','physical_dt','wrong_new_reuse'):
                broken=copy.deepcopy(original)
                if change=='prefix_count': broken['reused_prefix_intervals']=0
                if change=='prefix_hash': broken['intervals'][0]['reused_from_audit_sha256']='4'*64
                if change=='reuse_flag': broken['intervals'][0]['reused_without_new_queries']=False
                if change=='new_count': broken['new_intervals_audited']=2
                if change=='physical_dt': broken['physical_interval_dt_s']=[.002,.002]
                if change=='wrong_new_reuse': broken['intervals'][1]['reused_without_new_queries']=True
                with self.assertRaises(ValueError,msg=change): check(broken)
            for change in ('old_scope','unknown_scope','wrong_new_dt','wrong_prior_row_dt'):
                rep=copy.deepcopy(report)
                if change=='old_scope': rep.pop('physical_dt_s_scope')
                if change=='unknown_scope': rep['physical_dt_s_scope']='arbitrary'
                if change=='wrong_new_dt': rep['physical_dt_s']=.001
                if change=='wrong_prior_row_dt': rep['runs']['fixture']['reports'][0]['dt_s']=.002
                with self.assertRaises(ValueError,msg=change): check(original,rep)

    def test_reconstruction_proof_separates_endpoint_and_interframe_caps(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'reconstruction-input-validation-only.json'
            q0=np.arange(12,dtype=float).reshape(4,3)
            frames={'source_initial_positions_mm':q0,'positions_mm':np.array([q0+.001,q0+.003]),
                'velocities_mm_s':np.ones((2,4,3)),'physical_times_s':np.array([.001,.003])}
            report={'coarse_contact_provenance':{key:'a'*64 for key in CONTACT_SOURCE_FIELDS.values()},
                'source_domain_sha256':'b'*64,'source_phase_sha256':'c'*64,'source_support_sha256':'d'*64,
                'site_resolution_mm':1.,'physical_dt_s':.002,'physical_dt_s_scope':NEW_INTERVAL_DT_SCOPE,
                'reconstruction_contract':RECONSTRUCTION_CONTRACT,**RECONSTRUCTION_LIMITS,
                'runs':{'fixture':{'reports':[{'accepted':True,'dt_s':dt,'endpoint':{'maximum_site_penetration_mm':.049}} for dt in (.001,.002)]}}}
            report['runs']['fixture']['reports'][1].update(reconstruction_contract=RECONSTRUCTION_CONTRACT,**RECONSTRUCTION_LIMITS)
            for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):
                report['coarse_contact_provenance'][key]='f'*64
            original=external_fixture(frames,'fixture','1'*64,'2'*64,report)
            del original['contact_sources_sha256']['audit_accepted_site_chords.py']
            original['contact_sources_sha256'].update({'audit_reconstruction_intervals.py':'e'*64,
                'continuous_closed_site_reconstruction.py':'9'*64})
            original.update(reconstruction_contract=RECONSTRUCTION_CONTRACT,**RECONSTRUCTION_LIMITS,
                reused_prefix_intervals=1,reused_prefix_audit_sha256='3'*64,
                new_intervals_audited=1,physical_interval_dt_s=[.001,.002])
            original['intervals'][0].update(reused_without_new_queries=True,reused_from_audit_sha256='3'*64)
            original['intervals'][1]['site_chord'].update(penetration_cap_mm=.055,source_sha256='9'*64,
                physical_contact_law_changed=False,**RECONSTRUCTION_LIMITS)
            def check(value,rep=report):
                path.write_text(json.dumps(value))
                return validate_closed_path_audit(path,frames,'fixture','1'*64,'2'*64,rep)
            self.assertEqual(check(original)['bounded_interframe_geometric_allowance_mm'],.005)
            self.assertTrue(check(original)['geometric_allowance_is_not_a_total_temporal_or_state_error_bound'])
            for change in ('prefix_cap','prefix_source','new_cap','new_source','endpoint_cap','allowance','law','top_contract'):
                broken=copy.deepcopy(original);chord=broken['intervals'][1]['site_chord']
                if change=='prefix_cap': broken['intervals'][0]['site_chord']['penetration_cap_mm']=.055
                if change=='prefix_source': broken['intervals'][0]['site_chord']['source_sha256']='9'*64
                if change=='new_cap': chord['penetration_cap_mm']=.056
                if change=='new_source': chord['source_sha256']='f'*64
                if change=='endpoint_cap': chord['physical_endpoint_penetration_cap_mm']=.055
                if change=='allowance': chord['interframe_reconstruction_allowance_mm']=.006
                if change=='law': chord['physical_contact_law_changed']=True
                if change=='top_contract': broken['reconstruction_contract']='unbounded'
                with self.assertRaises(ValueError,msg=change): check(broken)
            for change in ('endpoint','frame_contract','frame_limit','report_limit'):
                rep=copy.deepcopy(report);row=rep['runs']['fixture']['reports'][1]
                if change=='endpoint': row['endpoint']['maximum_site_penetration_mm']=.05001
                if change=='frame_contract': row.pop('reconstruction_contract')
                if change=='frame_limit': row['physical_endpoint_penetration_cap_mm']=.055
                if change=='report_limit': rep['certified_interframe_penetration_cap_mm']=.056
                with self.assertRaises(ValueError,msg=change): check(original,rep)

    def test_new_reconstruction_keeps_both_stricter_prefixes_and_endpoint_cap(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'versioned-contract-validation-only.json';q0=np.arange(12,dtype=float).reshape(4,3)
            frames={'source_initial_positions_mm':q0,'positions_mm':np.array([q0+.001,q0+.003,q0+.005]),
                'velocities_mm_s':np.ones((3,4,3)),'physical_times_s':np.array([.001,.003,.005])}
            policy={'reconstruction_contract':RECONSTRUCTION_020_CONTRACT,**RECONSTRUCTION_020_LIMITS,
                'reconstruction_contract_schema':'fixed_reconstruction_accuracy_v2','trajectory_error_bound_proven':False,
                'reconstruction_allowance_scope':'Fixed geometric approximation, not a total trajectory/state error bound'}
            report={'coarse_contact_provenance':{key:'a'*64 for key in CONTACT_SOURCE_FIELDS.values()},
                'source_domain_sha256':'b'*64,'source_phase_sha256':'c'*64,'source_support_sha256':'d'*64,
                'site_resolution_mm':1.,'physical_dt_s':.002,'physical_dt_s_scope':NEW_INTERVAL_DT_SCOPE,**policy,
                'runs':{'fixture':{'reports':[{'accepted':True,'dt_s':dt,'endpoint':{'maximum_site_penetration_mm':.049}} for dt in (.001,.002,.002)]}}}
            rows=report['runs']['fixture']['reports'];rows[1].update(reconstruction_contract=RECONSTRUCTION_CONTRACT,**RECONSTRUCTION_LIMITS);rows[2].update(policy)
            for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):report['coarse_contact_provenance'][key]='f'*64
            original=external_fixture(frames,'fixture','1'*64,'2'*64,report)
            original['contact_sources_sha256'].update(RECONSTRUCTION_020_PROOF_SOURCES)
            original['contact_sources_sha256']['continuous_closed_site_reconstruction.py']='9'*64
            original.update(policy,reused_prefix_intervals=2,reused_prefix_audit_sha256='3'*64,
                new_intervals_audited=1,physical_interval_dt_s=[.001,.002,.002])
            for row in original['intervals'][:2]:row.update(reused_without_new_queries=True,reused_from_audit_sha256='3'*64)
            original['intervals'][1]['site_chord'].update(penetration_cap_mm=.055,source_sha256='9'*64,
                physical_contact_law_changed=False,**RECONSTRUCTION_LIMITS)
            original['intervals'][2]['site_chord'].update(policy,penetration_cap_mm=.07,
                source_sha256=RECONSTRUCTION_020_PROOF_SOURCES['continuous_closed_site_reconstruction_020.py'],physical_contact_law_changed=False)
            def check(value,rep=report):
                path.write_text(json.dumps(value));return validate_closed_path_audit(path,frames,'fixture','1'*64,'2'*64,rep)
            self.assertEqual(check(original)['bounded_interframe_geometric_allowance_mm'],.02)
            for field,bad in (('reconstruction_contract_schema','unknown'),('trajectory_error_bound_proven',True),
                    ('physical_endpoint_penetration_cap_mm',.07),('interframe_reconstruction_allowance_mm',.021),
                    ('certified_interframe_penetration_cap_mm',.071)):
                broken=copy.deepcopy(original);broken[field]=bad
                with self.assertRaises(ValueError,msg=field):check(broken)
                broken=copy.deepcopy(original);broken['intervals'][2]['site_chord'][field]=bad
                with self.assertRaises(ValueError,msg='chord '+field):check(broken)
            broken=copy.deepcopy(original);broken['contact_sources_sha256']['audit_reconstruction_020_intervals.py']='1'*64
            with self.assertRaises(ValueError):check(broken)
            broken=copy.deepcopy(original);broken['intervals'][2]['site_chord']['penetration_cap_mm']=.071
            with self.assertRaises(ValueError):check(broken)
            rep=copy.deepcopy(report);rep['runs']['fixture']['reports'][2]['endpoint']['maximum_site_penetration_mm']=.053
            with self.assertRaises(ValueError):check(original,rep)
            rep=copy.deepcopy(report)
            for item in (rep,rep['runs']['fixture']['reports'][2]):
                item.pop('reconstruction_contract_schema');item.pop('trajectory_error_bound_proven')
            self.assertEqual(check(original,rep)['bounded_interframe_geometric_allowance_mm'],.02)
            rep['runs']['fixture']['reports'][2].pop('reconstruction_allowance_scope')
            with self.assertRaises(ValueError):check(original,rep)

    def test_new_policy_requires_exact_admitted_paired_proofs_not_geometric_eligibility(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);times=np.array([.005,.00625]);paths={}
            for case in ('passive','zygomaticus_frontalis'):
                p=root/(case+'.json');p.write_text(json.dumps({'schema':'continuous_closed_site_external_audit_v1',
                    'qualified':True,'case':case,'audited_snapshots_unchanged_during_audit':True,'physical_times_s':times.tolist()}));paths[case]=str(p)
            cursor=root/'cursor.json';cursor.write_text(json.dumps({'accepted_physical_time_s':.00625,'accepted_interval_audits':paths}))
            snapshot=read_paired_cursor(cursor);ap=Path(paths['zygomaticus_frontalis']);digest=sha_file(ap)
            self.assertEqual(validate_paired_cursor_binding(snapshot,'zygomaticus_frontalis',ap,digest,times)['accepted_physical_time_s'],.00625)
            with self.assertRaises(ValueError):validate_paired_cursor_binding(None,'zygomaticus_frontalis',ap,digest,times)
            with self.assertRaises(ValueError):validate_paired_cursor_binding(snapshot,'zygomaticus_frontalis',ap,'1'*64,times)
            with self.assertRaises(ValueError):validate_paired_cursor_binding(snapshot,'zygomaticus_frontalis',ap,digest,np.array([.005,.0075]))
            bad=copy.deepcopy(snapshot);bad['document']['accepted_physical_time_s']=.0075
            with self.assertRaises(ValueError):validate_paired_cursor_binding(bad,'zygomaticus_frontalis',ap,digest,times)
            pp=Path(paths['passive']);proof=json.loads(pp.read_text());proof['qualified']=False;pp.write_text(json.dumps(proof))
            with self.assertRaises(ValueError):validate_paired_cursor_binding(snapshot,'zygomaticus_frontalis',ap,digest,times)

    def test_qualified_tangent_factory_preserves_base_provider_and_controller_hash_binding(self):
        from tangent_source_identity import validate_epigraph_factory_identity,REVISION,FACTORY_SHA,CURVATURE_SHA,BASE_FACTORY_SHA,QUALIFIED_SOURCES
        report={'guard_module_sha256':FACTORY_SHA,'coarse_contact_provenance':{
            'interior_feature_tangent_revision':REVISION,'interior_feature_tangent_source_sha256':FACTORY_SHA,
            'interior_feature_curvature_source_sha256':CURVATURE_SHA,'base_epigraph_factory_sha256':BASE_FACTORY_SHA,
            'exterior_branch_provider_sha256':BASE_FACTORY_SHA,'unchanged_energy_gradient_guards':True}}
        cursor={'document':{'source_sha256':{'/qualified/'+k:v for k,v in QUALIFIED_SOURCES.items()}}}
        qualified=validate_epigraph_factory_identity(report,cursor)
        self.assertTrue(qualified['numerical_tangent_only']);self.assertFalse(qualified['geometric_auditor_executes_this_tangent'])
        with self.assertRaises(ValueError):validate_epigraph_factory_identity(report)
        for key in report['coarse_contact_provenance']:
            broken=copy.deepcopy(report);broken['coarse_contact_provenance'][key]='unqualified'
            with self.assertRaises(ValueError):validate_epigraph_factory_identity(broken,cursor)
        for suffix in QUALIFIED_SOURCES:
            broken=copy.deepcopy(cursor);broken['document']['source_sha256']['/qualified/'+suffix]='a'*64
            with self.assertRaises(ValueError):validate_epigraph_factory_identity(report,broken)
        old={'guard_module_sha256':BASE_FACTORY_SHA,'coarse_contact_provenance':{'exterior_branch_provider_sha256':BASE_FACTORY_SHA}}
        self.assertIsNone(validate_epigraph_factory_identity(old))
        old['coarse_contact_provenance']['interior_feature_tangent_source_sha256']=FACTORY_SHA
        with self.assertRaises(ValueError):validate_epigraph_factory_identity(old)

    def test_case_contrast_retains_actual_numerical_parameter_difference(self):
        from measure_latest_pair import numerical_pair_contrast
        a={'accepted_frame_evidence':[{'contact_accounting':{'barrier_scale':.04},'lower_mu_site_difference_mm':.002}],
            'barrier_scales':[.2,.04,.008],'physical_times_s':[.01]}
        p=copy.deepcopy(a);p['accepted_frame_evidence'][0]['contact_accounting']['barrier_scale']=.008
        p['accepted_frame_evidence'][0]['lower_mu_site_difference_mm']=.001
        contrast=numerical_pair_contrast(a,p,0,0)
        self.assertFalse(contrast['same_final_barrier_scale']);self.assertFalse(contrast['cumulative_barrier_scale_histories_equal'])
        self.assertFalse(contrast['exact_isolated_muscle_effect_claim']);self.assertTrue(contrast['both_cases_use_the_same_fixed_continuation_schedule'])
        self.assertEqual(contrast['active_lower_mu_sensitivity_mm'],.002)
        a.pop('barrier_scales');p.pop('barrier_scales')
        self.assertIsNone(numerical_pair_contrast(a,p,0,0)['both_cases_use_the_same_fixed_continuation_schedule'])

    def test_latest_pair_cursor_requires_admitted_time_and_both_proofs(self):
        from measure_latest_pair import cursor_pair_inputs
        cursor={'accepted_physical_time_s':.01,'accepted_interval_audits':{'passive':'passive.json','zygomaticus_frontalis':'active.json'}}
        paths,t=cursor_pair_inputs(cursor)
        self.assertEqual(paths,[Path('active.json'),Path('passive.json')]);self.assertEqual(t,.01)
        for value in (0.,-1.,float('nan')):
            broken=copy.deepcopy(cursor);broken['accepted_physical_time_s']=value
            with self.assertRaises(ValueError):cursor_pair_inputs(broken)
        broken=copy.deepcopy(cursor);del broken['accepted_interval_audits']['passive']
        with self.assertRaises(ValueError):cursor_pair_inputs(broken)

    def test_comparison_ignores_only_independent_setup_wall_timings(self):
        from compare_actual import comparable_contact_provenance
        a={'setup_seconds':1.,'initial_site_state':{'site_query_seconds':2.,'feasible':True},
            'rigid_query':{'build_seconds':3.,'source_sha256':'a'*64},'physical_dt_s':.00125}
        b=copy.deepcopy(a);b['setup_seconds']=4.;b['initial_site_state']['site_query_seconds']=5.;b['rigid_query']['build_seconds']=6.
        self.assertEqual(comparable_contact_provenance(a),comparable_contact_provenance(b))
        self.assertEqual(a['setup_seconds'],1.)
        b['physical_dt_s']=.005
        self.assertNotEqual(comparable_contact_provenance(a),comparable_contact_provenance(b))
        b['physical_dt_s']=a['physical_dt_s'];b['rigid_query']['source_sha256']='b'*64
        self.assertNotEqual(comparable_contact_provenance(a),comparable_contact_provenance(b))

    def test_approximate_stationarity_is_explicit_and_full_unregularized(self):
        report={'stationarity_accuracy_policy':POLICY,'force_state_dynamics_source_sha256':'a'*64,
            'force_state_accuracy_source_sha256':'b'*64}
        row={'dynamic_residual_max_N':1.36e-7,'force_tolerance_N':1e-7,'approximate_stationarity':True,
            'absolute_force_target_passed':False,'dt_s':.00125,'relative_dynamic_balance_max':4.44e-6,
            'force_component_magnitude_sum_max_N':1.36e-7/4.44e-6,
            'stationarity_accuracy':{'policy':POLICY,'route':'full_qp_force_state_fallback','qualified':True,
                'full_qp_actual_reaction_qualified':True,'qp_solved':True,'search_only':False,
                'numerical_relative_diagonal_regularization':0.,'actual_comaximality_satisfied':True,
                'branch_weights_nonnegative':True,'group_weight_sum_error':1e-15,
                'relative_force_balance':4.44e-6,'relative_force_tolerance':1e-5,
                'maximum_occupied_site_correction_mm':3.98e-8,'maximum_occupied_site_correction_tolerance_mm':.0005,
                'maximum_occupied_site_velocity_correction_mm_s':3.98e-8/.00125,
                'source_constraint_direction_error_mm':1e-20,
                'residual_projection_scope':'source_origin_support_reactions_only',
                'all_physical_contact_and_barrier_reactions_included':True}}
        self.assertTrue(validate_stationarity_accuracy(row,report)['approximate_stationarity'])
        for key,bad in (('full_qp_actual_reaction_qualified',False),('qp_solved',False),('search_only',True),
                ('numerical_relative_diagonal_regularization',.001),('actual_comaximality_satisfied',False),
                ('branch_weights_nonnegative',False),('group_weight_sum_error',1e-9),('relative_force_balance',1e-4),
                ('maximum_occupied_site_correction_mm',.0006),('source_constraint_direction_error_mm',1e-7),
                ('maximum_occupied_site_velocity_correction_mm_s',1.),('residual_projection_scope','seam projected'),
                ('relative_force_tolerance',1e-4),('maximum_occupied_site_correction_tolerance_mm',.005)):
            broken=copy.deepcopy(row);broken['stationarity_accuracy'][key]=bad
            with self.assertRaises(ValueError,msg=key): validate_stationarity_accuracy(broken,report)
        for key,bad in (('approximate_stationarity',False),('absolute_force_target_passed',True),
                ('force_tolerance_N',1e-6),('relative_dynamic_balance_max',0.)):
            broken=copy.deepcopy(row);broken[key]=bad
            with self.assertRaises(ValueError,msg=key): validate_stationarity_accuracy(broken,report)
        with self.assertRaises(ValueError): validate_stationarity_accuracy(row,{})
        with self.assertRaises(ValueError): validate_stationarity_accuracy(row,{'stationarity_accuracy_policy':POLICY})
        primary={'dynamic_residual_max_N':1e-8,'force_tolerance_N':1e-7}
        self.assertFalse(validate_stationarity_accuracy(primary,{})['approximate_stationarity'])
        self.assertFalse(validate_stationarity_accuracy(primary,{'stationarity_accuracy_policy':POLICY})['approximate_stationarity'])
        primary['dynamic_residual_max_N']=1.1e-7
        with self.assertRaises(ValueError): validate_stationarity_accuracy(primary,{})

    def test_variable_timing_preserves_adaptive_intervals_without_states(self):
        times=np.array([.005,.00625,.006875])
        q=np.arange(36,dtype=float).reshape(3,4,3)
        t=variable_movie_timing(times,q,50.)
        self.assertEqual(t['presentation_timestamps_ticks'],[0,62500,93750])
        self.assertEqual(t['sample_duration_ticks'],[62500,31250,31250])
        self.assertFalse(t['terminal_hold_is_new_physical_interval'])
        for invalid in (times[:1],np.array([.005,.005,.00625]),np.array([.005,.00625,.006])):
            with self.assertRaises(ValueError): variable_movie_timing(invalid,q[:len(invalid)],50.)
        for factor in (0.,.5,np.nan):
            with self.assertRaises(ValueError): variable_movie_timing(times,q,factor)
        same=q.copy();same[1]=same[0]
        with self.assertRaises(ValueError): variable_movie_timing(times,same,50.)
        with self.assertRaises(ValueError): variable_movie_timing(np.array([1.,1.+1e-9,2.]),q,1.)

    def test_real_encoder_decodes_each_variable_timestamp_once(self):
        # Synthetic solid-color codec fixtures only, never physics or a deliverable.
        with tempfile.TemporaryDirectory(prefix='vfr-codec-fixture-',dir=Path(__file__).parent) as td:
            folder=Path(td);paths=[]
            for i,color in enumerate(('red','green','blue')):
                path=folder/f'fixture-{i}.png';Image.new('RGB',(64,64),color).save(path);paths.append(path)
            timing=variable_movie_timing(np.array([.005,.00625,.006875]),np.arange(36.).reshape(3,4,3),50.)
            movie=folder/'codec-fixture-not-physics.mp4'
            receipt=encode_variable_movie(paths,movie,timing,(64,64))
            self.assertEqual(receipt['encoded_frames'],3)
            self.assertTrue(receipt['encoding_verification']['every_presentation_timestamp_verified'])
            self.assertEqual(receipt['encoding_verification']['decoded_presentation_timestamps_ticks'],[0,62500,93750])
            broken=copy.deepcopy(timing);broken['presentation_timestamps_ticks'][1]+=1
            with self.assertRaises(RuntimeError): verify_variable_movie(movie,broken,(64,64))

    def test_snapshot_copies_only_hash_bound_existing_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); source=p/'source.npz'
            np.savez(source,temporary_validation_fixture=np.array([1.,2.]))
            row={'accepted_frames':1,'actual_times_s':[.01],
                 'accepted_npz_path':str(source),'accepted_npz_sha256':sha_file(source)}
            report={'runs':{'fixture':row}}
            report_bytes=json.dumps(report).encode()
            (p/'report.json').write_bytes(report_bytes)
            copied=snapshot(p,'fixture',p/'copy')
            self.assertTrue(copied['byte_exact_snapshot'])
            self.assertEqual((p/'copy/report.json').read_bytes(),report_bytes)
            self.assertEqual((p/'copy/accepted.npz').read_bytes(),source.read_bytes())
            with self.assertRaises(ValueError): snapshot(p,'fixture',p/'copy')
            row['accepted_npz_sha256']='0'*64
            (p/'report.json').write_text(json.dumps(report))
            with self.assertRaises(RuntimeError): snapshot(p,'fixture',p/'bad-copy')

if __name__=='__main__': unittest.main(verbosity=2)
