"""Storage-only synthetic reader fixtures; never physical/movie frames."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from report_storage import read_report, write_report_snapshot, copy_report_snapshot
from renderer_report_binding import (validate_report_storage_receipt,recheck_report_bindings,
    STORAGE_AUDITOR_SHA,STORAGE_HELPER_SHA)
from literal_renderer import sha_file
from accepted_frames import load_accepted_frames
from snapshot_actual import snapshot
from paired_cursor_binding import read_paired_cursor,validate_paired_cursor_binding


def binding(stored):
    return dict(report_file=str(stored.path),report_sha256=stored.file_sha256,
        report_raw_sha256=stored.raw_sha256,report_storage=stored.identity())


class RendererStorageTests(unittest.TestCase):
    def test_exact_reviewed_helper(self):
        self.assertEqual(sha_file(Path(__file__).with_name('report_storage.py')),STORAGE_HELPER_SHA)

    def test_storage_receipt_requires_raw_identity_and_all_companions(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);saved=write_report_snapshot(p/'report.json',b'{ "diagnostic": Infinity }\n')
            proof=dict(contact_sources_sha256={'audit_reconstruction_020_intervals.py':STORAGE_AUDITOR_SHA},
                mechanics_report_raw_sha256=saved.raw_sha256,mechanics_report_storage=saved.identity())
            validate_report_storage_receipt(proof,saved)
            relocated=copy_report_snapshot(saved.path,p/'relocated/report.json')
            validate_report_storage_receipt(proof,relocated)
            for change in ('raw','length','missing_companion','missing_identity'):
                bad=copy.deepcopy(proof)
                if change=='raw':bad['mechanics_report_raw_sha256']='0'*64
                if change=='length':bad['mechanics_report_storage']['required_files'][1]['bytes']+=1
                if change=='missing_companion':bad['mechanics_report_storage']['required_files'].pop()
                if change=='missing_identity':del bad['mechanics_report_storage']
                with self.subTest(change=change),self.assertRaises(ValueError):validate_report_storage_receipt(bad,saved)

    def test_final_recheck_detects_payload_only_change(self):
        with tempfile.TemporaryDirectory() as td:
            stored=write_report_snapshot(Path(td)/'report.json',b'{"fixture":1}')
            audit=binding(stored);recheck_report_bindings(audit)
            stored.payload_path.write_bytes(stored.payload_bytes+b'x')
            self.assertEqual(sha_file(stored.path),stored.file_sha256)
            with self.assertRaises(ValueError):recheck_report_bindings(audit)

    def test_final_recheck_authenticates_paired_admission_payload(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);stored=write_report_snapshot(p/'report.json',b'{"fixture":1}')
            paired=write_report_snapshot(p/'pair/report.json',b'{"fixture":2}')
            audit=binding(stored);audit['admitted_paired_cursor']=dict(
                paired_admission_report_path=str(paired.path),paired_admission_report_sha256=paired.file_sha256,
                paired_admission_report_raw_sha256=paired.raw_sha256,paired_admission_report_storage=paired.identity())
            recheck_report_bindings(audit);paired.payload_path.write_bytes(b'bad')
            with self.assertRaises(ValueError):recheck_report_bindings(audit)

    def test_snapshot_retains_exact_compressed_wrapper_and_payload(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);np.savez(p/'fixture.npz',fixture=np.array([1.]))
            raw=json.dumps({'runs':{'fixture':{'accepted_frames':1,'actual_times_s':[.001],
                'accepted_npz_path':str(p/'fixture.npz'),'accepted_npz_sha256':sha_file(p/'fixture.npz')}}}).encode()
            stored=write_report_snapshot(p/'report.json',raw);out=snapshot(p,'fixture',p/'copy')
            copied=read_report(p/'copy/report.json',expected_file_sha256=stored.file_sha256)
            self.assertEqual(copied.file_bytes,stored.file_bytes);self.assertEqual(copied.raw,raw)
            self.assertEqual(copied.payload_bytes,stored.payload_bytes)
            self.assertEqual(out['snapshot_report_storage'],copied.identity())
            self.assertEqual(out['source_report_storage'],stored.identity())

    def test_paired_cursor_records_compressed_admission_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);times=np.array([.005,.00625]);proofs={};runs={}
            for case in ('passive','zygomaticus_frontalis'):
                pp=p/(case+'.json');pp.write_text(json.dumps(dict(schema='continuous_closed_site_external_audit_v1',
                    qualified=True,case=case,audited_snapshots_unchanged_during_audit=True,physical_times_s=times.tolist(),
                    accepted_npz_sha256='a'*64)));proofs[case]=str(pp)
                runs[case]=dict(actual_times_s=times.tolist(),accepted_npz_sha256='a'*64)
            stored=write_report_snapshot(p/'pair/report.json',json.dumps(dict(runs=runs)).encode())
            cp=p/'cursor.json';cp.write_text(json.dumps(dict(accepted_physical_time_s=.00625,
                accepted_history_directory=str(p/'pair'),accepted_interval_audits=proofs)))
            result=validate_paired_cursor_binding(read_paired_cursor(cp),'passive',proofs['passive'],sha_file(proofs['passive']),times)
            self.assertEqual(result['paired_admission_report_sha256'],stored.file_sha256)
            self.assertEqual(result['paired_admission_report_storage'],stored.identity())
            stored.payload_path.write_bytes(b'bad')
            with self.assertRaises(ValueError):validate_paired_cursor_binding(read_paired_cursor(cp),'passive',proofs['passive'],sha_file(proofs['passive']),times)

    def test_accepted_loader_plain_and_compressed_are_identical(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);nodes=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
            g=dict(sha256='fixture',nodes=nodes,tets=np.array([[0,1,2,3]]),
                exterior_tet=np.array([0]),exterior_barycentric=np.array([[.25,.25,.25,.25]]))
            q=np.array([nodes+.01,nodes+.02]);ts=np.array([.001,.002]);vv=(q-np.array([nodes,q[0]]))/.001
            np.savez(p/'fixture.npz',positions_mm=q,velocities_mm_s=vv,physical_times_s=ts,source_initial_positions_mm=nodes,scale_mm_per_L0=1.)
            rows=[dict(accepted=True,status='converged',fixture_only=False,physical_time_s=float(t),feasible=True,
                endpoint=dict(feasible=True),minimum_J=1.,dynamic_residual_max_N=1e-8,force_tolerance_N=1e-7,
                dt_s=.001,frame_path=dict(feasible=True,whole_segment_certified=True)) for t in ts]
            raw=json.dumps(dict(source_domain_sha256='fixture',runs=dict(fixture=dict(accepted_frames=2,actual_times_s=ts.tolist(),reports=rows,accepted_npz_sha256=sha_file(p/'fixture.npz'))))).encode()
            plain=write_report_snapshot(p/'plain.json',raw,compressed=False);compressed=write_report_snapshot(p/'compressed.json',raw)
            loaded=[]
            for stored in (plain,compressed):
                frames,audit=load_accepted_frames(p/'fixture.npz',stored.path,'fixture',g)
                self.assertEqual(audit['report_sha256'],stored.file_sha256);self.assertEqual(audit['report_raw_sha256'],plain.raw_sha256)
                self.assertEqual(audit['report_storage'],stored.identity());recheck_report_bindings(audit);loaded.append(frames)
            for key in loaded[0]:np.testing.assert_equal(loaded[0][key],loaded[1][key])

if __name__=='__main__':unittest.main()
