"""Bind new-policy render inputs to one captured admitted paired cursor.

The controller may keep advancing. Only frozen proof artifacts are rehashed;
the live cursor is not a manual render/admission lock.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from report_storage import read_report


def read_paired_cursor(path):
    path=Path(path);raw=path.read_bytes()
    return {'source_path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),
        'document':json.loads(raw),'raw_bytes':raw}


def validate_paired_cursor_binding(snapshot,case,audit_path,audit_sha256,times):
    if not isinstance(snapshot,dict):raise ValueError('New reconstruction policy requires a captured admitted paired cursor')
    raw=snapshot.get('raw_bytes')
    if not isinstance(raw,bytes) or hashlib.sha256(raw).hexdigest()!=snapshot.get('sha256') or json.loads(raw)!=snapshot.get('document'):
        raise ValueError('Paired cursor snapshot bytes/identity do not match')
    cursor=snapshot['document'];end=cursor.get('accepted_physical_time_s',float('nan'))
    if not np.isfinite(end) or not len(times) or end!=float(times[-1]):
        raise ValueError('New-policy history is not the admitted paired cursor endpoint')
    paths=cursor.get('accepted_interval_audits',{});selected={}
    if case not in ('passive','zygomaticus_frontalis'):raise ValueError('Unsupported paired cursor case')
    for key in ('passive','zygomaticus_frontalis'):
        name=paths.get(key)
        if not isinstance(name,str) or not name:raise ValueError('Both admitted independent proof references are required')
        path=Path(name);proof_raw=path.read_bytes();digest=hashlib.sha256(proof_raw).hexdigest();proof=json.loads(proof_raw)
        if proof.get('schema')!='continuous_closed_site_external_audit_v1' or proof.get('qualified') is not True or \
                proof.get('case')!=key or proof.get('audited_snapshots_unchanged_during_audit') is not True or \
                not np.array_equal(np.asarray(proof.get('physical_times_s',[])),times):
            raise ValueError('Cursor does not select two qualified proofs for this exact physical history')
        selected[key]={'path':str(path.resolve()),'sha256':digest,
            'accepted_npz_sha256':proof.get('accepted_npz_sha256'),'mechanics_report_sha256':proof.get('mechanics_report_sha256')}
        if key==case and digest!=audit_sha256:
            raise ValueError('This independent proof is not the one selected by the admitted paired cursor')
    binding={'source_cursor_path':snapshot['source_path'],'source_cursor_sha256':snapshot['sha256'],
        'accepted_physical_time_s':float(end),'selected_independent_proofs':selected,
        'live_cursor_need_not_remain_unchanged_after_snapshot':True}
    directory=cursor.get('accepted_history_directory')
    if directory is not None:
        report_path=Path(directory)/'report.json';stored_report=read_report(report_path);report=stored_report.document
        for key in selected:
            run=report.get('runs',{}).get(key,{})
            if run.get('accepted_npz_sha256')!=selected[key]['accepted_npz_sha256'] or \
                    not np.array_equal(np.asarray(run.get('actual_times_s',[])),times):
                raise ValueError('Cursor admission report does not preserve the exact independently audited pair')
        binding.update(paired_admission_report_path=str(report_path.resolve()),
            paired_admission_report_sha256=stored_report.file_sha256,
            paired_admission_report_raw_sha256=stored_report.raw_sha256,
            paired_admission_report_storage=stored_report.identity(),
            paired_numerical_parameter_contrast=report.get('paired_numerical_parameter_contrast'))
    return binding
