"""Hash-bound passive q-only numerical seeds and declared frame approximation."""
import json
from pathlib import Path
import numpy as np
from source_transfer import sha
from report_storage import load_report, report_sha256
from accepted_history import validate_resume_interval_audits, validate_accepted_history

RECONSTRUCTION_FIELDS = dict(reconstruction_contract='fixed_0p005mm_interframe_allowance_v1',
    physical_endpoint_penetration_cap_mm=.05, interframe_reconstruction_allowance_mm=.005,
    certified_interframe_penetration_cap_mm=.055,
    reconstruction_allowance_scope='Fixed inter-frame geometric reconstruction allowance, not a total temporal or state error bound')


def audited_passive_endpoints(audit_path, current_report, source_initial):
    """Returns independent q copies only; no active q/v/time history is loaded."""
    audit_path=Path(audit_path);audit=json.loads(audit_path.read_text())
    if audit.get('qualified') is not True or audit.get('case') != 'passive':
        raise ValueError('A qualified passive interval audit is required')
    path=Path(audit['mechanics_report_path'])
    if report_sha256(path)!=audit['mechanics_report_sha256']:
        raise ValueError('Frozen passive report changed')
    prior=load_report(path)
    for key in ('source_domain_sha256','source_phase_sha256','source_support_sha256',
                'source_support_bundle_sha256','guard_module_sha256','activation_ramp_s',
                'physical_mass_kg','damping_per_second','gravity_mm_s2'):
        if prior.get(key)!=current_report.get(key):
            raise ValueError('Passive numerical seed physical model differs: '+key)
    for key in ('clearance_mm','penalty_stiffness_N_per_mm4','maximum_allowed_site_penetration_mm',
                'guard_scope','rigid_query_source_sha256','compact_barrier_source_sha256',
                'exterior_branch_provider_sha256','exterior_branch_source_sha256'):
        if prior['coarse_contact_provenance'].get(key)!=current_report['coarse_contact_provenance'].get(key):
            raise ValueError('Passive seed physical contact differs: '+key)
    validate_resume_interval_audits(prior,[audit_path],['passive'])
    saved=Path(audit['accepted_npz_snapshot_path'])
    if sha(saved)!=audit['accepted_npz_sha256']:
        raise ValueError('Frozen passive arrays changed')
    with np.load(saved,allow_pickle=False) as z:
        if not np.array_equal(z['source_initial_positions_mm'],source_initial):
            raise ValueError('Passive source reference changed')
        q=z['positions_mm'].copy();v=z['velocities_mm_s'].copy();times=z['physical_times_s'].tolist()
    rows=[r for r in prior['runs']['passive']['reports'] if r['accepted']]
    validate_accepted_history(source_initial,q,v,times,rows)
    return {t:position.copy() for t,position in zip(times,q)},dict(audit_path=str(audit_path.resolve()),
        audit_sha256=sha(audit_path),accepted_npz_sha256=audit['accepted_npz_sha256'],
        use='q-only Newton initial guess; active physical q_previous, velocity, time and prediction unchanged')
