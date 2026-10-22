"""Lossless report-storage identity checks for the renderer, not physics gates."""
from report_storage import read_report

STORAGE_HELPER_SHA='530fcaaa21d2f1e8d72abe586d5b6713c1bdba38e57e751fcd1c750692af02f0'
STORAGE_AUDITOR_SHA='3d5169ddc75bd456c024826f97a82bb66c703c45eccb703e7686d9ba0fb66990'


def _content_identity(identity):
    """Allow an exact-byte relocated copy, never missing or changed companions."""
    if not isinstance(identity,dict):raise ValueError('Missing complete report storage identity')
    files=identity.get('required_files')
    if not isinstance(files,list) or not files or any(not isinstance(f,dict) or not isinstance(f.get('path'),str) or not f['path'] for f in files):
        raise ValueError('Missing report storage required-file identities')
    return {**{key:identity.get(key) for key in ('report_storage_format','file_sha256','raw_sha256','raw_bytes')},
        'required_files':[{key:f.get(key) for key in ('role','sha256','bytes')} for f in files]}


def validate_report_storage_receipt(proof,stored_report):
    storage_aware=proof.get('contact_sources_sha256',{}).get('audit_reconstruction_020_intervals.py')==STORAGE_AUDITOR_SHA
    if storage_aware or 'mechanics_report_raw_sha256' in proof or 'mechanics_report_storage' in proof:
        if proof.get('mechanics_report_raw_sha256')!=stored_report.raw_sha256 or \
                _content_identity(proof.get('mechanics_report_storage'))!=_content_identity(stored_report.identity()):
            raise ValueError('Independent proof report raw/payload identities differ from the exact loaded bytes')


def recheck_report_bindings(audit):
    """Authenticate wrapper AND payload after export, including paired admission."""
    stored=read_report(audit['report_file'],expected_file_sha256=audit['report_sha256'])
    if stored.raw_sha256!=audit['report_raw_sha256'] or stored.identity()!=audit['report_storage']:
        raise RuntimeError('Immutable report storage identity changed during rendering')
    binding=audit.get('admitted_paired_cursor') or {}
    if binding.get('paired_admission_report_path') is not None:
        paired=read_report(binding['paired_admission_report_path'],expected_file_sha256=binding['paired_admission_report_sha256'])
        if paired.raw_sha256!=binding['paired_admission_report_raw_sha256'] or paired.identity()!=binding['paired_admission_report_storage']:
            raise RuntimeError('Immutable paired admission report storage identity changed during rendering')
