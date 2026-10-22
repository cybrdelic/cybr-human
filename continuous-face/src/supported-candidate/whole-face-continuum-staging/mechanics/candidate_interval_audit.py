"""External interval qualification at a saved-case boundary, never a solve."""
import json
from pathlib import Path
import subprocess
import sys
from source_transfer import sha
from report_storage import load_report


def audit_candidate_case(report_path, case, previous_audit, output_tag, *, runner, env, cwd, run_command=subprocess.run):
    """Save no mechanics state; audit runner owns its immutable snapshots."""
    report_path,previous_audit,runner=map(Path,(report_path,previous_audit,runner))
    before=load_report(report_path)['runs'][case]['accepted_npz_sha256']
    output=runner.parent/'temporal-audits'/f'{output_tag}.json'
    if output.exists():
        raise ValueError('A new immutable audit output tag is required')
    log=report_path.parent/f'{output_tag}-external-audit.log'
    with log.open('w') as stream:
        proc=run_command([sys.executable,'-u',str(runner),'--mechanics-report',str(report_path),
            '--case',case,'--output-tag',output_tag,'--previous-audit',str(previous_audit)],
            cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    audit=json.loads(output.read_text()) if output.exists() else None
    matching=bool(isinstance(audit,dict) and audit.get('case')==case
        and audit.get('accepted_npz_sha256')==before
        and audit.get('audited_snapshots_unchanged_during_audit') is True
        and audit.get('contact_sources_unchanged_during_audit') is True)
    return dict(path=str(output.resolve()),sha256=sha(output) if output.exists() else None,
        exit_code=proc.returncode,qualified=bool(proc.returncode==0 and matching and audit.get('qualified') is True),
        matching_candidate_and_immutable_proof=matching,
        scope='External continuous closed-owner physical-site cap and carrier-J interval; no global surface certificate')
