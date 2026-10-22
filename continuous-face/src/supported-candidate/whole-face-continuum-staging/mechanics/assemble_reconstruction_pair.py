"""Copy already audited case histories into one immutable matched report."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from source_transfer import sha
from accepted_history import validate_resume_interval_audits,validate_accepted_history
from reconstruction_seed import RECONSTRUCTION_FIELDS

def main():
    p=argparse.ArgumentParser();p.add_argument('--passive-report',type=Path,required=True);p.add_argument('--active-report',type=Path,required=True)
    p.add_argument('--audits',type=Path,nargs=2,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    passive=json.loads(a.passive_report.read_text());active=json.loads(a.active_report.read_text());r=copy.deepcopy(active)
    for key in ('source_domain_sha256','source_phase_sha256','source_support_sha256','guard_module_sha256','physical_mass_kg','activation_ramp_s',*RECONSTRUCTION_FIELDS):
        if passive[key]!=active[key]:raise ValueError('Paired source/physics/reconstruction mismatch: '+key)
    r['runs']={'passive':copy.deepcopy(passive['runs']['passive']),'zygomaticus_frontalis':copy.deepcopy(active['runs']['zygomaticus_frontalis'])}
    if r['runs']['passive']['actual_times_s']!=r['runs']['zygomaticus_frontalis']['actual_times_s']:raise ValueError('Not matched physical times')
    validate_resume_interval_audits(r,a.audits,r['runs'])
    for run in r['runs'].values():
        if any(not row['accepted'] for row in run['reports']):raise ValueError('Failed diagnostic rows cannot become paired history')
        if sha(Path(run['accepted_npz_path']))!=run['accepted_npz_sha256']:raise ValueError('Changed actual state file')
        with np.load(run['accepted_npz_path']) as z:
            validate_accepted_history(z['source_initial_positions_mm'],z['positions_mm'],z['velocities_mm_s'],z['physical_times_s'],run['reports'])
    a.output.mkdir(exist_ok=False,parents=True)
    for case,run in r['runs'].items():
        target=(a.output/f'{case}-accepted.npz').resolve();shutil.copyfile(run['accepted_npz_path'],target)
        run['accepted_npz_path']=str(target);assert sha(target)==run['accepted_npz_sha256']
    r.update(status='matched_independently_audited_physical_histories',accepted_facial_frames=sum(x['accepted_frames'] for x in r['runs'].values()),
        paired_admission=dict(source_reports={str(x.resolve()):sha(x) for x in (a.passive_report,a.active_report)},
            per_case_interval_proofs={str(x.resolve()):sha(x) for x in a.audits},
            saved_positions_velocities_times_unchanged=True,alternative_strict_branch_not_merged=True))
    r['passive_reconstruction_policy_transition']=passive.get('reconstruction_policy_transition')
    r['passive_accuracy_policy_transition']=passive.get('accuracy_policy_transition')
    (a.output/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    print(json.dumps(dict(report_path=str((a.output/'report.json').resolve()),physical_times_s=r['runs']['passive']['actual_times_s'])))
if __name__=='__main__':main()
