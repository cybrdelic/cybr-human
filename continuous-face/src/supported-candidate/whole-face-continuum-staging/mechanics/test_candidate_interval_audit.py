import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from candidate_interval_audit import audit_candidate_case

class CandidateAuditTests(unittest.TestCase):
    def test_pass_and_failed_interval_never_admit_by_exit_alone(self):
        for qualified,changed,code,expected in [(True,False,0,True),(False,False,2,False),(True,True,0,False)]:
            with self.subTest(qualified=qualified,changed=changed,code=code),TemporaryDirectory() as td:
                p=Path(td);(p/'temporal-audits').mkdir();report=p/'report.json'
                report.write_text(json.dumps(dict(runs=dict(passive=dict(accepted_npz_sha256='exact')))))
                def run(*args,**kwargs):
                    (p/'temporal-audits/new.json').write_text(json.dumps(dict(case='passive',qualified=qualified,
                        accepted_npz_sha256='wrong' if changed else 'exact',audited_snapshots_unchanged_during_audit=True,
                        contact_sources_unchanged_during_audit=True)))
                    return SimpleNamespace(returncode=code)
                result=audit_candidate_case(report,'passive',p/'prior.json','new',runner=p/'runner.py',
                    env={},cwd=p,run_command=run)
                self.assertEqual(result['qualified'],expected)
    def test_existing_proof_is_not_overwritten(self):
        with TemporaryDirectory() as td:
            p=Path(td);(p/'temporal-audits').mkdir();(p/'temporal-audits/new.json').write_text('{}')
            report=p/'report.json';report.write_text(json.dumps(dict(runs=dict(passive=dict(accepted_npz_sha256='x')))))
            with self.assertRaisesRegex(ValueError,'new immutable'):
                audit_candidate_case(report,'passive',p/'prior.json','new',runner=p/'runner.py',env={},cwd=p)

if __name__=='__main__':unittest.main()
