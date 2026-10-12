#!/usr/bin/env python3
"""Fast stdlib-only source/dependency/NPZ contracts. No build or render."""
from pathlib import Path
import argparse,ast,hashlib,json,struct,unittest,zipfile
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--appearance-root',type=Path,required=True);args,rest=p.parse_known_args();A=args.appearance_root.resolve()
sha=lambda b:hashlib.sha256(b).hexdigest()
def headers(path):
 out={}
 with zipfile.ZipFile(path) as z:
  for name in sorted(z.namelist()):
   with z.open(name) as f:
    assert f.read(6)==b'\x93NUMPY';version=tuple(f.read(2));n=struct.unpack('<H' if version==(1,0) else '<I',f.read(2 if version==(1,0) else 4))[0];h=ast.literal_eval(f.read(n).decode('latin1').strip());assert not str(h['descr']).endswith('O');out[name]={'dtype':h['descr'],'shape':list(h['shape']),'fortran_order':h['fortran_order']}
 return out
class ContractTests(unittest.TestCase):
 def test_python_syntax(self):
  files=list(R.rglob('*.py'));self.assertGreater(len(files),10)
  for path in files:compile(path.read_text(),str(path),'exec')
 def test_exact_source_manifest(self):
  manifest=json.loads((R/'SOURCE_MANIFEST.json').read_text());actual={str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='SOURCE_MANIFEST.json'};self.assertEqual(set(manifest),actual)
  for name,want in manifest.items():self.assertEqual(sha((R/name).read_bytes()),want,name)
 def test_repository_dependency_bytes(self):
  d=json.loads((R/'DEPENDENCIES.json').read_text());self.assertFalse(d['verified_remote_tree_truncated']);self.assertEqual(d['appearance_root'],'procedural/appearance')
  for x in d['files']:
   self.assertEqual(x['repository_path'],'procedural/appearance/'+x['appearance_relative_path']);b=(A/x['appearance_relative_path']).read_bytes();self.assertEqual(sha(b),x['sha256'],x['repository_path']);self.assertEqual(hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),x['git_blob_sha'])
 def test_opt_in_and_pinned_outputs(self):
  p=json.loads((R/'PROFILE.json').read_text());c=json.loads((R/'CONTROL_CONTRACT.json').read_text());self.assertIs(p['selected'],False);self.assertIs(p['mechanical_rest_default'],False);self.assertEqual(p['preferred_head_sha256'],'545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e');self.assertEqual(p['head_sha256'],c['expected_output']['head_sha256']);self.assertEqual(c['expected_output']['vertices'],742924);self.assertEqual(c['expected_output']['triangles'],1485840)
 def test_typed_control_inputs(self):
  c=json.loads((R/'CONTROL_CONTRACT.json').read_text());actual={str(p.relative_to(R)) for p in R.rglob('*.npz')};self.assertEqual(set(c['source_npz_headers']),actual)
  for name,want in c['source_npz_headers'].items():self.assertEqual(headers(R/name),want,name)
 def test_callable_construction_interfaces(self):
  c=json.loads((R/'CONTROL_CONTRACT.json').read_text())
  for name,want in c['required_functions'].items():
   functions={n.name:n for n in ast.parse((R/name).read_text()).body if isinstance(n,ast.FunctionDef)}
   for fun,count in want.items():self.assertIn(fun,functions);self.assertEqual(len(functions[fun].args.args),count,(name,fun))
 def test_no_baked_head_or_render_assets(self):
  for p in R.rglob('*'):
   self.assertNotIn(p.name,['head.npz','candidate_head.npz']);self.assertNotIn(p.suffix.lower(),['.png','.jpg','.blend'])
if __name__=='__main__':unittest.main(argv=[__file__,*rest],verbosity=2)
