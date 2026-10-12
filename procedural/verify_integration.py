#!/usr/bin/env python3
"""Non-mutating package verification; never builds, solves or renders."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=json.loads((ROOT/'MANIFEST.json').read_text());fail=[]
 for rel,r in m['files'].items():
  p=ROOT/rel
  if not p.is_file() or sha(p)!=r['sha256']:fail.append(rel)
 if fail:raise SystemExit('Payload mismatch: '+repr(fail))
 n=0
 for p in ROOT.rglob('*.py'):ast.parse(p.read_text());n+=1
 work=ROOT/'mechanics/regional-reference/workspace';a=work/'human-preferred-v53-anatomical-assembly-20261011/resumed-source/test_parameterized_bonds.mjs'
 test=subprocess.run(['node',str(a)],check=True,capture_output=True,text=True)
 B=work/'human-preferred-v53-mechanics-20261011';g=np.load(B/'cheek-remesh-v9/geometry.npz');domain=json.loads((B/'cheek-case-v3/domain.json').read_text());rd=domain['arrays']['rest'];v=np.frombuffer((B/'cheek-case-v3/arrays.bin').read_bytes(),dtype=rd['dtype'],count=int(np.prod(rd['shape'])),offset=rd['byte_offset']).reshape(rd['shape']);t=g['tetrahedra'];base=np.linalg.det(np.stack([v[t[:,i]]-v[t[:,0]] for i in [1,2,3]],axis=2));states={}
 if np.any(base<=0):raise SystemExit('Invalid original rest tetrahedra')
 for name in ['neutral-0','right-02','right-08','right-16','released-0']:
  d=json.loads((B/'cheek-equilibrium-v3'/f'{name}-state.json').read_text());x=np.array(d['positions']).reshape(-1,3);J=np.linalg.det(np.stack([x[t[:,i]]-x[t[:,0]] for i in [1,2,3]],axis=2))/base
  if not np.isfinite(x).all() or J.min()<=.25 or not d['converged']:raise SystemExit('Original saved state failed '+name)
  fixed=g['fixed_nodes'];fixed=fixed if fixed.dtype.kind in 'iu' else np.flatnonzero(fixed)
  if not np.array_equal(x[fixed],v[fixed]):raise SystemExit('Fixed coordinates changed '+name)
  states[name]={'minJ_recomputed':float(J.min()),'converged':d['converged'],'fixed_exact':True}
 print(json.dumps({'payload_hashes':len(m['files']),'python_sources_parsed':n,'attachment_tests':test.stdout.strip(),'saved_internal_states':states,'new_numerical_solve':False,'fullhead_pose_claim':False},indent=2))
if __name__=='__main__':main()
