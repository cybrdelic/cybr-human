#!/usr/bin/env python3
"""Replay parity-based authoring into a temporary directory; never rewrite fixtures."""
import contextlib,hashlib,io,itertools,json,sys,tempfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/prototypes'))
import build_revised_distributed_attachment as replay
import build_distributed_fixed_collar as collar
P=ROOT/'output/distributed-origin-attachment-feasibility-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshots():return {str(p.relative_to(P)):sha(p) for sub in ['frozen','revised'] for p in sorted((P/sub).rglob('*')) if p.is_file()}
start=time.monotonic();before=snapshots();results=[]
# All six ID permutations: declared forward winding succeeds; reverse rejects.
x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
for q in itertools.permutations(range(3)):
 odd=sum(q[i]>q[j] for i,j in ((0,1),(0,2),(1,2)))%2
 rejected=False
 try:replay.layer_tets(x,np.array([q]),{0:3,1:4,2:5})
 except AssertionError:rejected=True
 assert rejected==bool(odd)
# A folded/negative layer is rejected rather than repaired.
try:replay.layer_tets(x*np.array([1,1,-1]),np.array([[0,1,2]]),{0:3,1:4,2:5})
except AssertionError:pass
else:raise AssertionError('Negative prism was not rejected')
with tempfile.TemporaryDirectory(prefix='distributed-parity-replay-') as tmp:
 replay.OUT=Path(tmp)
 with contextlib.redirect_stdout(io.StringIO()):replay.main()
 for name in ['right','left']:
  fresh=np.load(replay.OUT/(name+'_attachment_candidate.npz'));frozen=np.load(P/'frozen'/(name+'_attachment.npz'));prior=np.load(P/'revised'/(name+'_attachment_candidate.npz'))
  assert set(fresh.files)==set(prior.files)
  for key in fresh.files:
   assert fresh[key].dtype==frozen[key].dtype==prior[key].dtype,key
   assert fresh[key].shape==frozen[key].shape==prior[key].shape,key
   assert fresh[key].tobytes()==frozen[key].tobytes()==prior[key].tobytes(),name+':'+key
  results.append(dict(side=name,original_array_fields_bitwise_identical=sorted(fresh.files),nodes=len(fresh['vertices']),tets=len(fresh['tetrahedra']),frozen_file_sha256=sha(P/'frozen'/(name+'_attachment.npz'))))
 # The earlier independent 297-face fixed-collar fixtures use the same helper.
 bone=np.load(ROOT/'output/authored-bone-support-v1/bone_support.npz')
 land=json.load(open(ROOT/'output/authored-bone-support-v1/landmarks.json'))
 patch=np.array(land['zygomaticus_major_origin_patch_center']['attachment_patch_triangles'])
 for side,name in [(1,'right'),(-1,'left')]:
  xx,tt,rf,ids=collar.build(side,bone['vertices'],bone['triangles'],patch)
  prior=np.load(P/(name+'_fixed_collar.npz'))
  for k,value in [('vertices',xx),('tetrahedra',tt),('root_faces',rf),('root_source_bone_vertex_ids',ids)]:assert value.tobytes()==prior[k].tobytes(),name+':'+k
assert snapshots()==before,'A pinned original output changed'
receipt=dict(status='pass',elapsed_seconds=time.monotonic()-start,orientation='Declared face permutation parity only; determinants reject without repairing.',full_replay='Original bone/tissue inputs through collar, projected caps, visibility bridge, tetrahedra and boundary extraction. Temporary outputs deleted after comparison.',negative_winding_tests='All six triangle ID permutations and one negative layer obey declared winding; no sign repair.',earlier_297_face_collar_arrays_bitwise_identical=True,all_pinned_original_and_frozen_file_hashes_unchanged=True,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/prototypes/build_revised_distributed_attachment.py',ROOT/'src/prototypes/build_distributed_fixed_collar.py']},pinned_file_sha256=before,sides=results)
(P/'parity_replay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
