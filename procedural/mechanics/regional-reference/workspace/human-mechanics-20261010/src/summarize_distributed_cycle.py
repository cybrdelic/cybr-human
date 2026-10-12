"""Exact state-specific local volume and signed native response summary."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(case,states):
    d=json.loads((case/'domain.json').read_text());raw=(case/'arrays.bin').read_bytes()
    a={k:np.frombuffer(raw,dtype=q['dtype'],count=np.prod(q['shape']),offset=q['byte_offset']).reshape(q['shape']) for k,q in d['arrays'].items()}
    t=a['nodes'];V=a['material'][:,0];rest=np.load(states/'rest-native/native_motion.npz');rows=[]
    for phase,folder in [('rest-0','rest-native'),('right-02','right-02-native'),('right-08','right-08-native'),('right-16','right-16-native'),('released-0','released-native')]:
        p=states/f'{phase}-state.json';s=json.loads(p.read_text())
        if not s['converged'] or s['sourceDomainSha256']!=sha(case/'domain.json'):raise ValueError('Unconverged or unpinned cycle state')
        x=np.asarray(s['positions']).reshape(-1,3);dm=np.stack([x[t[:,i]]-x[t[:,0]] for i in (1,2,3)],-1);J=np.linalg.det(dm)/(6*V)
        if abs(float(J.min())-s['minJ'])>1e-10:raise ValueError('Independent J reconstruction mismatch')
        motion=states/folder/'native_motion.npz';m=np.load(motion)
        if not np.array_equal(m['vertex_ids'],rest['vertex_ids']):raise ValueError('Cycle binding changed')
        u=m['displacement']-rest['displacement'];corners=[]
        for label,id in [('right',25600),('left',25471)]:
            k=np.flatnonzero(m['vertex_ids']==id)[0];corners.append(dict(side=label,native_vertex_id=id,vector_xyz=u[k].tolist(),magnitude=float(np.linalg.norm(u[k]))))
        k=int(J.argmin());h=int(J.argmax())
        rows.append(dict(phase=phase,state_sha256=sha(p),native_motion_sha256=sha(motion),
           activation=s['activation'],residual=s['residual'],energy=s['energy'],
           minimum_J=float(J[k]),maximum_J=float(J[h]),minimum_J_tet=k,maximum_J_tet=h,
           total_volume_ratio=float(np.sum(J*V)/np.sum(V)),
           minimum_J_rest_center_design_units=(a['rest'][t[k]].mean(0)*1000).tolist(),
           maximum_native_motion_from_rest=float(np.linalg.norm(u,axis=1).max()),
           maximum_FEM_motion_from_reference=float(np.linalg.norm(x-a['rest'],axis=1).max()*1000),
           corners=corners,certified_boundary_paths=s['boundaryFeasibility']['certified'],
           muscle_diagnostics=s['muscleDiagnostics']))
    report=dict(schema='distributed-root-contraction-release-v1',domain_sha256=sha(case/'domain.json'),states=rows,
      coordinate_convention='X lateral, Y superior, Z anterior; all length values are authored design units, not calibrated millimetres',
      limits=['Local J range is reported per state; total volume cannot hide local compression or expansion',
       'Poor bridge conditioning remains; convergence does not establish mesh independence or biological accuracy',
       'Rigid first collars transmit load to exact bone roots; all continuation and oral commands remain zero',
       'Static-bone boundary feasibility is distinct from the approximate nodal force model and dense skin self-contact',
       'Released state is a numerical return with explicit error, not exact return'])
    (states/'cycle_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:r[k] for k in ['phase','residual','minimum_J','maximum_J','maximum_native_motion_from_rest']} for r in rows],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('states',type=Path);a=p.parse_args();run(a.case,a.states)
