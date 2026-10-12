"""Read-only rest/contraction/release result, including exact source pins."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];p=R/'output/bilateral-equilibrium-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['rest-0','right-02-converged','released-0'];states=[json.loads((p/(n+'-state.json')).read_text()) for n in names]
assert all(s['converged'] and s['boundaryFeasibility']['certified'] for s in states)
assert len({s['sourceDomainSha256'] for s in states})==1
motions=[np.load(p/n/'native_motion.npz') for n in ['rest-native','right-02-native','released-native']]
assert all(np.array_equal(motions[0]['vertex_ids'],q['vertex_ids']) for q in motions)
fem=[np.array(s['positions']).reshape(-1,3) for s in states]
result={'schema':'connected-bilateral-native-cycle-v1','unit_note':'Authored design units, not calibrated millimetres','states':{},'static_bone_guard_minimum_separation_design_units':.01,'guard_module_sha256':{n:sha(R/'src/prototypes'/n) for n in ['complete_boundary_contact.mjs','static_bone_swept_bvh.mjs','guarded_static_bone_fem.mjs']},'limits':['Small incoming right-cheek command; left and entire oral volume undriven','Offset bone-origin supports and omitted tissue/cavity clamps remain','Complete static-bone boundary feasibility is distinct from approximate nodal contact forces','No moving-bone CCD, continuous J, tissue self-contact or dense-skin clearance certification','Return error is measured, not exact; rest face remains the stylized native-v16 construction']}
for i,(n,s) in enumerate(zip(names,states)):
 result['states'][n]={'state_sha256':sha(p/(n+'-state.json')),'activation':s['activation'],'residual':s['residual'],'minimumJ':s['minJ'],'maximum_FEM_motion_from_rest':float(np.linalg.norm(fem[i]-fem[0],axis=1).max()*1000),'maximum_native_motion_from_rest':float(np.linalg.norm(motions[i]['displacement']-motions[0]['displacement'],axis=1).max()),'mean_right_fibre_stretch':s['muscleDiagnostics'][0]['meanStretch'],'all_accepted_boundary_paths_certified':bool(s['boundaryFeasibility']['certified'])}
(p/'cycle_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
