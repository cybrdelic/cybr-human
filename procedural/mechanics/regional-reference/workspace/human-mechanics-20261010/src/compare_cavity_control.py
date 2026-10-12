"""Compare exact same-command native motion and support resultants.
The combined cavity/support revision is not a cavity-only causal ablation.
"""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[R/'output/bilateral-equilibrium-v1',R/'output/bilateral-cavity-equilibrium-v1']
proposal=json.loads((R/'output/passive-commissure-proposal-v1/proposal.json').read_text())
records=[]
for i,p in enumerate(paths):
    state=p/('right-02-converged-state.json' if i==0 else 'right-02-state.json')
    a=np.load(p/'right-02-native/native_motion.npz');b=np.load(p/'rest-native/native_motion.npz')
    if not np.array_equal(a['vertex_ids'],b['vertex_ids']):raise ValueError('Rest and active binding mismatch')
    ids=a['vertex_ids'];v=a['reference_positions'];u=a['displacement']-b['displacement'];lookup={int(k):j for j,k in enumerate(ids)}
    corners=[]
    for q in proposal['landmarks']:
        k=lookup[int(q['native_vertex_id'])];corners.append(dict(side=q['side'],native_vertex_id=int(q['native_vertex_id']),
            rest_position_xyz=v[k].tolist(),displacement_xyz=u[k].tolist(),magnitude=float(np.linalg.norm(u[k]))))
    reaction=json.loads((p/'reaction_accounting.json').read_text());s=json.loads(state.read_text())
    records.append(dict(label='original row16 / no passive oral fibres' if i==0 else 'row24 / passive oral fibres',
        source_state_sha256=sha(state),source_motion_sha256=sha(p/'right-02-native/native_motion.npz'),
        activation=s['activation'],converged=s['converged'],residual=s['residual'],minJ=s['minJ'],
        maximum_native_motion=float(np.linalg.norm(u,axis=1).max()),corners=corners,
        support_vectors_xyz=reaction['vectors_xyz'],support_vector_magnitudes=reaction['vector_magnitudes'],
        balance_plus_free_gradient_error=reaction['balance_plus_free_gradient_error']))
changes=[]
for a,b in zip(records[0]['corners'],records[1]['corners']):
    changes.append(dict(side=a['side'],displacement_change_xyz=(np.array(b['displacement_xyz'])-a['displacement_xyz']).tolist(),
        magnitude_ratio=b['magnitude']/a['magnitude'] if a['magnitude'] else None))
report=dict(schema='same-command-cavity-passive-support-comparison-v1',coordinate_convention='X lateral, Y superior, Z anterior',
    design_units='Authored uncalibrated units; vectors are not exaggerated',records=records,corner_changes=changes,
    limits=['Both deeper cut and passive fibre support changed; this pair cannot isolate cavity-only causality',
            'Both cases retain the earlier finite-offset bone origin pins',
            'Force resultant magnitudes can cancel and are not additive load fractions',
            'Positive J and static-bone boundary certification do not validate anatomy or full dense-skin contact'])
out=paths[1]/'same_command_control_comparison.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
