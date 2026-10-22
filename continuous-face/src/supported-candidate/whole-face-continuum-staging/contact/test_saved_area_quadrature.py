"""One frozen-geometry area test; no new broadphase, source motion or adoption."""
from pathlib import Path
import json,time,sys,signal
import numpy as np
from scipy.sparse import csr_matrix
from continuum_contact import sha
from area_quadrature_probe import reference_vertex_areas,evaluate_area_rule
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'mechanics'))
from source_transfer import load_geometry_with_source_fibres


def main():
    root=Path(__file__).resolve().parent;output=root/'repaired-v1-area-quadrature-test.json'
    if output.exists():raise ValueError('one saved-geometry test already exists; do not repeat unchanged')
    signal.alarm(90);started=time.perf_counter()
    domain=root.parent/'geometry/cut-pilot/repaired-v1/cut-continuum.npz';phase=domain.parent/'cut-muscle-phases.npz'
    solid=load_geometry_with_source_fibres(domain,phase)
    boundary=root/'repaired-v1-dyadic-boundary.npz';cache=root/'repaired-v1-broadphase-frozen.npz'
    with np.load(boundary) as z:F,P,W=[z[k] for k in ('triangles','exterior_tet','exterior_barycentric')]
    C=csr_matrix((W.ravel(),(np.repeat(np.arange(len(W)),4),solid.T[P].ravel())),shape=(len(W),len(solid.X)));C.eliminate_zeros();C.sort_indices()
    with np.load(cache) as z:pairs=z['baseline_pairs'];assert str(z['source_sha256'])==sha(domain)
    with np.load(root/'repaired-v1-initial-self-force.npz') as z:old=z['self_gradient_N']
    area=reference_vertex_areas(C,F,solid.X);e,g,details=evaluate_area_rule(C,F,solid.X,area,pairs,coefficient=.001)
    norms=np.linalg.norm(g,axis=1);oldnorm=np.linalg.norm(old,axis=1);gravity=solid.mass*9810.;ratio=norms/gravity;oldratio=oldnorm/gravity
    nodes=set(np.argsort(ratio)[-20:].tolist())|{7550,7518};D=details.pop('active_carrier_difference')
    node_records=[]
    for node in sorted(nodes,key=lambda i:ratio[i],reverse=True):
        col=D[:,node].tocoo();contribution=col.data*details['active_point_force_factor_N'][col.row]
        top=np.argsort(abs(contribution))[-5:][::-1]
        node_records.append(dict(node=int(node),reference_mm=solid.X[node].tolist(),mass_N_s2_per_mm=float(solid.mass[node]),
            unit_pair_force_N=float(oldnorm[node]),unit_pair_force_to_own_gravity=float(oldratio[node]),
            area_rule_force_N=float(norms[node]),area_rule_force_to_own_gravity=float(ratio[node]),
            area_rule_contact_only_F_over_m_mm_s2=float(norms[node]/solid.mass[node]),
            force_reduction_factor=None if norms[node]==0 else float(oldnorm[node]/norms[node]),
            largest_area_quadrature_contributions=[dict(vertex=int(details['active_vertex_ids'][col.row[i]]),
                target_face=int(details['active_target_faces'][col.row[i]]),gap_mm=float(details['active_gap_mm'][col.row[i]]),
                reference_area_mm2=float(details['active_reference_area_mm2_values'][col.row[i]]),
                carrier_difference=float(col.data[i]),contribution_force_norm_N=float(abs(contribution[i]))) for i in top]))
    arrays={k:v for k,v in details.items() if isinstance(v,np.ndarray)}
    report=dict(status='saved_geometry_area_rule_test_only',source_sha256=sha(domain),boundary_sha256=sha(boundary),frozen_pairs_sha256=sha(cache),
        quadrature_module_sha256=sha(root/'area_quadrature_probe.py'),accepted_motion=False,actual_facial_frames=0,new_broadphase_runs=0,
        coefficient_choice='0.001 N/mm^3 is the old numerical pair coefficient divided by an explicitly declared 1 mm^2 reference patch area; not calibrated or tuned to the hotspot',
        force_per_unit_area_coefficient_scale=1000.,details={k:v for k,v in details.items() if k not in arrays},
        energy_N_mm=e,max_node_force_N=float(norms.max()),maximum_force_to_own_gravity=float(ratio.max()),
        nodes_above_gravity={str(c):int(np.sum(ratio>c)) for c in (1,10,100,1000)},
        original_nodes_above_gravity={str(c):int(np.sum(oldratio>c)) for c in (1,10,100,1000)},
        hotspot_node_7550_remains_above_gravity=bool(ratio[7550]>1),node_records=node_records,
        original_finite_pair_geometry_guards_unchanged=True,physical_connectivity_or_local_activation_exemptions_added=False,
        caveat='Point quadrature can under-resolve localized finite triangle contact. A small quadrature force is not automatically a contact or conditioning certificate. F/m is not the constrained coupled acceleration. Native geometry remains failed.',
        seconds=time.perf_counter()-started)
    report['conclusion']='The tested fixed lumped-vertex reference-area rule does not resolve the observed weak-cut force/mass imbalance; this is not a proof about every resolved area integral.' if ratio.max()>1 else 'This declared quadrature reduces the saved force/gravity ratios below1, but resolution and geometry remain unqualified; no adoption.'
    np.savez_compressed(output.with_suffix('.npz'),area_rule_gradient_N=g,reference_vertex_area_mm2=area,source_sha256=np.array(report['source_sha256']),**arrays)
    output.write_text(json.dumps(report,indent=2)+'\n');signal.alarm(0)
    print(json.dumps({k:report[k] for k in ('status','seconds','energy_N_mm','max_node_force_N','maximum_force_to_own_gravity','nodes_above_gravity','hotspot_node_7550_remains_above_gravity','conclusion')},indent=2))

if __name__=='__main__':main()
