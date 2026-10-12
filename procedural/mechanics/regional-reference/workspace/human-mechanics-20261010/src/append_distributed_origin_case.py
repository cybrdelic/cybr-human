"""Join exact-source bone roots to the preserved tissue through authored volumes.

Source-only assembler. It does not accept an attachment merely because this
program can append it: independent conditioning/intersection/contact gates are
required before a solved state may be selected.
"""
from pathlib import Path
import argparse, copy, json, os
import numpy as np
from build_bilateral_cavity_case import read_arrays, write_arrays, sha
from build_bilateral_mechanics_case import boundary, surface_audit

def build(base, candidates, out):
    base, candidates, out = map(Path.resolve, (base, candidates, out))
    d=json.loads((base/'domain.json').read_text()); old=read_arrays(base,d)
    manifest=json.loads((candidates/'manifest.json').read_text())
    if manifest['combined_bone_binary_sha256']!=sha(base/d['bones']['binary']): raise ValueError('Frozen attachment bone pin differs')
    bv=np.fromfile(base/d['bones']['binary'],dtype='<f8',count=d['bones']['vertex_count']*3).reshape(-1,3)
    nv=d['bones']['vertex_count']//2; nf=d['bones']['triangle_count']//2
    vertices=[old['rest']]; cells=[old['nodes']]; offsets=[]; interfaces=[]; muscles=copy.deepcopy(d['muscles'])
    N=len(old['rest']); T=len(old['nodes']); all_root=[]; all_collar=[]
    base_boundary=np.asarray(d['boundary_triangles']);base_face_map={tuple(sorted(map(int,f))):i for i,f in enumerate(base_boundary)}
    for side,label in [(1,'right'),(-1,'left')]:
        entry=next(q for q in manifest['sides'] if q['side']==side)
        p=candidates/entry['array_file']
        if sha(p)!=entry['sha256']: raise ValueError('Frozen candidate pin differs')
        a=np.load(p)
        v=a['vertices']*.001; t=a['tetrahedra']; roots=np.arange(len(a['root_source_bone_vertex_ids']))
        if not np.array_equal(roots,a['root_local_node_ids']):
            raise ValueError('Root prefix contract changed')
        shared=a['shared_tissue_local_node_ids']; target=a['shared_tissue_source_node_ids']
        if len(np.unique(shared))!=len(shared) or len(np.unique(target))!=len(target): raise ValueError('Repeated shared node')
        if not np.array_equal(v[shared],old['rest'][target]): raise ValueError('Shared tissue coordinates moved')
        root_bone=a['root_source_bone_vertex_ids']+(nv if side<0 else 0)
        if not np.array_equal(v[roots],bv[root_bone]): raise ValueError('Root is not bit-identical actual bone')
        new=np.setdiff1d(np.arange(len(v)),shared); mapping=np.full(len(v),-1,dtype=np.int64)
        mapping[shared]=target; mapping[new]=N+np.arange(len(new)); mapped=mapping[t]
        vertices.append(v[new]); cells.append(mapped); all_root.extend(mapping[roots]); all_collar.extend(mapping[a['fixed_local_nodes']])
        # The connector is passive, direction-bearing tissue. It is not silently
        # driven by the incoming cheek's activation command.
        region=np.asarray(a['region_tet_offsets']); deformable=np.arange(int(region[1]),len(t))
        direction=a['material_space_path_directions'][deformable]
        if np.any(a['activation']!=0): raise ValueError('Attachment must remain passive')
        if not np.all(a['shared_tissue_owner_is_active_body']): raise ValueError('Tip does not enter incoming active body')
        if not np.isfinite(direction).all() or np.any(np.linalg.norm(direction,axis=1)<1e-10): raise ValueError('Invalid connector fibre')
        direction=direction/np.linalg.norm(direction,axis=1)[:,None]
        muscles.append(dict(id=f'{label}-passive-distributed-origin-continuation',
            tetrahedra=(T+deformable).tolist(),fibreDirections=direction.tolist(),
            stiffness=24000.,activation=0.,volumeFraction=1.,
            scope='Authored passive anisotropic origin continuation; not a calibrated tendon or newly activated muscle'))
        tip=a['boundary_faces'][np.isin(a['boundary_faces'],shared).all(1)]
        if len(tip)!=len(a['shared_tissue_source_face_ids']): raise ValueError('Shared tip face count differs')
        mapped_tip=mapping[tip]
        current_face_ids=[base_face_map[tuple(sorted(map(int,f)))] for f in mapped_tip]
        interfaces.append(dict(side=side,source_candidate_sha256=sha(p),
            root_nodes=mapping[roots].tolist(),root_combined_bone_vertex_ids=root_bone.tolist(),
            root_face_nodes=mapping[a['root_faces']].tolist(),
            root_combined_bone_triangle_ids=(a['root_source_bone_triangle_ids']+(nf if side<0 else 0)).tolist(),
            fixed_collar_nodes=mapping[a['fixed_local_nodes']].tolist(),
            shared_tissue_nodes=target.tolist(),
            shared_tissue_face_nodes=mapped_tip.tolist(),shared_tissue_parent_case_face_ids=current_face_ids,
            shared_tissue_owner_tet_ids=a['shared_tissue_owner_tet_ids'].tolist(),
            original_pre_cavity_source_face_ids=a['shared_tissue_source_face_ids'].tolist(),
            appended_tet_range=[T,T+len(t)],region_tet_offsets=(T+region).tolist()))
        offsets.append(mapping);N+=len(new);T+=len(t)
    x=np.concatenate(vertices);t=np.concatenate(cells); newt=t[len(old['nodes']):]
    dm=np.stack([x[newt[:,i]]-x[newt[:,0]] for i in (1,2,3)],-1);det=np.linalg.det(dm)
    if det.min()<=0: raise ValueError('Nonpositive connector cell')
    grad=np.zeros((len(newt),4,3));grad[:,1:]=np.linalg.inv(dm);grad[:,0]=-grad[:,1:].sum(1)
    vol=det/6;E=3200.;nu=.43;mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))+mu
    mat=np.r_[old['material'],np.c_[vol,np.full(len(vol),mu),np.full(len(vol),lam)]]
    mass=np.r_[old['mass'],np.zeros(len(x)-len(old['rest']))]
    for j in range(4):np.add.at(mass,newt[:,j],1000*vol/4)
    arrays=dict(rest=x.astype('<f8'),nodes=t.astype('<u4'),gradients=np.r_[old['gradients'],grad].astype('<f8'),material=mat.astype('<f8'),mass=mass.astype('<f8'))
    bf=boundary(t);audit=surface_audit(bf)
    if audit['open_or_nonmanifold_edges'] or audit['inconsistent_winding_edges'] or audit['nonmanifold_vertex_count']:
        raise ValueError('Appended assembly has invalid boundary')
    # Only the old offset-origin pins are superseded. The exact cut-gradient
    # star is re-derived so a shared omitted-interface node cannot be released.
    cuts=np.asarray(d['frozen_transfer_gradient_nodes']); cut_star=np.unique(t[np.isin(t,cuts).any(1)])
    old_origins=np.unique(np.concatenate([q['nodes'] for q in d['bone_origin_supports']]))
    fixed=np.unique(np.r_[np.setdiff1d(d['fixed_nodes'],old_origins),cut_star,all_collar])
    internal_shared=np.unique(np.concatenate([q['shared_tissue_nodes'] for q in interfaces]))
    springs=np.setdiff1d(d['omitted_deep_tissue_spring_nodes'],np.r_[internal_shared,fixed])
    out.mkdir(parents=True,exist_ok=False);contract=write_arrays(out,arrays);os.link(base/d['bones']['binary'],out/'bones.bin')
    np.savez_compressed(out/'rest_geometry.npz',vertices=x*1000,tetrahedra=t,
        boundary_faces=bf,parent_tissue_node_count=len(old['rest']),parent_tissue_tet_count=len(old['nodes']))
    new=copy.deepcopy(d);new.update(model=dict(schema='cybr-layered-fem-v1',nodes=len(x),tetrahedra=len(t)),arrays=contract,
       fixed_nodes=fixed.tolist(),boundary_triangles=bf.tolist(),muscles=muscles,
       omitted_deep_tissue_spring_nodes=springs.tolist(),bone_origin_supports=[],
       superseded_offset_origin_supports=d['bone_origin_supports'],distributed_origin_attachments=interfaces,
       immutable_bonded_root_nodes=sorted(map(int,all_root)),parent_case_domain_sha256=sha(base/'domain.json'),
       parent_tissue_geometry_sha256=d['source_geometry_sha256'],source_geometry_sha256=sha(out/'rest_geometry.npz'),
       status='Distributed-origin input assembly; final overlap/quality and guarded solve acceptance still required')
    new['limits']=[z for z in new['limits'] if 'Rigid origin supports retain' not in z and 'Original finite-offset origin pins remain' not in z]
    new['limits'] += ['Exact-source bone root and first finite collar are rigid; connector bulk is passive and deformable',
      'Attachment Dm condition reaches 2618; this is a warned numerical stability experiment, not a well-conditioned or calibrated anatomical model',
      'Only exact immutable root nodes omit repulsive nodal contact; all deformable nodes retain it',
      'Every unbonded boundary pair requires complete continuous static-bone clearance certification']
    (out/'domain.json').write_text(json.dumps(new,separators=(',',':'))+'\n')
    report=dict(schema='distributed-origin-append-v1',domain_sha256=sha(out/'domain.json'),
       arrays_sha256=sha(out/'arrays.bin'),parent_domain_sha256=sha(base/'domain.json'),
       original_tissue_coordinates_cells_material_exact=True,nodes=len(x),tetrahedra=len(t),
       boundary=audit,fixed_nodes=len(fixed),superseded_offset_origin_pins=old_origins.tolist(),
       old_origin_pins_retained_only_as_other_cut_constraints=np.intersect1d(old_origins,cut_star).tolist(),
       removed_shared_interface_springs=np.intersect1d(d['omitted_deep_tissue_spring_nodes'],internal_shared).tolist(),
       appended_min_volume=float(vol.min()),appended_max_Dm_condition=float(np.linalg.cond(dm).max()))
    (out/'construction_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.base,a.candidates,a.output)
