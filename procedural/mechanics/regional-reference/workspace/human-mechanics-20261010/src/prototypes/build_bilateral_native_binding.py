#!/usr/bin/env python3
"""Original chart-space binding for a conforming native-v16 midface/oral FEM.

Exact native IDs and subdivision material UVs supply correspondence. No nearest
world-space fit, head copy, mechanical solve or geometry-acceptance claim.
"""
from pathlib import Path
from collections import defaultdict
import argparse, hashlib, json, subprocess, tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT/'inputs/native_midface_patch_v16/native_midface_patch_v16.npz'
ORAL = ROOT/'inputs/native_oral_charts_v16.npz'
HEAD = ROOT.parent/'human-reference-pipeline-20261009/output/native-connected-head-study/shared-head-v16/head.npz'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def barycentric(point, triangle):
    m=np.stack([triangle[1]-triangle[0],triangle[2]-triangle[0]],axis=-1)
    uv=np.linalg.solve(m,point-triangle[0])
    w=np.r_[1-uv.sum(),uv]
    if w.min() < -1e-12 or w.max() > 1+1e-12:
        return None
    w[np.abs(w)<1e-12]=0
    w[np.abs(w-1)<1e-12]=1
    w/=w.sum()
    return w

def support_key(nodes,weights):
    return tuple(sorted((int(n),round(float(w),14)) for n,w in zip(nodes,weights) if w>0))

def boundary_owners(tets):
    faces=np.concatenate([np.delete(tets,k,axis=1) for k in range(4)])
    owners=np.tile(np.arange(len(tets)),4)
    keys=np.sort(faces,axis=1)
    order=np.lexsort((keys[:,2],keys[:,1],keys[:,0]))
    keys,owners=keys[order],owners[order]
    start=np.r_[True,np.any(keys[1:]!=keys[:-1],axis=1)]
    where=np.flatnonzero(start)
    counts=np.diff(np.r_[where,len(keys)])
    if counts.max()>2:
        raise ValueError('Nonmanifold tetrahedron face incidence')
    return {tuple(map(int,keys[i])):int(owners[i]) for i,n in zip(where,counts) if n==1}

def verify_original_transfer(payload,vertices,tets):
    """Exercise the unchanged recovered JavaScript functions at stored rest."""
    rest=vertices*.001
    dm=np.stack([rest[tets[:,k]]-rest[tets[:,0]] for k in [1,2,3]],axis=-1)
    det=np.linalg.det(dm)
    if np.any(det<=0):
        raise ValueError('Stored geometry has nonpositive rest tetrahedra')
    gradients=np.zeros((len(tets),4,3))
    gradients[:,1:]=np.linalg.inv(dm)
    gradients[:,0]=-gradients[:,1:].sum(axis=1)
    material=np.c_[det/6,np.zeros((len(tets),2))]
    # Only volume entries enter recovery; zero coefficients are not a material
    # assignment for a mechanics model. This is an isolated transfer check.
    data={k:payload[k].ravel().tolist() for k in ['points','indices','weights']}
    data.update(rest=rest.ravel().tolist(),nodes=tets.ravel().tolist(),
                gradients=gradients.ravel().tolist(),material=material.ravel().tolist())
    module=(ROOT/'src/affine_surface_transfer.mjs').as_uri()
    code=f"""
import fs from 'node:fs';
import {{recoverSurfaceGradients,applyAffineTransfer}} from {json.dumps(module)};
const a=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const rest=Float64Array.from(a.rest),nodes=Uint32Array.from(a.nodes),
  gradients=Float64Array.from(a.gradients),material=Float64Array.from(a.material),
  points=Float64Array.from(a.points),indices=Uint32Array.from(a.indices),
  weights=Float64Array.from(a.weights),count=rest.length/3;
const identity=new Float64Array(count*9);
for(let n=0;n<count;n++)identity[n*9]=identity[n*9+4]=identity[n*9+8]=1;
const exact=applyAffineTransfer(points,indices,weights,rest,rest,identity);
const recovered=recoverSurfaceGradients(rest,rest,nodes,gradients,material,count);
const actual=applyAffineTransfer(points,indices,weights,rest,rest,recovered);
let exactError=0,recoveredError=0,gradientError=0;
for(let i=0;i<points.length;i++){{exactError=Math.max(exactError,Math.abs(exact[i]-points[i])*1000);recoveredError=Math.max(recoveredError,Math.abs(actual[i]-points[i])*1000);}}
for(let i=0;i<recovered.length;i++)gradientError=Math.max(gradientError,Math.abs(recovered[i]-identity[i]));
if(exactError>1e-10||recoveredError>1e-5)throw Error('Rest-transfer numerical gate failed');
console.log(JSON.stringify({{verifiedVertices:points.length/3,recoveryNodeCount:count,
 identityGradientMaxCoordinateErrorDesignUnits:exactError,
 originalFloat32RecoveredMaxCoordinateErrorDesignUnits:recoveredError,
 originalFloat32RecoveredMaxGradientError:gradientError,
 identityGateDesignUnits:1e-10,originalFloat32GateDesignUnits:1e-5,
 identityExactAlgebraically:true,originalRecoveredOutputByteExact:false}}));
"""
    with tempfile.TemporaryDirectory(prefix='native-binding-check-') as td:
        p=Path(td)/'transfer_input.json'
        p.write_text(json.dumps(data,separators=(',',':')))
        del data
        result=subprocess.run(['node','--input-type=module','-e',code,str(p)],
                              check=True,text=True,capture_output=True)
    return json.loads(result.stdout)

def build(geometry,output):
    geometry,output=Path(geometry),Path(output)
    if output.exists():
        raise FileExistsError('Choose a new binding output directory')
    p,o,g=np.load(PATCH),np.load(ORAL),np.load(geometry)
    if sha(HEAD)!=str(p['source_head_sha256']) or sha(ORAL)!=str(p['source_oral_charts_sha256']):
        raise ValueError('Native source hash differs from patch provenance')
    hv=np.load(HEAD)['vertices']
    vertices,tets=g['vertices'],g['tetrahedra']
    columns=g['surface_column_nodes']
    surface_tri=g['all_surface_triangles']
    surface_parent=g['surface_triangle_parent_faces']
    outer_ids=g['outer_column_native_ids']
    quads=p['patch_quads_local']
    count=len(quads)
    if not np.array_equal(g['all_surface_quads'][:count],quads):
        raise ValueError('Candidate parent quads differ from exact patch')
    if not np.array_equal(vertices[columns[:,0]],hv[outer_ids]):
        raise ValueError('Candidate external samples moved from exact native source')
    oral_grid=g['oral_node_grid']
    if oral_grid.shape!=(120,5,5):
        raise ValueError('This builder requires the shared stride-4 oral wedge')
    if not np.array_equal(oral_grid[:,-1,:],g['midface_column_nodes'][g['oral_interface_columns']]):
        raise ValueError('The oral interface does not genuinely share all depth nodes')
    owners=boundary_owners(tets)
    parent_tri=defaultdict(list)
    for i,parent in enumerate(surface_parent):
        parent_tri[int(parent)].append(i)
    records={}
    duplicate_records=0
    duplicate_seam_records=0
    def put(native_id,nodes,weights,region,parent,uv):
        nonlocal duplicate_records,duplicate_seam_records
        native_id=int(native_id)
        nodes=np.asarray(nodes,dtype=np.int64)
        weights=np.asarray(weights,dtype=float)
        if len(set(nodes.tolist()))!=3 or tuple(sorted(nodes.tolist())) not in owners:
            raise ValueError('Binding support is not an actual volume boundary triangle')
        if weights.min()<0 or abs(weights.sum()-1)>1e-13:
            raise ValueError('Nonpositive or nonpartitioned material-chart weights')
        record=(nodes.copy(),weights.copy(),int(region),int(parent),np.asarray(uv,float),
                owners[tuple(sorted(nodes.tolist()))])
        if native_id in records:
            previous=records[native_id]
            if support_key(previous[0],previous[1])!=support_key(nodes,weights):
                raise ValueError(f'Inconsistent shared-edge support for native vertex {native_id}')
            duplicate_records+=1
            if previous[2]!=region:
                duplicate_seam_records+=1
            return
        records[native_id]=record

    dense_faces=p['dense_native_triangles']
    uv_template=p['parent_subtriangle_corner_uv_template']
    dense_children=p['dense_face_parent_subtriangle_index']
    # Each parent owns exactly 128 native triangles in the verified patch.
    for face in range(count):
        selection=slice(face*128,(face+1)*128)
        native=dense_faces[selection].ravel()
        corners=uv_template[dense_children[selection]].reshape(-1,2)
        uv_map={}
        for nid,uv in zip(native,corners):
            nid=int(nid)
            if nid in uv_map and not np.array_equal(uv_map[nid],uv):
                raise ValueError('A parent material chart assigns inconsistent UVs')
            uv_map[nid]=uv
        supports=[]
        for ti in parent_tri[face]:
            cols=surface_tri[ti]
            try:
                coords=np.array([uv_map[int(outer_ids[n])] for n in cols])
            except KeyError as error:
                raise ValueError('Refined support has no exact native parent UV') from error
            supports.append((columns[cols,0],coords))
        if len(supports) not in (2,3):
            raise ValueError('Unexpected coarse/split parent triangle count')
        for nid,uv in uv_map.items():
            for nodes,coords in supports:
                w=barycentric(uv,coords)
                if w is not None:
                    put(nid,nodes,w,0,face,uv)
                    break
            else:
                raise ValueError('Native material point is outside its exact parent support')

    # Use the actual top-triangle diagonals for BOTH external and mucosal
    # charts. A reversed oral quad can change its diagonal; no guessed split.
    for side,depth,region in [('external',0,1),('mucosal',4,2)]:
        native_rows=o[f'{side}_vertex_ids']
        positions=o[f'{side}_positions']
        if not np.array_equal(hv[native_rows],positions):
            raise ValueError('Oral chart coordinates differ from exact native source')
        for angular in range(120):
            nxt=(angular+1)%120
            for radial in range(4):
                face=count+angular*4+radial
                top_nodes=[oral_grid[angular,radial,0],oral_grid[nxt,radial,0],
                           oral_grid[nxt,radial+1,0],oral_grid[angular,radial+1,0]]
                uv_map=dict(zip(map(int,top_nodes),np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])))
                supports=[]
                for ti in parent_tri[face]:
                    cols=surface_tri[ti]
                    coords=np.array([uv_map[int(columns[col,0])] for col in cols])
                    supports.append((columns[cols,depth],coords))
                rows=range(radial*4,radial*4+5) if radial==3 else range(radial*4,radial*4+4)
                for row in rows:
                    for delta in range(4):
                        angle=angular*4+delta
                        uv=np.array([delta/4,(row-4*radial)/4])
                        for nodes,coords in supports:
                            w=barycentric(uv,coords)
                            if w is not None:
                                put(native_rows[row,angle],nodes,w,region,face,uv)
                                break
                        else:
                            raise ValueError('Oral material point not in actual boundary support')

    expected=np.unique(np.r_[p['dense_native_vertex_ids'],o['external_vertex_ids'].ravel(),o['mucosal_vertex_ids'].ravel()])
    vertex_ids=np.array(sorted(records),dtype=np.int32)
    if not np.array_equal(vertex_ids,expected):
        raise ValueError('Missing or duplicated native material vertex')
    indices=np.array([records[int(i)][0] for i in vertex_ids],dtype=np.uint32)
    weights=np.array([records[int(i)][1] for i in vertex_ids])
    points=hv[vertex_ids]*.001
    rest=vertices*.001
    offsets=points-(rest[indices]*weights[:,:,None]).sum(axis=1)
    analytic=(rest[indices]+points[:,None,:]-rest[indices])*weights[:,:,None]
    reconstructed=analytic.sum(axis=1)/weights.sum(axis=1)[:,None]
    analytic_error=np.max(np.abs(reconstructed-points))*1000
    if analytic_error>1e-10:
        raise ValueError('Identity rest reconstruction failed')
    regions=np.array([records[int(i)][2] for i in vertex_ids],dtype=np.uint8)
    payload={'vertex_ids':vertex_ids,'points':points,'indices':indices,'weights':weights,
             'rest_offsets':offsets,'regions':regions,
             'parent_faces':np.array([records[int(i)][3] for i in vertex_ids],dtype=np.int32),
             'parent_uv':np.array([records[int(i)][4] for i in vertex_ids]),
             'boundary_owner_tetrahedra':np.array([records[int(i)][5] for i in vertex_ids],dtype=np.int32),
             'nodes_per_vertex':np.array(3,dtype=np.int32),'required_gradient_recovery_count':np.array(len(vertices),dtype=np.int32),
             'solver_units_per_design_unit':np.array(.001),
             'source_geometry_sha256':np.array(sha(geometry)),
             'source_head_sha256':p['source_head_sha256'],
             'source_patch_sha256':np.array(sha(PATCH)),
             'source_oral_charts_sha256':np.array(sha(ORAL))}
    cut_rows=[]
    for name in ['outer_cut','right_eye_cut','left_eye_cut','right_nostril_cut','left_nostril_cut']:
        ids=p[f'{name}_dense_native_ids']
        rows=np.searchsorted(vertex_ids,ids)
        if not np.array_equal(vertex_ids[rows],ids):
            raise ValueError('Protected cut is absent from binding')
        payload[f'{name}_binding_rows']=rows.astype(np.int32)
        cut_rows.extend(rows)
    mucosal_cut=np.searchsorted(vertex_ids,o['mucosal_vertex_ids'][16])
    payload['deep_mucosal_cut_binding_rows']=mucosal_cut.astype(np.int32)
    cut_rows=np.unique(np.r_[cut_rows,mucosal_cut]).astype(np.int32)
    payload['unmatched_cut_binding_rows']=cut_rows
    cut_nodes=np.unique(indices[cut_rows][weights[cut_rows]>0]).astype(np.uint32)
    payload['unmatched_cut_support_node_ids']=cut_nodes
    # One conservative way to ensure each cut support's recovered F is identity
    # is to freeze the complete incident tet star. This is provenance, not an
    # applied or recommended final anatomical boundary condition.
    payload['cut_recovery_incident_tetrahedra']=np.flatnonzero(np.isin(tets,cut_nodes).any(axis=1)).astype(np.int32)
    payload['cut_recovery_incident_node_ids']=np.unique(tets[payload['cut_recovery_incident_tetrahedra']]).astype(np.uint32)
    payload['oral_shared_interface_node_ids']=oral_grid[:,-1,:].astype(np.uint32)
    transfer=verify_original_transfer(payload,vertices,tets)
    output.mkdir(parents=True)
    target=output/'native_surface_binding.npz'
    np.savez_compressed(target,**payload)
    report={'schema':'original-native-v16-bilateral-material-chart-binding-v1',
        'status':'PASS correspondence/rest transfer only; volume intersection/contact acceptance is still pending',
        'source_geometry':str(geometry.resolve()),'source_geometry_sha256':sha(geometry),
        'source_head_sha256':str(p['source_head_sha256']),'source_patch_sha256':sha(PATCH),
        'source_oral_charts_sha256':sha(ORAL),'builder_sha256':sha(__file__),
        'output_file':target.name,'output_sha256':sha(target),'output_bytes':target.stat().st_size,
        'bound_native_vertices':len(vertex_ids),'unique_native_vertices':len(np.unique(vertex_ids)),
        'region_assignment_counts':{str(i):int(np.sum(regions==i)) for i in range(3)},
        'region_names':{'0':'midface exterior, including shared vermilion border','1':'oral exterior, excluding prior shared IDs','2':'oral mucosa, excluding shared wetline IDs'},
        'duplicate_chart_records_consistency_checked':duplicate_records,'cross_region_duplicate_records_consistency_checked':duplicate_seam_records,
        'shared_oral_interface_nodes':int(len(np.unique(oral_grid[:,-1,:]))),
        'minimum_weight':float(weights.min()),'maximum_weight':float(weights.max()),
        'maximum_partition_error':float(np.max(np.abs(weights.sum(axis=1)-1))),
        'maximum_rest_offset_design_units':float(np.linalg.norm(offsets,axis=1).max()*1000),
        'rest_offset_quantiles_design_units':np.quantile(np.linalg.norm(offsets,axis=1)*1000,[0,.5,.95,.99,1]).tolist(),
        'all_support_triangles_actual_volume_boundary':True,'nearest_world_fit_used':False,
        'rest_transfer_verification':transfer,
        'unmatched_cut_native_vertices':len(cut_rows),'unmatched_cut_support_nodes':len(cut_nodes),
        'compatibility':{'applyAffineTransfer':'Flatten points, indices and weights in C order; x/rest must use solver units (.001*design coordinates); nodesPerVertex=3.',
            'recoverSurfaceGradients':f'Use count={len(vertices)} with ORIGINAL global node ordering. Mucosal support is not a prefix of exterior nodes. Supply actual volume gradients and material volumes from this exact geometry.',
            'rest_residual':'The original formula transports q-X_i separately per support, not one untransformed additive offset. rest_offsets is diagnostic.',
            'storage':'NPZ numeric arrays plus this JSON manifest. No full head is copied.'},
        'cut_semantics':{'outer_cut':'Beyond the shared head rectangle is omitted.',
            'eye_cuts':'Inner lids and sockets remain unbound.',
            'nostril_cuts':'Native vestibular walls/caps remain unbound.',
            'deep_mucosal_cut':'Native oral cavity beyond row16 remains unbound.',
            'necessary_runtime_constraint':'The omitted side remains at rest. Bound dense cut vertices must also remain exactly at rest. Pinning only support positions is insufficient when recovered deformation gradients transport a nonzero residual. Either enforce a verified rest-frozen incident collar/gradient rule, or author coherent continuation. Stored incident stars are provenance and are not applied constraints.'},
        'limitations':['Binding does not establish a nonintersecting volume, support clearance or valid active motion.',
            'No full-head motion acceptance until omitted-boundary displacement and tangent/clearance gates pass.',
            'Original recovered Float32 gradient accumulation is not byte-exact identity; its measured error is separately reported.',
            'Anatomical and material parameters remain authored approximations, not validated biology.'],
        'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in payload.items()}}
    (output/'binding_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','bound_native_vertices','shared_oral_interface_nodes','minimum_weight','maximum_partition_error','maximum_rest_offset_design_units','rest_transfer_verification','output_file','output_sha256','output_bytes']},indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,default=ROOT/'output/bilateral-midface-fitted-v2/rest_geometry.npz')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.geometry,args.output)
