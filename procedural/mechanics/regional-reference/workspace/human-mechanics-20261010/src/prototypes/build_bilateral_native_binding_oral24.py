#!/usr/bin/env python3
"""Exact native-v16 chart binding for the row24 extended oral volume.

Exact native IDs and subdivision material UVs supply correspondence. No nearest
world-space fit, head copy, mechanical solve or geometry-acceptance claim.
"""
from pathlib import Path
from collections import defaultdict
import argparse, hashlib, json, subprocess, tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT/'inputs/native_midface_patch_v16/native_midface_patch_v16.npz'
ORIGINAL_ORAL = ROOT/'inputs/native_oral_charts_v16.npz'
ORAL = ROOT/'inputs/native_oral_charts_v16_mucosa24.npz'
HEAD = ROOT.parent/'human-reference-pipeline-20261009/output/native-connected-head-study/shared-head-v16/head.npz'
HEAD_SHA256 = 'f4fa73042fab1534b45c0982b47ae668b16eb990c2029e6a06cfc9b94b01df14'
ORAL_SHA256 = 'ef348176e1402a0afb3bd57b178323eb1e48b1a74919212d70ba85364c346e02'
EXTERNAL_ROWS = np.array([0,4,8,12,16],dtype=np.int32)
MUCOSAL_ROWS = np.array([0,6,12,18,24],dtype=np.int32)
EXPECTED_NATIVE_VERTICES = 87444+3840

def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def material_samples(radial,coarse_rows,angular):
    """Native integer samples and exact UVs, unique across radial cells.

    The terminal row belongs to the last cell. Angular endpoint samples belong
    to the next cell, including the periodic 479-to-0 seam. The same material
    fraction describes exterior stride4 and mucosal stride6 without guessing
    correspondence from world positions.
    """
    if not 0<=angular<120 or not 0<=radial<4:
        raise ValueError('Invalid exact oral material cell')
    start,end=int(coarse_rows[radial]),int(coarse_rows[radial+1])
    if end<=start:
        raise ValueError('Invalid exact oral material cell')
    for row in range(start,end+int(radial==3)):
        for delta in range(4):
            yield row,4*angular+delta,np.array([delta/4,(row-start)/(end-start)])

def binding_rows(vertex_ids,ids):
    rows=np.searchsorted(vertex_ids,ids)
    if np.any(rows>=len(vertex_ids)) or not np.array_equal(vertex_ids[rows],ids):
        raise ValueError('Declared native cut is absent from the exact binding')
    return rows.astype(np.int32)

def recovery_star(tets,indices,weights,cut_rows):
    """All and only incident tets of positive-weight cut support nodes."""
    cut_nodes=np.unique(indices[cut_rows][weights[cut_rows]>0]).astype(np.uint32)
    incident=np.flatnonzero(np.isin(tets,cut_nodes).any(axis=1)).astype(np.int32)
    nodes=np.unique(tets[incident]).astype(np.uint32)
    if not np.isin(cut_nodes,nodes).all():
        raise ValueError('An unmatched-cut support node has no incident tetrahedron')
    return cut_nodes,incident,nodes

def ring_edges(ids):
    return np.sort(np.c_[ids,np.roll(ids,-1)],axis=1)

def boundary_edges(triangles):
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],
                                  triangles[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    if np.any(counts>2):
        raise ValueError('Native oral chart has a nonmanifold edge')
    return unique[counts==1]

def sorted_edges(edges):
    return edges[np.lexsort((edges[:,1],edges[:,0]))]

def verify_native_chart_topology(o):
    expected=np.r_[ring_edges(o['external_vertex_ids'][-1]),
                   ring_edges(o['mucosal_vertex_ids'][-1])]
    if not np.array_equal(boundary_edges(o['native_surface_triangles']),sorted_edges(expected)):
        raise ValueError('Native oral surface has unexpected unmatched boundaries')
    if not np.array_equal(boundary_edges(o['cap_native_triangles']),
                          sorted_edges(ring_edges(o['mucosal_vertex_ids'][24]))):
        raise ValueError('Row24 is not the exact remaining native cap boundary')
    interior=np.setdiff1d(o['cap_native_vertex_ids'],o['mucosal_vertex_ids'][24])
    if len(interior)!=1681 or not np.array_equal(interior,np.sort(o['cap_interior_native_vertex_ids'])):
        raise ValueError('Remaining native cavity cap identity differs')
    return interior

def bind_explicit_mucosal_facets(native_rows,triangles,material_uv,parent_faces,put):
    """Bind an independently meshed native mucosal boundary in material space.

    Each facet has actual volume-node IDs and per-corner integer [angle,row]
    coordinates. At the periodic seam, use angle480 for native angle0. This
    contract also supports an annular extension behind the original oral16
    volume, without pretending its boundary follows exterior depth columns.
    `put` owns actual-boundary, native-ID, and duplicate-support validation.
    Stored UVs are global chart coordinates [angle/480,row/(row_count-1)].
    """
    triangles=np.asarray(triangles)
    material_uv=np.asarray(material_uv)
    parent_faces=np.asarray(parent_faces)
    if (native_rows.ndim!=2 or native_rows.shape[1]!=480 or triangles.ndim!=2 or
        triangles.shape[1]!=3 or material_uv.shape!=(len(triangles),3,2) or
        parent_faces.shape!=(len(triangles),) or not np.isfinite(material_uv).all() or
        not np.array_equal(material_uv,np.rint(material_uv))):
        raise ValueError('Invalid exact native mucosal facet contract')
    if (np.any(material_uv[:,:,0]<0) or np.any(material_uv[:,:,0]>480) or
        np.any(material_uv[:,:,1]<0) or np.any(material_uv[:,:,1]>=len(native_rows))):
        raise ValueError('Mucosal material support is outside the native chart')
    for nodes,coords,parent in zip(triangles,material_uv,parent_faces):
        coords=np.asarray(coords,dtype=float)
        if np.ptp(coords[:,0])>=240:
            raise ValueError('Mucosal facet crosses the periodic seam without unwrapping')
        if abs(np.linalg.det((coords[1:]-coords[0]).T))<1e-12:
            raise ValueError('Degenerate native material facet')
        lo,hi=coords.min(axis=0).astype(int),coords.max(axis=0).astype(int)
        for row in range(lo[1],hi[1]+1):
            for angle in range(lo[0],hi[0]+1):
                weights=barycentric(np.array([angle,row],dtype=float),coords)
                if weights is None:
                    continue
                uv=np.array([(angle%480)/480,row/(len(native_rows)-1)])
                put(native_rows[row,angle%480],nodes,weights,2,parent,uv)

def verify_explicit_mucosal_corners(native_rows,triangles,material_uv,vertices,hv):
    """Native identity is given by integer material coordinates, never XYZ fit."""
    coords=np.asarray(material_uv)
    if not np.isfinite(coords).all() or not np.array_equal(coords,np.rint(coords)):
        raise ValueError('Mucosal facet corners are not exact native material samples')
    coords=coords.astype(np.int64)
    if (np.any(coords[:,:,0]<0) or np.any(coords[:,:,0]>480) or
        np.any(coords[:,:,1]<0) or np.any(coords[:,:,1]>=len(native_rows))):
        raise ValueError('Mucosal facet corner lies outside the native chart')
    ids=native_rows[coords[:,:,1],coords[:,:,0]%480]
    error=float(np.max(np.abs(vertices[triangles]-hv[ids])))
    if error>1e-10:
        raise ValueError('Mucosal support corners differ from their native coordinates beyond roundoff')
    return error

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
    """Exercise the unchanged original JavaScript formula and cut collar.

    The binary fixture avoids large JSON copies. The collar probe is a
    synthetic transfer check, not a mechanics solve or an accepted motion.
    """
    rest=vertices*.001
    dm=np.stack([rest[tets[:,k]]-rest[tets[:,0]] for k in [1,2,3]],axis=-1)
    det=np.linalg.det(dm)
    if np.any(det<=0):
        raise ValueError('Stored geometry has nonpositive rest tetrahedra')
    gradients=np.zeros((len(tets),4,3))
    gradients[:,1:]=np.linalg.inv(dm)
    gradients[:,0]=-gradients[:,1:].sum(axis=1)
    material=np.c_[det/6,np.zeros((len(tets),2))]
    arrays={k:np.asarray(payload[k],dtype='<f8' if k in ['points','weights'] else '<u4')
            for k in ['points','indices','weights','unmatched_cut_binding_rows',
                      'unmatched_cut_support_node_ids','cut_recovery_incident_node_ids']}
    arrays.update(rest=np.asarray(rest,dtype='<f8'),nodes=np.asarray(tets,dtype='<u4'),
                  gradients=np.asarray(gradients,dtype='<f8'),material=np.asarray(material,dtype='<f8'))
    module=(ROOT/'src/affine_surface_transfer.mjs').as_uri()
    code=f"""
import fs from 'node:fs';
import {{recoverSurfaceGradients,applyAffineTransfer}} from {json.dumps(module)};
const contract=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const raw=fs.readFileSync(process.argv[2]),buffer=raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength),a={{}};
for(const [name,q] of Object.entries(contract)){{const T=q.dtype==='<f8'?Float64Array:Uint32Array;a[name]=new T(buffer,q.offset,q.count);}}
const {{rest,nodes,gradients,material,points,indices,weights}}=a,count=rest.length/3;
const identity=new Float64Array(count*9);
for(let n=0;n<count;n++)identity[n*9]=identity[n*9+4]=identity[n*9+8]=1;
const exact=applyAffineTransfer(points,indices,weights,rest,rest,identity);
const recovered=recoverSurfaceGradients(rest,rest,nodes,gradients,material,count);
const actual=applyAffineTransfer(points,indices,weights,rest,rest,recovered);
let exactError=0,recoveredError=0,gradientError=0;
for(let i=0;i<points.length;i++){{exactError=Math.max(exactError,Math.abs(exact[i]-points[i])*1000);recoveredError=Math.max(recoveredError,Math.abs(actual[i]-points[i])*1000);}}
for(let i=0;i<recovered.length;i++)gradientError=Math.max(gradientError,Math.abs(recovered[i]-identity[i]));
if(exactError>1e-10||recoveredError>1e-5)throw Error('Rest-transfer numerical gate failed');
const fixed=new Set(a.cut_recovery_incident_node_ids),x=Float64Array.from(rest);
let perturbedNodes=0;
for(let n=0;n<count;n++)if(!fixed.has(n)){{
  perturbedNodes++;
  for(let k=0;k<3;k++)x[n*3+k]+=1e-5*Math.sin((n+1)*(k+1)*.37);
}}
const probeF=recoverSurfaceGradients(x,rest,nodes,gradients,material,count);
let cutGradientError=0;
for(const n of a.unmatched_cut_support_node_ids)for(let j=0;j<9;j++)
  cutGradientError=Math.max(cutGradientError,Math.abs(probeF[n*9+j]-identity[n*9+j]));
if(cutGradientError>1e-5)throw Error('Actual incident-star collar does not freeze cut gradients');
// Same verified numerical correction used by the existing runtime. Positions
// and the affine formula are untouched; a genuinely strained F is never reset.
for(const n of a.unmatched_cut_support_node_ids){{
  probeF.fill(0,n*9,n*9+9);probeF[n*9]=probeF[n*9+4]=probeF[n*9+8]=1;
}}
const probe=applyAffineTransfer(points,indices,weights,x,rest,probeF);
let cutError=0;
for(const row of a.unmatched_cut_binding_rows)for(let k=0;k<3;k++)
  cutError=Math.max(cutError,Math.abs(probe[row*3+k]-points[row*3+k])*1000);
if(cutError>1e-10)throw Error('Unmatched-cut native transfer does not remain at rest');
console.log(JSON.stringify({{verifiedVertices:points.length/3,recoveryNodeCount:count,
 identityGradientMaxCoordinateErrorDesignUnits:exactError,
 originalFloat32RecoveredMaxCoordinateErrorDesignUnits:recoveredError,
 originalFloat32RecoveredMaxGradientError:gradientError,
 identityGateDesignUnits:1e-10,originalFloat32GateDesignUnits:1e-5,
 identityExactAlgebraically:true,originalRecoveredOutputByteExact:false,
 syntheticCollarProbe:{{notAMechanicalState:true,perturbedNoncollarNodes:perturbedNodes,
 frozenIncidentStarNodes:fixed.size,maximumCutGradientErrorBeforeNumericalCorrection:cutGradientError,
 maximumNativeCutCoordinateErrorDesignUnits:cutError,coordinateGateDesignUnits:1e-10,
 unchangedOriginalAffineFormula:true,positionTaperApplied:false}}}}));
"""
    with tempfile.TemporaryDirectory(prefix='native-oral24-binding-check-') as td:
        contract={}
        with (Path(td)/'arrays.bin').open('wb') as f:
            for name,a in arrays.items():
                f.write(b'\0'*((-f.tell())%8))
                contract[name]={'offset':f.tell(),'count':a.size,'dtype':a.dtype.str}
                a.tofile(f)
        meta=Path(td)/'contract.json'
        meta.write_text(json.dumps(contract,separators=(',',':')))
        result=subprocess.run(['node','--input-type=module','-e',code,str(meta),str(Path(td)/'arrays.bin')],
                              check=True,text=True,capture_output=True)
    return json.loads(result.stdout)

def build(geometry,output,geometry_sha256):
    geometry,output=Path(geometry),Path(output)
    if output.exists():
        raise FileExistsError('Choose a new binding output directory')
    if len(geometry_sha256)!=64 or sha(geometry)!=geometry_sha256:
        raise ValueError('Geometry differs from the explicitly supplied accepted pin')
    if sha(HEAD)!=HEAD_SHA256 or sha(ORAL)!=ORAL_SHA256:
        raise ValueError('The pinned native source or oral24 chart has changed')
    p,o,g=np.load(PATCH),np.load(ORAL),np.load(geometry)
    if (str(p['source_head_sha256'])!=HEAD_SHA256 or str(o['source_head_sha256'])!=HEAD_SHA256 or
        sha(ORIGINAL_ORAL)!=str(p['source_oral_charts_sha256']) or
        str(o['source_original_chart_sha256'])!=str(p['source_oral_charts_sha256'])):
        raise ValueError('Native oral24 source differs from original patch provenance')
    cavity_annulus='cavity_extension_mucosal_support_triangles' in g
    base_mucosal_rows=EXTERNAL_ROWS.copy() if cavity_annulus else g['oral_mucosal_rows']
    external_rows=EXTERNAL_ROWS if cavity_annulus else g['oral_external_rows']
    explicit_mucosa=cavity_annulus or 'oral_mucosal_binding_triangles' in g
    if (o['external_vertex_ids'].shape!=(17,480) or o['mucosal_vertex_ids'].shape!=(25,480) or
        not np.array_equal(external_rows,EXTERNAL_ROWS) or
        base_mucosal_rows.shape!=(5,) or base_mucosal_rows[0]!=0 or
        np.any(np.diff(base_mucosal_rows)<=0) or
        (not explicit_mucosa and not np.array_equal(base_mucosal_rows,MUCOSAL_ROWS)) or
        (explicit_mucosa and base_mucosal_rows[-1] not in (16,24))):
        raise ValueError('Geometry or native source has a different oral material chart')
    bind_base_mucosa=True
    if explicit_mucosa:
        explicit_triangles=g['cavity_extension_mucosal_support_triangles' if cavity_annulus else 'oral_mucosal_binding_triangles']
        explicit_uv=g['cavity_extension_mucosal_support_uv' if cavity_annulus else 'oral_mucosal_binding_material_uv']
        # Existing all_surface_quads stops at the base wedge. Negative IDs
        # preserve the exact independent boundary-facet index without making
        # up an unrecorded annulus quad or conflating it with a base parent.
        explicit_parents=(-1-np.arange(len(explicit_triangles),dtype=np.int32)
                          if cavity_annulus else g['oral_mucosal_binding_parent_faces'])
        first_explicit_row=int(explicit_uv[:,:,1].min())
        if first_explicit_row==0:
            bind_base_mucosa=False
        elif first_explicit_row!=base_mucosal_rows[-1]:
            raise ValueError('Explicit mucosal sidewall does not cover or continue the base chart')
        if explicit_uv[:,:,1].max()!=24:
            raise ValueError('Explicit mucosal sidewall does not reach the exact row24 boundary')
    cap_interior=verify_native_chart_topology(o)
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
        raise ValueError('This builder requires the shared 120-angle, five-depth oral24 wedge')
    if not np.array_equal(oral_grid[:,-1,:],g['midface_column_nodes'][g['oral_interface_columns']]):
        raise ValueError('The oral interface does not genuinely share all depth nodes')
    endpoint_errors={}
    for side,depth,rows in [('external',0,EXTERNAL_ROWS),('mucosal',4,base_mucosal_rows)]:
        sampled=o[f'{side}_vertex_ids'][rows[:,None],np.arange(0,480,4)[None,:]].T
        endpoint_errors[side]=float(np.max(np.abs(vertices[oral_grid[:,:,depth]]-hv[sampled])))
        if endpoint_errors[side]>(0 if side=='external' else 1e-10):
            raise ValueError(f'The {side} endpoint differs from its exact native material sample')
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
        if not np.isfinite(weights).all() or weights.min()<0 or abs(weights.sum()-1)>1e-13:
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

    # Use the actual top-triangle diagonals for the base wedge. A reversed oral
    # quad can change its diagonal; no guessed split. An annular continuation
    # uses its independently declared actual native boundary facets below.
    charts=[('external',0,1,EXTERNAL_ROWS)]
    if bind_base_mucosa:
        charts.append(('mucosal',4,2,base_mucosal_rows))
    for side,depth,region,coarse_rows in charts:
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
                for row,angle,uv in material_samples(radial,coarse_rows,angular):
                    for nodes,coords in supports:
                        w=barycentric(uv,coords)
                        if w is not None:
                            put(native_rows[row,angle],nodes,w,region,face,uv)
                            break
                    else:
                        raise ValueError('Oral material point not in actual boundary support')

    explicit_only_ids=set()
    if explicit_mucosa:
        if not np.array_equal(hv[o['mucosal_vertex_ids']],o['mucosal_positions']):
            raise ValueError('Native mucosal positions changed from the exact head')
        endpoint_errors['explicit_mucosa']=verify_explicit_mucosal_corners(o['mucosal_vertex_ids'],explicit_triangles,explicit_uv,vertices,hv)
        def put_explicit(*record):
            if int(record[0]) not in records:
                explicit_only_ids.add(int(record[0]))
            put(*record)
        bind_explicit_mucosal_facets(o['mucosal_vertex_ids'],explicit_triangles,explicit_uv,
                                     explicit_parents,put_explicit)

    expected=np.unique(np.r_[p['dense_native_vertex_ids'],o['external_vertex_ids'].ravel(),o['mucosal_vertex_ids'].ravel()])
    vertex_ids=np.array(sorted(records),dtype=np.int32)
    if not np.array_equal(vertex_ids,expected):
        raise ValueError('Missing or duplicated native material vertex')
    if len(vertex_ids)!=EXPECTED_NATIVE_VERTICES or np.isin(cap_interior,vertex_ids).any():
        raise ValueError('Unexpected bound count or remaining native cap was accidentally bound')
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
             'parent_uv_is_global_mucosal':np.array([int(i) in explicit_only_ids for i in vertex_ids]),
             'boundary_owner_tetrahedra':np.array([records[int(i)][5] for i in vertex_ids],dtype=np.int32),
             'nodes_per_vertex':np.array(3,dtype=np.int32),'required_gradient_recovery_count':np.array(len(vertices),dtype=np.int32),
             'solver_units_per_design_unit':np.array(.001),
             'source_geometry_sha256':np.array(sha(geometry)),
             'source_head_sha256':p['source_head_sha256'],
             'source_patch_sha256':np.array(sha(PATCH)),
             'source_oral_charts_sha256':np.array(sha(ORAL)),
             'source_original_oral_charts_sha256':np.array(sha(ORIGINAL_ORAL)),
             'oral_external_rows':EXTERNAL_ROWS.copy(),'oral_mucosal_rows':base_mucosal_rows.copy(),
             'deep_mucosal_cut_native_ids':o['mucosal_vertex_ids'][24].astype(np.int32),
             'remaining_cap_interior_native_ids':cap_interior.astype(np.int32)}
    cut_rows=[]
    for name in ['outer_cut','right_eye_cut','left_eye_cut','right_nostril_cut','left_nostril_cut']:
        ids=p[f'{name}_dense_native_ids']
        rows=binding_rows(vertex_ids,ids)
        payload[f'{name}_binding_rows']=rows
        cut_rows.extend(rows)
    mucosal_cut=binding_rows(vertex_ids,o['mucosal_vertex_ids'][24])
    payload['deep_mucosal_cut_binding_rows']=mucosal_cut
    cut_rows=np.unique(np.r_[cut_rows,mucosal_cut]).astype(np.int32)
    payload['unmatched_cut_binding_rows']=cut_rows
    cut_nodes,incident,incident_nodes=recovery_star(tets,indices,weights,cut_rows)
    payload['unmatched_cut_support_node_ids']=cut_nodes
    # One conservative way to ensure each cut support's recovered F is identity
    # is to freeze the complete incident tet star. This is provenance, not an
    # applied or recommended final anatomical boundary condition.
    payload['cut_recovery_incident_tetrahedra']=incident
    payload['cut_recovery_incident_node_ids']=incident_nodes
    deep_nodes,deep_incident,deep_incident_nodes=recovery_star(tets,indices,weights,mucosal_cut)
    payload['deep_mucosal_cut_support_node_ids']=deep_nodes
    payload['deep_mucosal_cut_recovery_incident_tetrahedra']=deep_incident
    payload['deep_mucosal_cut_recovery_incident_node_ids']=deep_incident_nodes
    expected_deep_nodes=(g['cavity_extension_native_cut_nodes' if cavity_annulus else 'oral_mucosal_deep_cut_node_ids']
                         if explicit_mucosa else oral_grid[:,-1,-1])
    if not np.array_equal(deep_nodes,np.unique(expected_deep_nodes).astype(np.uint32)):
        raise ValueError('Deep native cut does not use exactly the actual inner row24 boundary nodes')
    payload['oral_shared_interface_node_ids']=oral_grid[:,-1,:].astype(np.uint32)
    transfer=verify_original_transfer(payload,vertices,tets)
    output.mkdir(parents=True)
    target=output/'native_surface_binding.npz'
    np.savez_compressed(target,**payload)
    report={'schema':'original-native-v16-bilateral-oral24-material-chart-binding-v1',
        'status':'PASS exact correspondence/rest transfer and synthetic frozen-cut continuity; mechanics/contact not established by binding',
        'source_geometry':str(geometry.resolve()),'source_geometry_sha256':sha(geometry),
        'required_geometry_sha256':geometry_sha256,
        'source_head_sha256':str(p['source_head_sha256']),'source_patch_sha256':sha(PATCH),
        'source_oral_charts_sha256':sha(ORAL),'builder_sha256':sha(__file__),
        'source_original_oral_charts_sha256':sha(ORIGINAL_ORAL),
        'original_transfer_module_sha256':sha(ROOT/'src/affine_surface_transfer.mjs'),
        'original_gradient_recovery_module_sha256':sha(ROOT/'reference-source/src/layered_fem_transfer_math.js'),
        'output_file':target.name,'output_sha256':sha(target),'output_bytes':target.stat().st_size,
        'bound_native_vertices':len(vertex_ids),'unique_native_vertices':len(np.unique(vertex_ids)),
        'additional_native_vertices_from_mucosal_extension':3840,
        'base_oral_rows_mapping':np.c_[EXTERNAL_ROWS,base_mucosal_rows].tolist(),
        'explicit_mucosal_boundary_facets':len(explicit_triangles) if explicit_mucosa else 0,
        'mucosal_architecture':'preserved oral16 wedge plus appended native cavity annulus' if cavity_annulus else 'declared native material support',
        'native_support_corner_maximum_coordinate_error_design_units':endpoint_errors,
        'native_support_coordinate_roundoff_gate_design_units':1e-10,
        'remaining_cap_interior_native_vertices':len(cap_interior),
        'row24_is_exact_native_cap_boundary':True,
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
        'cut_effective_support_rule':'Strictly positive barycentric weights only, after 1e-12 exact-edge roundoff snapping; zero-weight triangle corners are excluded.',
        'zero_weight_cut_support_slots_excluded':int(np.sum(weights[cut_rows]==0)),
        'unmatched_cut_incident_tetrahedra':len(incident),'unmatched_cut_incident_nodes':len(incident_nodes),
        'deep_mucosal_cut_native_vertices':len(mucosal_cut),'deep_mucosal_cut_support_nodes':len(deep_nodes),
        'deep_mucosal_cut_incident_tetrahedra':len(deep_incident),
        'deep_mucosal_cut_incident_nodes':len(deep_incident_nodes),
        'compatibility':{'applyAffineTransfer':'Flatten points, indices and weights in C order; x/rest must use solver units (.001*design coordinates); nodesPerVertex=3.',
            'recoverSurfaceGradients':f'Use count={len(vertices)} with ORIGINAL global node ordering. Mucosal support is not a prefix of exterior nodes. Supply actual volume gradients and material volumes from this exact geometry.',
            'rest_residual':'The original formula transports q-X_i separately per support, not one untransformed additive offset. rest_offsets is diagnostic.',
            'parent_uv':'Local parent-quad UV unless parent_uv_is_global_mucosal is true. Explicit mucosal facets store global [angle/480,row/24]; native IDs and boundary owner tetrahedra remain authoritative.',
            'parent_faces':'Nonnegative IDs reference all_surface_quads. For cavity-annulus facets, -1-index references the exact source cavity_extension_mucosal_support_triangles index.',
            'storage':'NPZ numeric arrays plus this JSON manifest. No full head is copied.'},
        'cut_semantics':{'outer_cut':'Beyond the shared head rectangle is omitted.',
            'eye_cuts':'Inner lids and sockets remain unbound.',
            'nostril_cuts':'Native vestibular walls/caps remain unbound.',
            'deep_mucosal_cut':'Native mucosal row24 is bound; the 1681 remaining native cap interior vertices remain unbound at rest.',
            'necessary_runtime_constraint':'All nodes in cut_recovery_incident_node_ids must stay at rest. Recompute ORIGINAL recovered gradients over all actual incident tetrahedra. Verify support gradients are identity within the original numerical tolerance before correcting only roundoff to exact identity. Preserve the original affine residual formula. Stored incident stars are required support provenance, not applied mechanics constraints; a consuming case must enforce and audit them.'},
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
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--geometry-sha256',required=True,help='Explicit pin of the positively accepted geometry; this tool does not grant acceptance.')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.geometry,args.output,args.geometry_sha256)
