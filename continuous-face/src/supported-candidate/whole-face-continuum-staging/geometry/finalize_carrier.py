"""Exact positive integration of selected voxel geometry via nested Freudenthal tets.
Common P1 displacement remains coarse. Fine tets are quadrature only, no new DOFs.
Agglomeration conserves volume/material/source phase and affine moments exactly.
"""
from pathlib import Path
import numpy as np,json,itertools,hashlib,time
from scipy.spatial import cKDTree
G=Path(__file__).parent;start=time.time();a=dict(np.load(G/'whole-face-continuum.npz'));phase=dict(np.load(G/'whole-face-muscle-phases.npz'));r=json.loads((G/'whole-face-geometry-report.json').read_text())
p=a['quadrature_points'].astype(float);material=a['quadrature_material_fraction'];H=float(a['carrier_spacing']);s=float(a['grid_spacing']);origin=a['grid_origin'];co=np.floor(origin/H)*H;perms=np.array(list(itertools.permutations(range(3))),int);cs=np.floor((p.max(0)+s-co)/H).astype(int)+2
centroid=[]
for perm in perms:
 vv=np.zeros((4,3));vv[1,perm[0]]=1;vv[2,perm[:2]]=1;vv[3]=1;centroid.append(vv.mean(0)-.5)
parent=[];points=[]
for delta in centroid:
 qq=p+s*delta;cc=np.floor((qq-co)/H).astype(int);ff=(qq-co)/H-cc;order=np.argsort(-ff,axis=1);tt=np.zeros(len(p),np.int32)
 for j,perm in enumerate(perms):tt[np.all(order==perm,axis=1)]=j
 key=np.ravel_multi_index((cc[:,0],cc[:,1],cc[:,2],tt),tuple(cs)+(6,));parent.append(key.astype(np.int32));points.append(qq.astype(np.float32))
parent=np.array(parent);size=int(np.prod(cs)*6);volume0=np.bincount(parent.ravel(),minlength=size)*s**3/6;active=np.flatnonzero(volume0>0);remap=np.full(size,-1,np.int32);remap[active]=np.arange(len(active));parent=remap[parent];vol=volume0[active];Q=len(active)
q=np.zeros((Q,3));mv=np.zeros((Q,5));unit=s**3/6
for j in range(6):
 for ax in range(3):q[:,ax]+=np.bincount(parent[j],weights=points[j][:,ax]*unit,minlength=Q)
 for mat in range(5):mv[:,mat]+=np.bincount(parent[j],weights=material[:,mat]*unit,minlength=Q)
q/=vol[:,None];fractions=mv/vol[:,None];fractions/=fractions.sum(1)[:,None]
c=np.array(np.unravel_index(active,tuple(cs)+(6,))).T;itets=np.zeros((Q,4,3),int);itets[:,0]=c[:,:3];itets[:,1]=c[:,:3];itets[:,2]=c[:,:3];itets[:,3]=c[:,:3]+1
for j,perm in enumerate(perms):
 ids=np.flatnonzero(c[:,3]==j);itets[ids,1,perm[0]]+=1
 for ax in perm[:2]:itets[ids,2,ax]+=1
unodes,inv=np.unique(itets.reshape(-1,3),axis=0,return_inverse=True);tets=inv.reshape(-1,4).astype(np.int32);nodes=co+H*unodes;x=nodes[tets];D=np.stack([x[:,1]-x[:,0],x[:,2]-x[:,0],x[:,3]-x[:,0]],axis=2);neg=np.linalg.det(D)<0;tets[neg,1:3]=tets[neg,2:0:-1];x=nodes[tets];D=np.stack([x[:,1]-x[:,0],x[:,2]-x[:,0],x[:,3]-x[:,0]],axis=2);invD=np.linalg.inv(D);tv=np.linalg.det(D)/6
bary=lambda pp,ids: np.einsum('nij,nj->ni',invD[ids],pp-x[ids,0])
qb=bary(q,np.arange(Q));qb=np.c_[1-qb.sum(1),qb]
# Each boundary vertex lies in closure of at least one positive nested integration tet.
# Find an active containing carrier, never extrapolate through an integration-empty cell.
bv=a['vertices'].astype(float);cent=x.mean(1);tree=cKDTree(cent);vt=np.full(len(bv),-1,np.int32);vb=np.zeros((len(bv),4))
for startrow in range(0,len(bv),4096):
 rr=np.arange(startrow,min(startrow+4096,len(bv)));idx=tree.query(bv[rr],k=64)[1];ww=np.einsum('nkij,nkj->nki',invD[idx],bv[rr,None,:]-x[idx,0]);ww=np.concatenate([1-ww.sum(2)[:,:,None],ww],2);good=ww.min(2)>=-1e-8
 if not np.all(good.any(1)):raise ValueError('Boundary has no containing active tet '+str(int((~good.any(1)).sum())))
 choice=good.argmax(1);vt[rr]=idx[np.arange(len(rr)),choice];vb[rr]=ww[np.arange(len(rr)),choice]
# Sparse source phases: equal native-overlap share conservatively transferred from fine cells.
oldq=phase['muscle_quadrature_index'];pid=phase['muscle_source_part'];frac=phase['muscle_fraction'].astype(float);pairs=np.concatenate([parent[j,oldq].astype(np.int64)*256+pid for j in range(6)]);weight=np.tile(frac*unit,6);up,ip=np.unique(pairs,return_inverse=True);pv=np.bincount(ip,weights=weight);mqi=(up//256).astype(np.int32);mpi=(up%256).astype(np.int16);mf=pv/vol[mqi]
sourcefrac=np.bincount(mqi,weights=mf,minlength=Q);sourceerr=float(abs(sourcefrac-fractions[:,1]).max());assert sourceerr<2e-7,sourceerr
# Resolve only floating-point phase partition roundoff, conserving total physical volume.
other=[0,2,3,4];fractions[:,other]*=((1-sourcefrac)/np.maximum(fractions[:,other].sum(1),1e-30))[:,None];fractions[:,1]=sourcefrac
mcent=np.zeros((len(up),3));phasepoints=np.concatenate([points[j][oldq] for j in range(6)])
for ax in range(3):mcent[:,ax]=np.bincount(ip,weights=weight*phasepoints[:,ax])/pv
nodeweight=np.zeros(len(nodes));np.add.at(nodeweight,tets.ravel(),(qb*vol[:,None]).ravel());assert np.all(nodeweight>0)
assert np.all(vol<=tv+1e-8),(vol-tv).max();assert qb.min()>-1e-8
# Physical volume geometry retained separately as literal fine-cell selection, not full atlas.
np.savez_compressed(G/'whole-face-integration-cells.npz',cell_ijk=a['quadrature_cell_ijk'],cell_material_fraction=material,cell_component=a['quadrature_component'],grid_origin=origin,grid_spacing=np.array(s),source_sha256=a['source_sha256'])
primary=np.full(Q,-1,np.int16);maxfrac=np.zeros(Q)
for qq,pp,ff in zip(mqi,mpi,mf):
 if ff>maxfrac[qq]:primary[qq]=pp;maxfrac[qq]=ff
# Nearest fine-cell owner on a boundary is only for geometric traceability, not final quadrature index.
a['boundary_owner_fine_cell']=a.pop('boundary_owner_quadrature');a.pop('quadrature_cell_ijk');a.pop('quadrature_source_overlap_count');a.pop('quadrature_component')
a.update(nodes=nodes,tets=tets,tet_reference_volume=tv,quadrature_points=q,quadrature_weights=vol,quadrature_tet=np.arange(Q,dtype=np.int32),quadrature_barycentric=qb,quadrature_material_fraction=fractions,quadrature_source_part=primary,exterior_tet=vt,exterior_barycentric=vb,node_integrated_basis_weight=nodeweight,quadrature_fine_subtet_count=(vol/unit).round().astype(np.int32))
np.savez_compressed(G/'whole-face-continuum-final.npz',**a)
np.savez_compressed(G/'whole-face-muscle-phases-final.npz',muscle_quadrature_index=mqi,muscle_source_part=mpi,muscle_fraction=mf,muscle_source_centroid=mcent,source_sha256=a['source_sha256'])
r['mechanical_carrier']={'nodes':len(nodes),'tets':len(tets),'positive_volume_min':float(tv.min()),'quadrature_rows':Q,'quadrature_positive_min':float(vol.min()),'quadrature_total_volume':float(vol.sum()),'material_volume':dict(zip(['matrix','muscle','fascia','fat','dermis'],mv.sum(0).tolist())),'fraction_sum_max_error':float(abs(fractions.sum(1)-1).max()),'orphan_dofs':int((nodeweight<=0).sum()),'exterior_vertices_affine_extrapolated_to_active_carrier':0,'exterior_affine_reconstruction_max_error':float(np.linalg.norm(np.einsum('ni,nij->nj',vb,nodes[tets[vt]])-bv,axis=1).max()),'quadrature_affine_reconstruction_max_error':float(np.linalg.norm(np.einsum('ni,nij->nj',qb,nodes[tets])-q,axis=1).max()),'quadrature_barycentric_min':float(qb.min()),'exterior_barycentric_min':float(vb.min()),'maximum_integrated_to_carrier_volume_ratio':float((vol/tv).max()),'integration':'Each occupied1-L0 voxel split into six nested positive Freudenthal tets (vol1/6 L0^3). Subtet centroid quadrature agglomerated by common6-L0 P1 carrier, preserving volume, phase weights and affine moments. Exact for selected voxel domain, NOT exact source clipping.','interface_policy':'Bonded common displacement, normalized constitutive mixture. No muscle contact barriers.'}
r['muscle_phases'].update(file='whole-face-muscle-phases-final.npz',rows=len(mqi),normalized_mixture_max_error=float(abs(sourcefrac-fractions[:,1]).max()),pre_correction_float32_roundoff=sourceerr,agglomeration='Conserved native source phase volumes from fine cell membership into common coarse tet rows')
r['geometry_sha256']=hashlib.sha256((G/'whole-face-continuum-final.npz').read_bytes()).hexdigest();r['carrier_finalization_seconds']=time.time()-start;(G/'whole-face-geometry-report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r['mechanical_carrier'],indent=2));print('DONE',time.time()-start)
