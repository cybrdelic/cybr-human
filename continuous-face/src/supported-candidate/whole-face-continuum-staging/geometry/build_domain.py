"""Isolated source-only homogenized full-head soft-tissue geometry. L0 units."""
from pathlib import Path
import sys,json,time,hashlib,itertools,ast
import numpy as np
from scipy.ndimage import map_coordinates,label,generate_binary_structure,binary_fill_holes,binary_closing,binary_dilation,distance_transform_edt,gaussian_filter
from scipy.spatial import cKDTree
G=Path(__file__).resolve().parent; B=G.parents[1]; SOURCE=B/'full-face-coverage-plan/restored'
sys.path.insert(0,str(G/'helpers'))
from voxel_support import VoxelSupport
# Import only audited pure definitions, never archive module main or source paths.
def defs(path,names,extra=None):
 tree=ast.parse(Path(path).read_text());sel=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
 ns=globals().copy();ns.update(extra or {});exec(compile(ast.Module(body=sel,type_ignores=[]),str(path),'exec'),ns);return ns
lib=defs(G/'helpers/build_reference.py',{'projected_aperture','aperture_at','boundary_from_cells'})
projected_aperture=lib['projected_aperture'];aperture_at=lib['aperture_at'];boundary_from_cells=lib['boundary_from_cells']
f=defs(G/'helpers/free_skin.py',{'part_vertices','FatLayer'});FatLayer=f['FatLayer']

def log(*x): print(*x,flush=True)
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mesh(pid):
 ff=F[PID==pid];vv,inv=np.unique(ff,return_inverse=True);return V[vv],inv.reshape(-1,3),vv

def raster_solid(vertices,faces,origin,spacing,shape):
 """Finite mesh signed z-ray scan on voxel centers. Returns center membership.
 Nonzero winding along each vertical ray; open mesh results separately flagged.
 Does not create halfspace obstacles. Native meshes unchanged.
 """
 tri=vertices[faces]; lo=np.maximum(0,np.ceil((tri.min((0,1))-origin)/spacing-.5).astype(int));hi=np.minimum(shape,np.floor((tri.max((0,1))-origin)/spacing-.5).astype(int)+1)
 if np.any(hi<=lo):return np.empty((0,3),np.int16),{'closed':False,'center_cells':0}
 ss=hi-lo; diff=np.zeros((ss[0],ss[1],ss[2]+1),np.int16); hits=0
 for t in tri:
  a,b,c=t; det=np.cross(b-a,c-a)[2]
  if abs(det)<1e-10:continue
  low=np.maximum(lo[:2],np.ceil((t[:,:2].min(0)-origin[:2])/spacing-.5-1e-8).astype(int));high=np.minimum(hi[:2],np.floor((t[:,:2].max(0)-origin[:2])/spacing-.5+1e-8).astype(int)+1)
  if np.any(high<=low):continue
  ii,jj=np.meshgrid(np.arange(low[0],high[0]),np.arange(low[1],high[1]),indexing='ij');ii=ii.ravel();jj=jj.ravel()
  xy=origin[:2]+spacing*(np.c_[ii,jj]+.5);d=xy-a[:2];ab=b[:2]-a[:2];ac=c[:2]-a[:2]
  u=(d[:,0]*ac[1]-d[:,1]*ac[0])/det;w=(ab[0]*d[:,1]-ab[1]*d[:,0])/det
  # Half-open ownership via tiny deterministic ray offset prevents edge double hits.
  good=(u>=-1e-9)&(w>=-1e-9)&(u+w<=1+1e-9)
  ii=ii[good];jj=jj[good];zz=a[2]+u[good]*(b[2]-a[2])+w[good]*(c[2]-a[2]);kk=np.floor((zz-origin[2])/spacing-.5+1e-9).astype(int)+1
  kk=np.clip(kk-lo[2],0,ss[2]);np.add.at(diff,(ii-lo[0],jj-lo[1],kk),int(np.sign(det)));hits+=len(ii)
 inside=np.cumsum(diff,axis=2)[:,:,:-1]!=0
 # Native triangles can have open cuts; do not extend their rays past source bounds.
 ijk=np.argwhere(inside)+lo
 edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1);_,cnt=np.unique(edges,axis=0,return_counts=True)
 return ijk.astype(np.int16),{'closed':bool(np.all(cnt==2)),'boundary_edges':int((cnt==1).sum()),'nonmanifold_edges':int((cnt>2).sum()),'center_cells':len(ijk),'ray_hits':hits,'ray_end_residual_columns':int((diff.sum(2)!=0).sum())}

def nasal_meshes():
 pid=next(i for i,p in enumerate(M['parts']) if p['name']=='Mucosa of nasal cavity');v,f,globalids=mesh(pid)
 de=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);ue,inv,cnt=np.unique(np.sort(de,axis=1),axis=0,return_inverse=True,return_counts=True);rem={tuple(e) for e in de[cnt[inv]==1]};loops=[]
 while rem:
  a,b=next(iter(rem));loop=[a];rem.remove((a,b))
  while b!=a:
   loop.append(b);edge=next((e for e in rem if e[0]==b),None)
   if edge is None:raise ValueError('Nasal boundary not oriented loop')
   rem.remove(edge);b=edge[1]
  loops.append(np.array(loop,int))
 centers=np.array([v[x].mean(0) for x in loops]);front=np.argmax(centers[:,2]);nv=list(v);nf=list(f);rep=[]
 for li,loop in enumerate(loops):
  ring=v[loop];center=ring.mean(0)
  if li==front:
   _,_,vh=np.linalg.svd(ring-center,full_matrices=False);n=vh[-1];n*=np.sign(n@(center-centers[1-li]));out=np.arange(len(nv),len(nv)+len(loop));nv.extend(ring+25*n)
   for j in range(len(loop)):
    a,b=int(loop[j]),int(loop[(j+1)%len(loop)]);c,d=int(out[j]),int(out[(j+1)%len(loop)]);nf.extend([[b,a,c],[b,c,d]])
   cap=out;cc=center+25*n;rep.append({'native_loop_vertices':globalids[loop].tolist(),'extension':25.,'direction':n.tolist(),'center':center.tolist()})
  else:cap=loop;cc=center
  ci=len(nv);nv.append(cc)
  for j in range(len(cap)):nf.append([int(cap[(j+1)%len(cap)]),int(cap[j]),ci])
 vv=np.array(nv);ff=np.array(nf,int);vleft=vv.copy();vleft[:,0]*=-1
 return [(vv,ff),(vleft,ff[:,[0,2,1]])],{'source_part':pid,'native_loops':len(loops),'front_loop':rep,'left_mirrored_from_recovered_right':True,'posterior_loop_planar_cap':True}

def main():
 global V,F,PID,M
 start=time.time();src=SOURCE/'exact-rest/anatomy-registered-rest.npz';sha=hashfile(src)
 assert sha=='2555740759e8dd2f50a504914d02f582ba13725c8c77da1fbbb67c55f4c9978d',sha
 aa=np.load(src);A={k:aa[k] for k in aa.files};V=A['vertices'];F=A['triangles'];PID=A['part_ids'];M=json.loads((SOURCE/'anatomy-reference/prior-body-handoff/anatomy-metadata.json').read_text());fat=FatLayer(A,M)
 cache=G/'source-envelope-field.npz';cachejson=G/'source-envelope-field.json'
 if cache.exists():
  s=np.load(cache);field=s['field'];fo=s['origin'];fs=float(s['spacing']);summary=json.loads(cachejson.read_text());log('Loaded envelope')
 else:
  s=VoxelSupport(A,M,already_registered=True,spacing=2.,include_superficial_connective=True,neck_support=False,neck_join=False,root_connective_closure_width=0.,surface_sampling_spacing=1.)
  field=s.field;fo=s.origin;fs=s.spacing;summary=s.summary;np.savez_compressed(cache,field=field,origin=fo,spacing=fs);cachejson.write_text(json.dumps(summary,indent=2));log('Envelope',field.shape,time.time()-start)
 spacing=1.; origin=np.floor((fo+4)/6)*6;hi=fo+(np.array(field.shape)-3)*fs;origin[1]=-96.;shape=np.ceil((hi-origin)/spacing).astype(int)
 shape[1]=min(shape[1],int(np.ceil((124-origin[1])/spacing)));shape=tuple(shape.tolist())
 # A 3-unit deep homogenized connective matrix backing is an explicit coarse approximation,
 # not atlas fascia thickness. Full finite muscle interiors are unioned with it.
 deep_backing=3.;occupancy=np.zeros(shape,bool);dfield=np.full(shape,100.,np.float32);cavity=np.zeros(shape,np.uint8);muscle_count=np.zeros(shape,np.uint8);muscle_pid=np.full(shape,-1,np.int16);rigid=np.zeros(shape,bool);rigid_pid=np.full(shape,-1,np.int16)
 for i in range(shape[0]):
  jj,kk=np.meshgrid(np.arange(shape[1]),np.arange(shape[2]),indexing='ij');p=origin+spacing*(np.c_[np.full(jj.size,i),jj.ravel(),kk.ravel()]+.5);d=map_coordinates(field,((p-fo)/fs).T,order=1,mode='constant',cval=100.);dfield[i]=d.reshape(shape[1:]);h=fat.thickness(p);occupancy[i]=((d>=-deep_backing)&(d<h+1.5)).reshape(shape[1:])
 np.savez_compressed(G/'domain-grid-base.npz',origin=origin,spacing=spacing,shape=shape,dfield=dfield,occupancy=occupancy)
 log('Base band',int(occupancy.sum()),time.time()-start)
 source_records=[];muscle_ids=[]
 if (G/'source-membership.npz').exists() and (G/'whole-face-geometry-report.json').exists():
  sm=np.load(G/'source-membership.npz');muscle_count=sm['muscle_count'];muscle_pid=sm['muscle_source_part'];rigid=sm['rigid'];rigid_pid=sm['rigid_source_part'];occupancy|=muscle_count>0
  source_records=json.loads((G/'whole-face-geometry-report.json').read_text())['source_records'];muscle_ids=[x['part_id'] for x in source_records if x['kind']=='muscle'];log('Loaded source memberships')
 else:
  for pid,part in enumerate(M['parts']):
   kind=part['kind'];name=part['name']
   ismuscle=kind=='muscle' and pid<193 and not any(w in name for w in ['rectus muscle','oblique muscle','Levator palpebrae','digastric','pterygoid'])
   isrigid=(kind=='bone' and pid<193) or (kind=='cartilage' and pid<193) or kind=='teeth'
   if not(ismuscle or isrigid):continue
   vv,ff,_=mesh(pid);ijk,r=raster_solid(vv,ff,origin,spacing,np.array(shape));r.update(part_id=pid,name=name,kind=kind);source_records.append(r)
   if len(ijk):
    ix=tuple(ijk.T)
    if ismuscle:
     muscle_ids.append(pid);muscle_count[ix]+=1;muscle_pid[ix]=pid;occupancy[ix]=True
    else:rigid[ix]=True;rigid_pid[ix]=pid
   log('Source',pid,name,len(ijk),'closed',r['closed'],time.time()-start)
 # Never let deep source structures create an external shell above their own finite atlas envelope.
 for i in range(shape[0]):
  rows=np.argwhere(occupancy[i]);p=origin+spacing*(np.c_[np.full(len(rows),i),rows]+.5);d=dfield[i][tuple(rows.T)];occupancy[i][tuple(rows.T)]&=(d<fat.thickness(p)+1.5)
 overlap_pre=int(np.sum(occupancy&rigid));occupancy&=~rigid
 np.savez_compressed(G/'source-membership.npz',muscle_count=muscle_count,muscle_source_part=muscle_pid,rigid=rigid,rigid_source_part=rigid_pid)
 # Recover source holes rather than impose ellipses or artificial globe spheres.
 named=lambda names:F[np.isin(PID,[i for i,p in enumerate(M['parts']) if p['name'] in names])]
 oral,oralrep=projected_aperture(V,named(['Orbicularis oris muscle.l','Orbicularis oris muscle.r']))
 eyes=[];eyereps=[]
 for side in ('l','r'):
  eye,rep=projected_aperture(V,named(['Palpebral part of orbicularis oculi.'+side,'Superior tarsus.'+side,'Inferior tarsus.'+side]));eyes.append(eye);eyereps.append(rep)
 # Depths are finite source-derived cut planes, not claimed missing oral lining/globes.
 oral_depth=float(V[np.unique(named(['Orbicularis oris muscle.l','Orbicularis oris muscle.r']))][:,2].min())
 eye_depth=float(V[np.unique(named(['Superior tarsus.l','Inferior tarsus.l','Superior tarsus.r','Inferior tarsus.r']))][:,2].min())-12.
 for i in range(shape[0]):
  jj,kk=np.meshgrid(np.arange(shape[1]),np.arange(shape[2]),indexing='ij');p=origin+spacing*(np.c_[np.full(jj.size,i),jj.ravel(),kk.ravel()]+.5);cl=np.zeros(len(p),np.uint8)
  cl[aperture_at(p,oral)&(p[:,2]>=oral_depth)]=1
  for k,eye in enumerate(eyes):cl[aperture_at(p,eye)&(p[:,2]>=eye_depth)]=2+k
  cavity[i]=cl.reshape(shape[1:])
 lumens,nasalrep=nasal_meshes()
 for k,(vv,ff) in enumerate(lumens):
  ijk,r=raster_solid(vv,ff,origin,spacing,np.array(shape));cavity[tuple(ijk.T)]=4+k;nasalrep[str(k)+'_raster']=r
 occupancy&=cavity==0
 # Preserve all substantive physical regions. No morphological bridge or hole-fill is applied.
 comp,n=label(occupancy,generate_binary_structure(3,1));counts=np.bincount(comp.ravel());counts[0]=0
 log('Components',sorted(counts[1:].tolist(),reverse=True)[:25],time.time()-start)
 # Disconnected numerical debris smaller than 8 unit cells is excluded, explicitly accounted.
 debris_ids=np.flatnonzero((counts>0)&(counts<8));removed=int(np.sum(np.isin(comp,debris_ids)));occupancy[np.isin(comp,debris_ids)]=False
 comp,n=label(occupancy,generate_binary_structure(3,1));counts=np.bincount(comp.ravel());counts[0]=0
 qijk=np.argwhere(occupancy).astype(np.int16);q=origin+(qijk+.5)*spacing;d=dfield[tuple(qijk.T)];h=fat.thickness(q);mc=muscle_count[tuple(qijk.T)];sp=muscle_pid[tuple(qijk.T)]
 bv,bf,owners=boundary_from_cells(occupancy,origin,spacing)
 # Boundary class from its outside-adjacent cell: 0 exterior,1 oral,2/3 eyes,4/5 nasal,6 rigid,7 deep matrix,8 domain cut.
 centers=bv[bf].mean(1);norm=np.cross(bv[bf[:,1]]-bv[bf[:,0]],bv[bf[:,2]]-bv[bf[:,0]]);norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-15);outside=centers+.1*norm;oi=np.floor((outside-origin)/spacing).astype(int);valid=np.all((oi>=0)&(oi<np.array(shape)),axis=1);bc=np.full(len(bf),8,np.uint8);insideids=np.flatnonzero(valid);o=tuple(oi[valid].T);bc[valid]=cavity[o];r=rigid[o];bc[insideids[(bc[valid]==0)&r]]=6;deep=(bc[insideids]==0)&(dfield[o]<-deep_backing+.1);bc[insideids[deep]]=7
 support_source=np.full(len(bf),-1,np.int16);support_source[valid]=rigid_pid[o]
 # Explicit source-only preliminary view is saved before mechanics assembly.
 np.savez_compressed(G/'whole-face-boundary-preview.npz',vertices=bv.astype(np.float32),triangles=bf.astype(np.int32),boundary_class=bc,source_part=support_source)
 np.savez_compressed(G/'cavity-guides.npz',oral_origin=oral['origin'],oral_spacing=oral['spacing'],oral_mask=oral['mask'],left_eye_origin=eyes[0]['origin'],left_eye_mask=eyes[0]['mask'],right_eye_origin=eyes[1]['origin'],right_eye_mask=eyes[1]['mask'],eye_spacing=eyes[0]['spacing'],right_nasal_vertices=lumens[0][0],right_nasal_triangles=lumens[0][1],left_nasal_vertices=lumens[1][0],left_nasal_triangles=lumens[1][1])
 log('PREVIEW',len(bv),len(bf),G/'whole-face-boundary-preview.npz',time.time()-start)
 report={'status':'coarse source-derived geometry, not solved or collision certified','schema':1,'source':{'path':str(src),'sha256':sha,'vertices':len(V),'triangles':len(F),'parts':len(M['parts']),'registration_reapplied':False,'units':'L0 design units; not calibrated millimeters'},'support':summary,'assumptions':{'carrier_spacing':6.,'integration_cell_spacing':1.,'dermis_thickness':1.,'outer_fascia_thickness':.5,'deep_homogenized_matrix_backing':deep_backing,'fat':fat.parameters(),'fat_nose_attenuation':'fixed declared anatomy-frame Gaussian 0.88 peak from source helper','cavity_cut_padding':0.,'oral_extrusion_depth':oral_depth,'ocular_extrusion_depth':eye_depth,'ocular_depth_extension':12.,'ears':'Absent: no recovered auricular cartilage or fabricated ears','globes':'Absent: no globe spheres fabricated','neck':'Excluded, no neck support/join/root closure','disconnected_debris_removed':removed,'source_membership':'finite native-triangle z-ray winding center sampling; open source caveats logged','opening_geometry':'native projected source holes and capped/extended native nasal mucosal loop, explicit finite continuation assumptions','no_morphological_hole_fill_or_bridging':True},'source_records':source_records,'cavities':{'oral':oralrep,'ocular':eyereps,'nasal':nasalrep},'preliminary':{'integration_cells':len(q),'volume':float(len(q)*spacing**3),'component_sizes':sorted(counts[1:].tolist(),reverse=True),'boundary_vertices':len(bv),'boundary_triangles':len(bf),'boundary_classes':{str(k):int((bc==k).sum()) for k in np.unique(bc)},'rigid_center_overlap_removed':overlap_pre},'limitations':['Voxel center membership does not prove complete voxel exterior to rigid source; boundary position quantization <=sqrt(3)/2 L0 plus envelope approximation.','Finite atlas overlaps are normalized, not stacked independent tissue layers.','Open/nonmanifold source meshes are approximated only within native finite bounds; source records show residual rays.','Source envelope has angular nearest-fill and smoothing; not a target outer face or contact certificate.','Carrier continuity does not prove biological fidelity or convergence.']}
 (G/'whole-face-geometry-report.json').write_text(json.dumps(report,indent=2))
 # One common P1 tetrahedral carrier. Freudenthal 6-tet regular cells. No fine-cell DOFs.
 H=6.;co=np.floor(origin/H)*H;perms=np.array(list(itertools.permutations(range(3))),int)
 def embed(p):
  c=np.floor((p-co)/H+1e-10).astype(int);f=(p-co)/H-c;order=np.argsort(-f,axis=1,kind='stable');which=np.zeros(len(p),int)
  for j,perm in enumerate(perms):which[np.all(order==perm,axis=1)]=j
  fs=np.take_along_axis(f,order,axis=1);w=np.c_[1-fs[:,0],fs[:,0]-fs[:,1],fs[:,1]-fs[:,2],fs[:,2]]
  return c,which,w
 qc,qt,qb=embed(q);vc,vt,vb=embed(bv)
 keys=np.r_[np.c_[qc,qt],np.c_[vc,vt]];ut,inv=np.unique(keys,axis=0,return_inverse=True);qtid=inv[:len(q)];vtid=inv[len(q):]
 # Full 4-node affine carrier tets. Integration-empty exterior-only tets are mapped to nearest active tet
 # via affine extrapolation, so they never create unsupported degrees of freedom.
 weight=np.bincount(qtid,weights=np.full(len(q),spacing**3),minlength=len(ut));active=np.flatnonzero(weight>0);utactive=ut[active];remap=np.full(len(ut),-1,int);remap[active]=np.arange(len(active));activecent=co+H*(utactive[:,:3]+.5);tree=cKDTree(activecent)
 miss=np.flatnonzero(weight[vtid]==0)
 if len(miss):
  nearest=tree.query(bv[miss],k=1)[1];vtid[miss]=active[nearest]
 qtid=remap[qtid];vtid=remap[vtid];ut=utactive
 vertices=np.repeat(ut[:,:3,None,:] if False else ut[:,:3,None],4,axis=2) if False else None
 itets=np.zeros((len(ut),4,3),int);itets[:,0]=ut[:,:3];itets[:,1]=ut[:,:3];itets[:,2]=ut[:,:3];itets[:,3]=ut[:,:3]+1
 for j,p in enumerate(perms):
  ids=np.flatnonzero(ut[:,3]==j);itets[ids,1,p[0]]+=1
  for ax in p[:2]:itets[ids,2,ax]+=1
 unodes,ti=np.unique(itets.reshape(-1,3),axis=0,return_inverse=True);tets=ti.reshape(-1,4);nodes=co+H*unodes
 # Recompute all affine weights in final ordering and orient every carrier positively.
 x=nodes[tets];D=np.stack([x[:,1]-x[:,0],x[:,2]-x[:,0],x[:,3]-x[:,0]],axis=2);neg=np.linalg.det(D)<0;tets[neg,1:3]=tets[neg,2:0:-1];x=nodes[tets];D=np.stack([x[:,1]-x[:,0],x[:,2]-x[:,0],x[:,3]-x[:,0]],axis=2);invD=np.linalg.inv(D)
 def bary(p,ids):
  w=np.einsum('nij,nj->ni',invD[ids],p-nodes[tets[ids,0]]);return np.c_[1-w.sum(1),w]
 qb=bary(q,qtid);vb=bary(bv,vtid)
 # Positive volume fractions using a 1-L0-thick local integration window. Muscle has native center ownership.
 fractions=np.zeros((len(q),5),np.float32)
 # 0 matrix,1 source muscle,2 fascia,3 fat,4 dermis. No duplicate tissue volume.
 skin=np.clip((d+.5)-(h+.5),0,1)-np.clip((d-.5)-(h+1.5),0,1)
 skin=np.clip(np.minimum(d+.5,h+1.5)-np.maximum(d-.5,h+.5),0,1)
 fas=np.clip(np.minimum(d+.5,.5)-np.maximum(d-.5,0.),0,1)
 adip=np.clip(np.minimum(d+.5,h+.5)-np.maximum(d-.5,.5),0,1)
 fractions[:,4]=skin;fractions[:,2]=fas;fractions[:,3]=adip;remaining=np.maximum(0,1-fractions.sum(1));fractions[:,1]=np.where(mc>0,remaining,0);fractions[:,0]=np.where(mc==0,remaining,0);fractions/=fractions.sum(1)[:,None]
 source_ids=np.array(muscle_ids,np.int16)
 # q source ownership primary is last overlapping native part; all overlaps are count-normalized in companion membership lists later if needed.
 nodeweight=np.zeros(len(nodes));np.add.at(nodeweight,tets[qtid].ravel(),(qb*np.full((len(q),1),spacing**3)).ravel());orphans=np.flatnonzero(nodeweight<=1e-10)
 boundary_owner=np.full(shape,-1,np.int32);boundary_owner[tuple(qijk.T)]=np.arange(len(q));bowner=boundary_owner[tuple(owners.T)]
 np.savez_compressed(G/'whole-face-continuum.npz',nodes=nodes,tets=tets.astype(np.int32),tet_reference_volume=np.abs(np.linalg.det(D))/6,quadrature_points=q.astype(np.float32),quadrature_weights=np.full(len(q),spacing**3,np.float32),quadrature_tet=qtid.astype(np.int32),quadrature_barycentric=qb.astype(np.float32),quadrature_material_fraction=fractions,quadrature_source_part=sp,quadrature_source_overlap_count=mc,quadrature_cell_ijk=qijk,quadrature_component=(comp[tuple(qijk.T)]-1).astype(np.int16),vertices=bv.astype(np.float32),triangles=bf.astype(np.int32),exterior_tet=vtid.astype(np.int32),exterior_barycentric=vb.astype(np.float32),boundary_class=bc,boundary_source_part=support_source,boundary_owner_quadrature=bowner,grid_origin=origin,grid_spacing=np.array(spacing),carrier_spacing=np.array(H),material_ids=np.arange(5,dtype=np.int8),source_muscle_part_ids=source_ids,node_integrated_basis_weight=nodeweight.astype(np.float32),source_sha256=np.array(sha))
 report['mechanical_carrier']={'nodes':len(nodes),'tets':len(tets),'positive_volume_min':float(np.min(np.linalg.det(D)/6)),'quadrature_positive_min':spacing**3,'quadrature_total_volume':float(len(q)*spacing**3),'material_volume':{k:float(fractions[:,j].sum()*spacing**3) for j,k in enumerate(['matrix','muscle','fascia','fat','dermis'])},'fraction_sum_max_error':float(abs(fractions.sum(1)-1).max()),'orphan_dofs':len(orphans),'exterior_vertices_affine_extrapolated_to_active_carrier':len(miss),'exterior_affine_reconstruction_max_error':float(np.linalg.norm(np.einsum('ni,nij->nj',vb,nodes[tets[vtid]])-bv,axis=1).max()),'quadrature_affine_reconstruction_max_error':float(np.linalg.norm(np.einsum('ni,nij->nj',qb,nodes[tets[qtid]])-q,axis=1).max()),'quadrature_barycentric_min':float(qb.min()),'exterior_barycentric_min':float(vb.min()),'integration':'1-point positive voxel-center quadrature through common P1 carrier; geometry quantization independent of DOFs','interface_policy':'Bonded common displacement, normalized constitutive mixture. No muscle contact barriers.'}
 report['seconds']=time.time()-start;report['geometry_sha256']=hashfile(G/'whole-face-continuum.npz');(G/'whole-face-geometry-report.json').write_text(json.dumps(report,indent=2));log('DONE',report['mechanical_carrier'],time.time()-start)
if __name__=='__main__':main()
