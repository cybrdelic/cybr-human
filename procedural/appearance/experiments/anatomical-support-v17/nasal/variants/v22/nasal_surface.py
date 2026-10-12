"""Coherent original anatomical volume, independent of loft/rim constraints.

An implicit smooth union carries the whole host, bridge, tip, alar and
columellar mass. Two finite oriented passages subtract real open vestibules.
Marching tetrahedra permits arbitrary local folds and new topology. An exact
outer attachment annulus preserves the preferred broad face.
"""
from pathlib import Path
import numpy as np,json,sys
from scipy.interpolate import PchipInterpolator
from scipy.spatial import Delaunay
from downstream import downstream,PROJECT
sys.path.insert(0,str(PROJECT/'src'))
from nasal_fresh_supported_base_surface import evaluate as old_surface
from nasal_fresh_basal_control_net import BOUNDARY,width
OUT=Path(__file__).resolve().parents[1];ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'geometry-source'
def smooth(x):
 x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))
def smin(a,b,k):
 h=np.maximum(k-np.abs(a-b),0)/k
 return np.minimum(a,b)-h*h*k*.25
def ellipsoid(p,c,r,axes=None):
 q=p-np.array(c)
 if axes is not None:q=q@axes.T
 a=q/np.array(r);k0=np.linalg.norm(a,axis=-1);k1=np.linalg.norm(a/np.array(r),axis=-1)
 return np.where(k1>1e-10,k0*(k0-1)/np.maximum(k1,1e-10),-min(r))
def tapered_passage(p,c,r,axes):
 # The finite vestibule widens toward the posterior/lateral sill and
 # narrows toward the anterior/medial apex. The connected volume itself
 # carries the rim: no superimposed ring geometry.
 q=(p-np.array(c))@axes.T
 taper=1-.20*np.tanh(1.4*q[:,1]/r[1])
 q[:,0]/=taper
 return ellipsoid(q,[0,0,0],r)
def host(uv):
 y=-40+55*uv[:,1];x=(2*uv[:,0]-1)*width(y)
 base=PchipInterpolator([-40,-32,-20,0,15],[121.1,117,113.8,114.5,119])(y)-.007*x*x
 old=old_surface(uv)[:,2]
 rho=np.sqrt((x/18.)**2+((y+28.)/13.)**2)
 return old-4.2*smooth((1-rho)/.50)
def xyz(uv,z):
 y=-40+55*uv[:,1];return np.c_[(2*uv[:,0]-1)*width(y),y,z]
def field(p,h=None,uv=None):
 if h is None:
  uv=np.c_[.5+p[:,0]/(2*width(p[:,1])),(p[:,1]+40)/55];h=host(uv)
 face=p[:,2]-h
 # A solid dorsal-to-host support volume, specified by broad sagittal
 # sections, prevents an arch tunnel behind an isolated bridge ellipsoid.
 # The rounded tip/alar masses below remain fully three-dimensional.
 yy=np.clip(p[:,1],-40,15)
 front=PchipInterpolator([-40,-35,-30,-26,-20,-12,0,15],[121.1,123.5,134.7,138.2,137.8,133.0,125.0,119.0])(yy)
 spread=PchipInterpolator([-40,-35,-30,-26,-20,-12,0,15],[3.0,5.2,9.7,10.3,9.0,6.4,5.2,4.0])(yy)
 projection=np.maximum(front-h,0)*np.exp(-(p[:,0]/spread)**2)
 d=face-projection
 d=smin(d,ellipsoid(p,[0,-26.0,132.5],[6.4,6.1,8.0]),4.0)
 for sign in [-1,1]:d=smin(d,ellipsoid(p,[sign*11.6,-27.2,123.8],[5.6,5.4,8.0]),4.6)
 d=smin(d,ellipsoid(p,[0,-32.7,124.0],[3.0,3.6,4.8]),3.4)
 d=smin(face,d,2.6)
 for sign in [-1,1]:
  n=np.array([sign*.10,-.91,.40]);n/=np.linalg.norm(n)
  long=np.array([-sign*.58,.22,.785]);long-=n*np.dot(long,n);long/=np.linalg.norm(long)
  short=np.cross(n,long);short/=np.linalg.norm(short)
  centre=np.array([sign*7.15,-31.65,126.0])
  passage=tapered_passage(p,centre+1.3*n,[2.55,4.8,9.0],np.array([short,long,n]))
  d=-smin(-d,passage,.55)
 edge=np.min(np.c_[uv,1-uv],axis=1)
 # Entire implicit addition fades at the actual host attachment, ensuring
 # the original outer curve and derivative survive without a mesh collar.
 return face+(d-face)*smooth(edge/.11)
def frontal_root(uv):
 h=host(uv);lo=np.full(len(uv),97.);hi=np.full(len(uv),149.)
 for _ in range(30):
  z=(lo+hi)/2;d=field(xyz(uv,z),h,uv);lo=np.where(d<0,z,lo);hi=np.where(d>=0,z,hi)
 return (lo+hi)/2
def marching():
 nu,nv,nz=121,151,137;u=np.linspace(.034,.966,nu);v=np.linspace(.034,.966,nv);z=np.linspace(99.,148.,nz)
 uv=np.stack(np.meshgrid(u,v,indexing='ij'),-1).reshape(-1,2);hh=host(uv);values=np.empty((nu*nv,nz),np.float32)
 for start in range(0,len(uv),1000):
  q=uv[start:start+1000];qq=np.repeat(q,nz,axis=0);pp=xyz(qq,np.tile(z,len(q)));values[start:start+len(q)]=field(pp,np.repeat(hh[start:start+len(q)],nz),qq).reshape(len(q),nz)
 values=values.reshape(nu,nv,nz);lower=np.minimum.reduce([values[a:a+nu-1,b:b+nv-1,c:c+nz-1] for a,b,c in [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]])
 upper=np.maximum.reduce([values[a:a+nu-1,b:b+nv-1,c:c+nz-1] for a,b,c in [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]])
 cells=np.argwhere((lower<0)&(upper>=0));offset=np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)])
 corners=cells[:,None]+offset;ids=(corners[:,:,0]*nv+corners[:,:,1])*nz+corners[:,:,2]
 tet=np.array([[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6],[0,5,1,6]])
 t=ids[:,tet].reshape(-1,4);val=values.ravel()[t];inside=val<0;count=inside.sum(1);t=t[(count>0)&(count<4)];inside=inside[(count>0)&(count<4)];count=count[(count>0)&(count<4)]
 # Emit crossed grid edges, then globally weld their unique endpoint keys.
 emitted=[];directions=[]
 def coordinates(ii):
  iz=ii%nz;iy=(ii//nz)%nv;ix=ii//(nz*nv);q=np.c_[u[ix],v[iy]];return xyz(q,z[iz]),q
 for c in [1,2,3]:
  take=count==c;tt=t[take];mask=inside[take]
  ii=np.argsort(~mask,axis=1);ordered=np.take_along_axis(tt,ii,axis=1)
  if c==1:
   ed=np.stack([np.stack([ordered[:,0],ordered[:,j]],axis=1) for j in [1,2,3]],axis=1)
   emitted.append(ed);dirs=coordinates(ordered[:,1:].reshape(-1))[0].reshape(-1,3,3).mean(1)-coordinates(ordered[:,0])[0];directions.append(dirs)
  elif c==3:
   ed=np.stack([np.stack([ordered[:,j],ordered[:,3]],axis=1) for j in [0,1,2]],axis=1)
   emitted.append(ed);dirs=coordinates(ordered[:,3])[0]-coordinates(ordered[:,:3].reshape(-1))[0].reshape(-1,3,3).mean(1);directions.append(dirs)
  else:
   edges=np.stack([np.stack([ordered[:,i],ordered[:,j]],axis=1) for i,j in [(0,2),(0,3),(1,3),(1,2)]],axis=1)
   dirs=coordinates(ordered[:,2:].reshape(-1))[0].reshape(-1,2,3).mean(1)-coordinates(ordered[:,:2].reshape(-1))[0].reshape(-1,2,3).mean(1)
   emitted.extend([edges[:,[0,1,2]],edges[:,[0,2,3]]]);directions.extend([dirs,dirs])
 all_edges=np.sort(np.concatenate(emitted).reshape(-1,2),axis=1);unique,inverse=np.unique(all_edges,axis=0,return_inverse=True);fa=values.ravel()[unique[:,0]];fb=values.ravel()[unique[:,1]];a,auv=coordinates(unique[:,0]);b,buv=coordinates(unique[:,1]);w=(fa/(fa-fb))[:,None];points=a+(b-a)*w;uv=auv+(buv-auv)*w;faces=inverse.reshape(-1,3)
 tri=points[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);flip=(normal*np.concatenate(directions)).sum(1)<0;faces[flip]=faces[flip][:,[0,2,1]]
 return points,uv,faces
def build():
 cut=np.load(SOURCE/'dependencies/nasal_cut.npz');nb=len(BOUNDARY);p,uv,f=marching()
 edges=np.r_[f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]];_,iv,ct=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True);be=edges[ct[iv]==1];inner=np.unique(be)
 # The only marching boundary must be the four outer domain planes. A
 # hidden rear cap or stray disconnected surface cannot be silently added.
 q=uv[inner];assert np.all(np.min(np.abs(np.c_[q-.034,q-.966]),axis=1)<1e-9)
 buv=np.c_[.5+BOUNDARY[:,0]/(2*width(BOUNDARY[:,1])),(BOUNDARY[:,1]+40)/55];buv[np.abs(buv)<1e-9]=0;buv[np.abs(buv-1)<1e-9]=1
 # Join the two convex attachment loops by angular zipper. Unlike an
 # unconstrained planar triangulation, every inner and outer boundary edge
 # is represented once; no centroid clipping can drop a corner triangle.
 outer_angle=np.arctan2(buv[:,1]-.5,buv[:,0]-.5);inner_angle=np.arctan2(q[:,1]-.5,q[:,0]-.5)
 oo=np.argsort(outer_angle);ii=np.argsort(inner_angle);oa=outer_angle[oo];ia=inner_angle[ii]
 source=np.r_[BOUNDARY,p];rings=[];i=j=0
 while i<len(oo) or j<len(ii):
  onext=oa[(i+1)%len(oo)]+(2*np.pi if i+1>=len(oo) else 0)
  inext=ia[(j+1)%len(ii)]+(2*np.pi if j+1>=len(ii) else 0)
  if i<len(oo) and (j>=len(ii) or onext<=inext):
   rings.append([oo[i%len(oo)],oo[(i+1)%len(oo)],nb+inner[ii[j%len(ii)]]]);i+=1
  else:
   rings.append([oo[i%len(oo)],nb+inner[ii[(j+1)%len(ii)]],nb+inner[ii[j%len(ii)]]]);j+=1
 af=np.array(rings);alluv=np.r_[buv,uv];aa=alluv[af[:,1]]-alluv[af[:,0]];bb=alluv[af[:,2]]-alluv[af[:,0]];flip=aa[:,0]*bb[:,1]-aa[:,1]*bb[:,0]<0;af[flip]=af[flip][:,[0,2,1]]
 ff=np.r_[nb+f,af]
 # Collapse only sub-micron existing marching edges, never host boundary
 # vertices. Near-isovalue grid corners otherwise produce sliver triangles.
 collapsed=0
 for _ in range(3):
  ee=np.unique(np.sort(np.r_[ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]],axis=1),axis=0)
  tiny=ee[(ee.min(1)>=nb)&(np.linalg.norm(source[ee[:,0]]-source[ee[:,1]],axis=1)<1e-5)]
  if not len(tiny):break
  remap=np.arange(len(source))
  for aa,bb in tiny:
   while remap[aa]!=aa:aa=remap[aa]
   while remap[bb]!=bb:bb=remap[bb]
   mapped=remap[ff]
   na=set(np.unique(mapped[(mapped==aa).any(1)]))-{aa,bb}
   nbhd=set(np.unique(mapped[(mapped==bb).any(1)]))-{aa,bb}
   edge_faces=((mapped==aa).any(1)&(mapped==bb).any(1)).sum()
   if edge_faces==2 and len(na&nbhd)==2:remap[max(aa,bb)]=min(aa,bb)
  for i in range(len(remap)):
   while remap[i]!=remap[remap[i]]:remap[i]=remap[remap[i]]
  ff=remap[ff];ff=ff[(ff[:,0]!=ff[:,1])&(ff[:,1]!=ff[:,2])&(ff[:,2]!=ff[:,0])]
  used=np.unique(ff);compact=np.full(len(source),-1,np.int64);compact[used]=np.arange(len(used));collapsed+=len(source)-len(used);source=source[used];ff=compact[ff]
 vv=downstream(source);error=float(np.abs(vv[:nb]-cut['boundary_points']).max());assert error<1e-9;vv[:nb]=cut['boundary_points']
 edge=np.r_[ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]];_,iv,ct=np.unique(np.sort(edge,axis=1),axis=0,return_inverse=True,return_counts=True);wind=np.bincount(iv,weights=np.where(edge[:,0]<edge[:,1],1,-1));tri=vv[ff];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
 report={'representation':'One coherent implicit anatomical volume with oriented finite passages, marching tetrahedra and exact host attachment annulus','vertices':len(vv),'triangles':len(ff),'minimum_area':float(area.min()),'boundary_edges':int((ct==1).sum()),'nonmanifold_edges':int((ct>2).sum()),'inconsistent_winding':int(((ct==2)&(wind!=0)).sum()),'seam_error':error,'collapsed_numerical_sliver_vertices':collapsed,'euler_characteristic':int(len(vv)-len(ct)+len(ff)),'selected':False}
 if report['boundary_edges']!=nb:
  bad=edge[(ct[iv]==1)&(edge.min(1)>=nb)];print('EXTRA_BOUNDARY',source[bad].tolist(),flush=True)
 assert report['boundary_edges']==nb and report['nonmanifold_edges']==report['inconsistent_winding']==0 and report['euler_characteristic']==1,report
 data={'vertices':vv,'source_vertices':source,'triangles':ff,'face_parts':np.zeros(len(ff),np.uint8),'boundary_head_vertex_ids':cut['boundary_head_vertex_ids'],'boundary_vertex_ids':np.arange(nb),'root_vertex_ids':np.empty((0,0),np.int64),'implicit_uv':uv}
 return data,report
if __name__=='__main__':
 data,r=build();np.savez_compressed(OUT/'nasal_patch.npz',**data);(OUT/'LOCAL_SURFACE_REPORT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
