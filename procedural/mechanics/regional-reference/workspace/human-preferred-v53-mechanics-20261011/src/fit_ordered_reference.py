"""Ordered ambient registration with a monotone height map and common depth field.
Exact selected v53 skin is an immutable target, not an edited simulation boundary.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.interpolate import RBFInterpolator,PchipInterpolator
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];O=B/'rest-fit-v4';C=R/'human-mechanics-20261010/output/bilateral-distributed-mechanics-v1'
sys.path.insert(0,str(R/'human-mechanics-20261010/src'));from build_bilateral_cavity_case import read_arrays
q=np.load(B/'rest-targets-v1/targets.npz');d=json.loads((C/'domain.json').read_text());a=read_arrays(C,d);x=a['rest']*1000;t=a['nodes'];sn=q['surface_nodes'];rn=q['rigid_nodes'];want=q['targets'][sn]-x[sn]
# A prescribed monotone height relationship, not independent oral-chart motion.
yk=np.array([-100,-80,-60,-50,-40,-25,-12,-9,0,20,60.]);shift=np.array([0,1,2.8,2.8,2,.6,0,0,0,0,0.]);yshift=PchipInterpolator(yk,shift)
yder=1+yshift.derivative()(np.linspace(-100,60,10001));assert yder.min()>0
key=np.round(x[sn,:2]/2.5).astype(int);_,ii=np.unique(key,axis=0,return_inverse=True);n=np.bincount(ii);p=np.array([np.bincount(ii,weights=x[sn,k])/n for k in range(2)]).T;dd=np.bincount(ii,weights=want[:,2])/n
field=RBFInterpolator(p/20,dd,kernel='thin_plate_spline',smoothing=.025,degree=1);roots=cKDTree(x[rn,:2])
def map_points(xx):
 out=xx.copy();out[:,1]+=yshift(xx[:,1]);dist=roots.query(xx[:,:2])[0];u=np.clip((dist-2)/14,0,1);fade=u**3*(10-15*u+6*u*u);out[:,2]+=field(xx[:,:2]/20)*fade;return out
xx=map_points(x);assert np.array_equal(xx[rn],x[rn]);D=lambda y:np.stack([y[t[:,j]]-y[t[:,0]] for j in [1,2,3]],axis=-1);J=np.linalg.det(D(xx)@np.linalg.inv(D(x)));res=np.linalg.norm(xx[sn]-q['targets'][sn],axis=1)
# Bone uses the same map plus its separate C2 lateral-rim clearance correction,
# retained exactly at the entire named root footprint.
bv=q['original_bone_vertices'];bone=map_points(bv);bt=q['bone_triangles'];S=lambda u:np.clip(u,0,1)**3*(10-15*np.clip(u,0,1)+6*np.clip(u,0,1)**2);w=S((abs(bv[:,0])-45)/10)*S((bv[:,1]+1)/8)*(1-S((bv[:,1]-22)/12));bone[:,0]-=np.sign(bv[:,0])*3*w;assert np.array_equal(bone[q['root_bone_vertex_ids']],bv[q['root_bone_vertex_ids']])
normal=lambda z:np.cross(z[bt[:,1]]-z[bt[:,0]],z[bt[:,2]]-z[bt[:,0]]);n0=normal(bv);n1=normal(bone);cos=np.sum(n0*n1,1)/np.linalg.norm(n0,axis=1)/np.linalg.norm(n1,axis=1)
O.mkdir(exist_ok=False);np.savez_compressed(O/'candidate_geometry.npz',vertices=xx,tetrahedra=t,boundary_faces=np.array(d['boundary_triangles']),bone_vertices=bone,bone_triangles=bt,surface_nodes=sn,rigid_nodes=rn,original_rest=x,control_points=p,control_displacements=dd,height_knots=yk,height_targets=yk+shift,displacement=xx-x)
report={'status':'Ordered continuous registration candidate; containment/physics pending','preferred_skin_modified':False,'control_count':len(p),'method':'X unchanged; monotone PCHIP Y map; one common smooth Z(X,Y) depth translation with exact root holdout. Distinct oral charts cannot separate coincident neighbors.','analytic_ambient_J_minimum_sampled_Y_derivative':float(yder.min()),'relative_discrete_J_range':[float(J.min()),float(J.max())],'nonpositive_cells':int((J<=0).sum()),'below_point15':int((J<=.15).sum()),'rigid_nodes_exact':np.array_equal(xx[rn],x[rn]),'surface_target_residual_quantiles':np.percentile(res,[0,50,95,100]).tolist(),'maximum_node_motion':float(np.linalg.norm(xx-x,axis=1).max()),'original_corner_pair_distance':float(np.linalg.norm(x[7594]-x[7578])),'mapped_corner_pair_distance':float(np.linalg.norm(xx[7594]-xx[7578])),'bone_minimum_normal_cosine':float(cos.min()),'bone_nonpositive_normals':int((cos<=0).sum()),'limits':['Continuum authoring only; no mechanics state reused.','Nonlinear-map positivity does not prove positive straight-sided tets, global overlap or exact skin containment.']};(O/'fit_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
