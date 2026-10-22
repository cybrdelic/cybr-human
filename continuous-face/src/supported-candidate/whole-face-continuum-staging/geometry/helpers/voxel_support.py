"""Bounded continuous exterior contact from the actual anatomical envelope.

A star-shaped outer envelope is sampled from selected real source surfaces.
Small angular gaps are interpolated before a 2-design-unit voxel distance field
is built. This deliberately closes exterior cavities at coarse scale; internal
oral/nasal folds require separate material regions. No preferred skin position
is read. It is an approximation, not a union/contact certificate.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt,gaussian_filter,map_coordinates
from exterior_support import ExteriorSupport,KIND_NAMES

class VoxelSupport:
 def __init__(self,packet,meta,already_registered=True,spacing=2.,include_superficial_connective=True,neck_support=False,include_external_gingiva=True,surface_sampling_spacing=None,root_connective_closure_width=0.,neck_join=False):
  surface=ExteriorSupport(packet,meta,already_registered=already_registered,include_superficial_connective=include_superficial_connective,angular_radius=0,include_external_gingiva=include_external_gingiva)
  self.source=surface;self.radial_center=surface.radial_center;self.spacing=float(spacing)
  # Independent local surface exclusion prevents the coarse/smoothed field
  # from allowing visible source tissue through the skin. This does not add
  # any preferred facial position or alter the authored fat rest band.
  from mesh_queries import SurfaceIndex
  self.surface_index=SurfaceIndex(surface.vertices,surface.faces[surface.face_ids])
  self.areas=surface.areas;self.centers=surface.centers
  points=np.r_[surface.triangles.reshape(-1,3),surface.centers]
  # Numerical sampling guard for explicitly selected external gum walls.
  # Keep every original sample and add a half-voxel outward copy. This pads
  # only the coarse support construction; native/rendered Gingiva is intact.
  gum=surface.kind==KIND_NAMES.index('gum')
  gum_padding=.5*self.spacing if np.any(gum) else 0.
  if gum_padding:
   points=np.r_[points,
    (surface.triangles[gum]+gum_padding*surface.normals[gum,None,:]).reshape(-1,3),
    surface.centers[gum]+gum_padding*surface.normals[gum]]
  d=points-self.radial_center;r=np.linalg.norm(d,axis=1);directions=d/np.maximum(r[:,None],1e-12)
  self.angular_shape=(256,512)
  lat=np.arccos(np.clip(directions[:,1],-1,1))*(self.angular_shape[0]-1)/np.pi
  lon=(np.arctan2(directions[:,2],directions[:,0])+np.pi)*self.angular_shape[1]/(2*np.pi)
  ri=np.rint(lat).astype(int);ci=np.rint(lon).astype(int)%self.angular_shape[1]
  radius=np.full(self.angular_shape,-np.inf);np.maximum.at(radius,(ri,ci),r)
  self.surface_sampling_spacing=surface_sampling_spacing
  self.surface_sampling_report=None
  self.root_connective_closure_width=float(root_connective_closure_width)
  self.root_connective_closure_report=None;self.root_connective_ribbons=None
  envelope_triangles=surface.triangles
  if self.root_connective_closure_width:
   if not include_superficial_connective:raise ValueError('Root connective closure requires superficial connective support')
   if surface_sampling_spacing is None:raise ValueError('Root connective closure requires area-covering triangle sampling')
   from surface_envelope_sampling import root_connective_ribbons
   from auricular_support import DEFAULT_PACKET
   with np.load(DEFAULT_PACKET) as auricle:ear_vertices=auricle['vertices'].copy()
   self.root_connective_ribbons,self.root_connective_closure_report=root_connective_ribbons(surface,packet['part_ids'],ear_vertices,self.root_connective_closure_width)
   authored=self.root_connective_ribbons['triangles']
   envelope_triangles=np.concatenate([surface.triangles,authored])
   points=np.concatenate([points,authored.reshape(-1,3)])
  if surface_sampling_spacing is not None:
   from surface_envelope_sampling import raster_outer_radius
   radius,self.surface_sampling_report=raster_outer_radius(envelope_triangles,self.radial_center,self.angular_shape,float(surface_sampling_spacing),extra_points=points)
   self.surface_sampling_report['actual_source_triangles']=len(surface.triangles)
   self.surface_sampling_report['authored_connective_triangles']=len(envelope_triangles)-len(surface.triangles)
  valid=np.isfinite(radius);self.angular_covered=float(valid.mean())
  # Periodic nearest fill is solely an exterior-envelope construction. Keep
  # this explicit instead of silently treating holes as body-free space.
  tiled=np.tile(radius,(1,3));mask=~np.isfinite(tiled)
  nearest=distance_transform_edt(mask,return_distances=False,return_indices=True)
  filled=tiled[tuple(nearest)][:,self.angular_shape[1]:2*self.angular_shape[1]]
  self.radius=gaussian_filter(filled,sigma=(.8,.8),mode=('nearest','wrap'))
  self.neck_join=bool(neck_join)
  if self.neck_join and not neck_support:raise ValueError("neck_join requires neck_support")
  self.neck=None;bounds_points=points
  if neck_support:
   from neck_support import NeckSupport
   self.neck=(NeckSupport.joined_to_head(packet,meta,self.radial_center,self.radius_at,
    head_max_radius=float(self.radius.max()),voxel_spacing=self.spacing,already_registered=True)
    if self.neck_join else NeckSupport(packet,meta,already_registered=True))
   yy,theta=np.meshgrid(np.linspace(*self.neck.y_bounds,80),np.linspace(-np.pi,np.pi,256,endpoint=False),indexing='ij')
   center=self.neck.center_at(yy.ravel());radius=self.neck.radial_at(yy.ravel(),theta.ravel())
   # Neck center_at returns XZ anatomical centerline coordinates.
   neck_points=np.c_[center[:,0]+radius*np.cos(theta.ravel()),yy.ravel(),center[:,1]+radius*np.sin(theta.ravel())]
   bounds_points=np.r_[points,neck_points]
  lo=np.floor((bounds_points.min(0)-24)/spacing)*spacing;hi=np.ceil((bounds_points.max(0)+24)/spacing)*spacing
  self.origin=lo;self.shape=tuple((np.rint((hi-lo)/spacing).astype(int)+1).tolist())
  occupancy=np.zeros(self.shape,bool)
  yy=lo[1]+np.arange(self.shape[1])*spacing;zz=lo[2]+np.arange(self.shape[2])*spacing
  for i in range(self.shape[0]):
   xx=np.full((self.shape[1],self.shape[2]),lo[0]+i*spacing);y,z=np.meshgrid(yy,zz,indexing='ij');p=np.stack([xx,y,z],-1)
   dr=p-self.radial_center;length=np.linalg.norm(dr,axis=-1);u=dr/np.maximum(length[...,None],1e-9)
   boundary=self.radius_at(u.reshape(-1,3)).reshape(length.shape);occupancy[i]=length<=boundary
   if self.neck is not None:occupancy[i]|=self.neck.occupancy(p)
  # Signed field uses the same voxel boundary for inside and outside.
  self.field=((distance_transform_edt(~occupancy)-distance_transform_edt(occupancy))*spacing).astype(np.float32)
  self.field=gaussian_filter(self.field,1.0,mode='nearest')
  self.gradient=np.asarray(np.gradient(self.field,spacing),np.float32)
  self.summary={'method':'selected actual anatomy -> sampled star-shaped exterior envelope -> continuous voxel distance field',
   'sdf_smoothing_voxels':1.0,'spacing_design_units':spacing,'grid_shape':list(self.shape),'grid_nodes':int(np.prod(self.shape)),
   'angular_sample_coverage':self.angular_covered,'triangle_interior_sampling':self.surface_sampling_report,'root_connective_closure':self.root_connective_closure_report,'source':surface.summary,'sectioned_neck':self.neck.summary if self.neck is not None else None,
   'local_actual_surface_contact_guard':{'clearance':.5,'query_centroid_band':8.,'law':'minimum residual of coarse field and continuous selected-source oriented gap','derivative':'same scalar branch exact derivative; no unit-normal substitution','limits':'finite candidate nearest patches, not all-body exact union; no change to fat rest-volume field'},
   'external_gingiva_sampling_guard':{'outward_padding_design_units':gum_padding,'voxel_fraction':.5,
    'selected_source_faces':int(gum.sum()),'added_samples':int(4*gum.sum()),
    'method':'additional original-face-normal-offset samples for selected external gum walls only; original samples retained',
    'visible_geometry':'unchanged native Gingiva mesh',
    'limits':'Numerical resolution margin; angular fill and smoothing can still under-cover unsampled geometry; no exact containment guarantee'},
   'limits':'Coarse outer envelope closes cavity openings; nearest angular fill, voxel aliasing and 0.8-bin smoothing; no exact union; no ear support'}
 def radius_at(self,direction):
  d=np.asarray(direction);lat=np.arccos(np.clip(d[:,1],-1,1))*(self.angular_shape[0]-1)/np.pi
  lon=(np.arctan2(d[:,2],d[:,0])+np.pi)*self.angular_shape[1]/(2*np.pi)
  # Explicit longitude extension avoids interpolation discontinuity at seam.
  extended=np.c_[self.radius,self.radius[:,:1]]
  return map_coordinates(extended,[lat,lon],order=1,mode='nearest')
 def field_value_gradient(self,points):
  """Value and exact within-cell derivative of the same trilinear field."""
  p=np.asarray(points);coord=(p-self.origin)/self.spacing
  valid=np.all((coord>=1)&(coord<np.array(self.shape)-2),axis=1)
  coord=np.clip(coord,0,np.array(self.shape)-1-1e-8)
  base=np.floor(coord).astype(int);t=coord-base
  value=np.zeros(len(p));gradient=np.zeros_like(p,dtype=float)
  for i in range(2):
   for j in range(2):
    for k in range(2):
     bits=np.array([i,j,k]);w=np.where(bits,t,1-t)
     f=self.field[base[:,0]+i,base[:,1]+j,base[:,2]+k]
     value+=f*np.prod(w,axis=1)
     for axis in range(3):
      others=[a for a in range(3) if a!=axis]
      gradient[:,axis]+=f*(1 if bits[axis] else -1)*np.prod(w[:,others],axis=1)/self.spacing
  return value,gradient,valid
 def query(self,points,**kwargs):
  p=np.asarray(points);d,grad,valid=self.field_value_gradient(p)
  norm=np.linalg.norm(grad,axis=1);n=grad/np.maximum(norm[:,None],1e-12);valid&=norm>.05;n[~valid]=0;grad[~valid]=0
  # One Newton support estimate, used only for display/correspondence. The
  # energy and contact use the actual field value and its raw derivative.
  q=p-d[:,None]*grad/np.maximum(norm[:,None]**2,1e-12)
  return {'points':q,'normals':n,'gradient':grad,'valid':valid,'distance':abs(d),'signed_gap':d,'kind':np.full(len(p),-1,np.int8),
          'part_id':np.full(len(p),-1,int),'face_id':np.full(len(p),-1,int),'barycentric':np.zeros_like(p)}
 def contact_query(self,points):
  """Coarse outer exclusion plus a finite selected-source clearance guard.

  The actual outward patches get a .5-unit nonpenetration margin. Their
  local oriented signs/candidate search remain approximate. Specific oral,
  nasal, ocular and auricular material laws supersede this in FreeSkin.
  """
  finite_guard=getattr(self,'finite_static_guard',None)
  if finite_guard is not None:return finite_guard.apply(points,self.query(points))
  from nasal_support import _oriented_gap
  p=np.asarray(points);r=self.query(p);minimum=np.full(len(p),1.35)
  source_distance=self.surface_index.tree.query(p,k=1)[0]
  rows=np.flatnonzero(source_distance<8.)
  if len(rows):
   s=self.source
   q,g,n,d,valid=_oriented_gap(p[rows],s.vertices,s.faces[s.face_ids],s.vertex_normals,self.surface_index)
   take=valid&((d-.5)<(r['signed_gap'][rows]-minimum[rows]))
   rr=rows[take];r['points'][rr]=q[take];r['gradient'][rr]=g[take];r['normals'][rr]=n[take]
   r['signed_gap'][rr]=d[take];r['valid'][rr]=True;minimum[rr]=.5
  r['minimum_gap']=minimum
  return r
 def enclosing_initialization(self,c,offset):
  d=c-self.radial_center;r=np.linalg.norm(d,axis=1);direction=d/np.maximum(r[:,None],1e-12);initial=c.copy();active=c[:,1]>=getattr(self,'minimum_material_y',-102)
  initial[active]=self.radial_center+(self.radius_at(direction[active])+offset)[:,None]*direction[active]
  info={'method':'same sampled actual anatomical exterior envelope','enclosing_offset':offset,'preferred_position_tether':False}
  if self.neck is not None:
   neck,provenance=self.neck.enclosing_neck_points(c,offset,join_y=-35)
   blend=np.clip((-35-c[:,1])/20,0,1)*active
   initial=initial*(1-blend[:,None])+neck*blend[:,None];info['neck']=provenance
  return initial,info
 def fat_rest_volume(self,fat_layer,multiplier,skin=1.35):
  if multiplier==0:return 0.
  volume=0;yy=self.origin[1]+np.arange(self.shape[1])*self.spacing;zz=self.origin[2]+np.arange(self.shape[2])*self.spacing
  for i in range(self.shape[0]):
   y,z=np.meshgrid(yy,zz,indexing='ij');x=np.full_like(y,self.origin[0]+i*self.spacing);p=np.stack([x,y,z],-1).reshape(-1,3)
   d=self.field[i].ravel();h=multiplier*fat_layer.thickness(p)
   volume+=int(np.sum((d>=skin)&(d<skin+h)&(p[:,1]>getattr(self,'minimum_material_y',-102))))
  return float(volume*self.spacing**3)
