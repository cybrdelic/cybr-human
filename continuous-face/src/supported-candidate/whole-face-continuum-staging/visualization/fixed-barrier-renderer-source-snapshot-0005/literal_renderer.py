"""Literal P1 boundary renderer; pixel shading never changes world positions."""
from __future__ import annotations
import ctypes as C
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
ATLAS = BASE / 'full-face-coverage-plan/restored/exact-rest/anatomy-registered-rest.npz'
ATLAS_SHA = '2555740759e8dd2f50a504914d02f582ba13725c8c77da1fbbb67c55f4c9978d'
BONES = list(range(19)) + [80, 83, 84]
PALETTE = np.array([[131,185,203], [80,65,72], [48,62,73], [48,62,73],
                    [66,61,72], [66,61,72], [114,118,110], [56,79,92],
                    [114,118,110]], dtype=np.float64)
# Class 8 is a declared domain/exclusion wall, not an invented exterior surface.
# Give it the same display-only color as the rigid-facing class 6.
CAMERA_ANGLES = (0., 32., 90.)

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def array(a, dtype, shape_tail):
    source = np.asarray(a)
    if np.issubdtype(np.dtype(dtype), np.integer):
        if not np.issubdtype(source.dtype, np.integer): raise ValueError('Noninteger index/class array')
        limits = np.iinfo(dtype)
        if source.size and (source.min() < limits.min or source.max() > limits.max):
            raise ValueError('Index/class outside target integer range')
    a = np.ascontiguousarray(a, dtype=dtype)
    if a.ndim != len(shape_tail)+1 or a.shape[1:] != shape_tail or not np.isfinite(a).all():
        raise ValueError('Wrong array shape or non-finite values')
    return a

def reconstruct_boundary(q, tets, parent, bary):
    """Only mapping operation allowed: sum_i bary[v,i] * saved_q[tets[parent[v],i]]."""
    q = array(q, np.float64, (3,))
    if not np.issubdtype(np.asarray(tets).dtype, np.integer): raise ValueError('Noninteger tets')
    if not np.issubdtype(np.asarray(parent).dtype, np.integer): raise ValueError('Noninteger parent')
    tets = array(tets, np.int32, (4,)); parent = array(parent, np.int32, ())
    bary = array(bary, np.float64, (4,))
    if len(bary) != len(parent) or not len(tets) or not len(parent): raise ValueError('Empty/mismatched mapping')
    if tets.min()<0 or tets.max()>=len(q) or parent.min()<0 or parent.max()>=len(tets):
        raise ValueError('Mapping index outside actual saved model')
    if abs(bary.sum(1)-1).max()>1e-8 or bary.min() < -1e-8:
        raise ValueError('Invalid P1 barycentric map')
    result = np.einsum('vi,vij->vj', bary, q[tets[parent]], optimize=False)
    if not np.isfinite(result).all(): raise ValueError('Non-finite reconstructed boundary')
    return np.ascontiguousarray(result)

def project(vertices, degrees):
    v = array(vertices, np.float64, (3,))
    if not np.isfinite(degrees): raise ValueError('Non-finite camera')
    a = np.deg2rad(degrees); c, s = np.cos(a), np.sin(a)
    with np.errstate(over='ignore', invalid='ignore'):
        p = np.column_stack((c*v[:,0]+s*v[:,2], v[:,1], -s*v[:,0]+c*v[:,2]))
    if not np.isfinite(p).all(): raise ValueError('Non-finite projection')
    return p

@dataclass(frozen=True)
class Camera:
    degrees: float
    width: int
    height: int
    center_x: float
    center_y: float
    pixels_per_unit: float
    depth_low: float
    depth_high: float

    def check_boundary(self, vertices):
        p = project(vertices, self.degrees)
        x = (p[:,0]-self.center_x)*self.pixels_per_unit+self.width/2
        y = self.height/2-(p[:,1]-self.center_y)*self.pixels_per_unit
        if x.min()<0 or x.max()>self.width or y.min()<0 or y.max()>self.height:
            raise ValueError('Boundary exceeds fixed camera; enlarge once for entire sequence, never per frame')
        return p

def make_cameras(reference, width=700, height=790, padding=.10):
    if width<1 or height<1 or not 0<padding<1: raise ValueError('Invalid viewport')
    projected = [project(reference, angle) for angle in CAMERA_ANGLES]
    ranges = [np.ptp(p[:,:2],axis=0) for p in projected]
    # Identical pixels/unit across views and both axes. Fixed for entire run.
    span_x = max(float(r[0]) for r in ranges)*(1+2*padding)
    span_y = max(float(r[1]) for r in ranges)*(1+2*padding)
    scale = min(width/max(span_x,1e-12), height/max(span_y,1e-12))
    result=[]
    for angle,p in zip(CAMERA_ANGLES, projected):
        center=(p[:,:2].min(0)+p[:,:2].max(0))/2
        low,high=np.percentile(p[:,2],[2,98])
        result.append(Camera(angle,width,height,*map(float,center),scale,float(low),float(high)))
    return result

def build_library():
    source=HERE/'rasterizer.cpp'; lib=HERE/'rasterizer.so'; record=HERE/'build.json'
    source_sha=sha_file(source)
    if not lib.exists() or not record.exists() or json.loads(record.read_text()).get('source_sha256')!=source_sha:
        command=['/usr/bin/c++','-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(lib)]
        subprocess.run(command,check=True,capture_output=True,text=True)
        record.write_text(json.dumps({'command':command,'source_sha256':source_sha,
            'library_sha256':sha_file(lib)},indent=2)+'\n')
    return lib

class Rasterizer:
    def __init__(self):
        self.lib=C.CDLL(str(build_library())); self.fn=self.lib.rasterize
        D=np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        I=np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        U=np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
        L=np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        self.fn.argtypes=[D,C.c_int64,I,C.c_int64,U,D,C.c_int,C.c_int,C.c_int,
            C.c_double,C.c_double,C.c_double,C.c_int,C.c_double,C.c_double,C.c_int,
            U,D,L,C.c_int64,L]; self.fn.restype=C.c_int

    def buffers(self, camera):
        h,w=camera.height,camera.width
        return (np.full((h,w,3),255,np.uint8),np.full((h,w),-np.inf),
                np.full((h,w),np.iinfo(np.int64).max,np.int64))

    def draw(self, buffers, projected, faces, classes, camera, palette=PALETTE,
             lighting=True, depth_cue=True, owner_offset=0):
        p=array(projected,np.float64,(3,))
        if not np.issubdtype(np.asarray(faces).dtype,np.integer): raise ValueError('Noninteger faces')
        f=array(faces,np.int32,(3,)); c=array(classes,np.uint8,()); pal=array(palette,np.float64,(3,))
        if len(c)!=len(f): raise ValueError('Triangle classes mismatch')
        counts=np.zeros(5,np.int64)
        rc=self.fn(p,len(p),f,len(f),c,pal,len(pal),camera.width,camera.height,
            camera.center_x,camera.center_y,camera.pixels_per_unit,int(lighting),
            camera.depth_low,camera.depth_high,int(depth_cue),*buffers,owner_offset,counts)
        if rc: raise ValueError(f'Rasterizer rejected input (code {rc})')
        return dict(zip(('submitted','edge_on','outside','covered_samples','depth_writes'),map(int,counts)))

def load_geometry(path):
    path=Path(path); identity=sha_file(path)
    with np.load(path,allow_pickle=False) as z:
        g={k:z[k] for k in ('nodes','tets','vertices','triangles','exterior_tet',
            'exterior_barycentric','boundary_class','source_sha256')}
    if str(g['source_sha256'])!=ATLAS_SHA: raise ValueError('Unexpected source identity')
    reconstructed=reconstruct_boundary(g['nodes'],g['tets'],g['exterior_tet'],g['exterior_barycentric'])
    error=float(np.max(np.abs(reconstructed-g['vertices'])))
    if error>1e-4: raise ValueError('Reference mapping inconsistent with literal boundary')
    g['path']=str(path.resolve()); g['sha256']=identity; g['rest_mapping_max_abs_error']=error
    return g

def load_bones(scale_mm_per_L0=1.):
    if sha_file(ATLAS)!=ATLAS_SHA: raise ValueError('Exact-rest atlas identity changed')
    with np.load(ATLAS,allow_pickle=False) as z:
        faces=z['triangles'][np.isin(z['part_ids'],BONES)]
        index,inverse=np.unique(faces,return_inverse=True)
        vertices=z['vertices'][index]*scale_mm_per_L0
    return array(vertices,np.float64,(3,)),array(inverse.reshape(-1,3),np.int32,(3,))

def font(size):
    path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(path,size) if Path(path).exists() else ImageFont.load_default()

def centered(draw,y,text,size,color='#354551'):
    draw.text((1125,y),text,font=font(size),fill=color,anchor='mt')

class Scene:
    def __init__(self,geometry,scale_mm_per_L0=1.,cameras=None):
        self.g=geometry
        if not np.isfinite(scale_mm_per_L0) or scale_mm_per_L0<=0: raise ValueError('Invalid physical unit scale')
        self.scale=float(scale_mm_per_L0)
        reference=geometry['vertices']*self.scale
        self.cameras=cameras or make_cameras(reference)
        self.raster=Rasterizer()
        self.bones,self.bone_faces=load_bones(self.scale)
        self.bone_projections=[project(self.bones,c.degrees) for c in self.cameras]
        self.bone_classes=np.zeros(len(self.bone_faces),np.uint8)
        self.bone_palette=np.array([[191.,193.,188.]])

    def render(self,vertices,*,reference_only,physical_time_s=None,case='',playback_note='',contact_mode='finite_surface',site_resolution_mm=None,contact_enforcement=None):
        if not reference_only and (physical_time_s is None or not np.isfinite(physical_time_s) or physical_time_s<0):
            raise ValueError('Actual saved physical time is required')
        start=time.perf_counter(); img=Image.new('RGB',(2250,1200),'#f4f6f8'); d=ImageDraw.Draw(img)
        stats=[]
        for i,(camera,bones) in enumerate(zip(self.cameras,self.bone_projections)):
            p=camera.check_boundary(vertices); buffers=self.raster.buffers(camera)
            a=self.raster.draw(buffers,p,self.g['triangles'],self.g['boundary_class'],camera)
            b=self.raster.draw(buffers,bones,self.bone_faces,self.bone_classes,camera,
                palette=self.bone_palette,depth_cue=False,owner_offset=len(self.g['triangles']))
            img.paste(Image.fromarray(buffers[0]),(45+735*i,180))
            d.text((395+735*i,135),('Front','Oblique 32 deg','Profile')[i],font=font(24),
                fill='#354551',anchor='mt')
            stats.append({'boundary':a,'bone':b,'pixels_visible':int(np.isfinite(buffers[1]).sum())})
        centered(d,24,'CYBR HUMAN / literal whole-face continuum',34,'#20313d')
        if reference_only:
            centered(d,77,'UNSOLVED REFERENCE ONLY | NO ACCEPTED PHYSICAL FRAME | NOT AN ANIMATION',23,'#943c35')
        else:
            prefix='COARSE BARRIER TRANSIENT' if contact_enforcement=='compact_cap_barrier_v1' else ('COARSE SITE-CONTACT TRANSIENT' if contact_mode=='coarse_physical_sites' else 'Accepted saved transient')
            centered(d,77,f'{prefix} | physical t = {physical_time_s:.9g} s | {case}',23)
            if playback_note: centered(d,110,playback_note,17)
        centered(d,995,'Blue: exterior. Dark: cavities/inner walls. Grey: native bones and rigid-facing cut walls.',20)
        centered(d,1030,'Literal 1x coordinates | no shape target, fitting, geometric smoothing or displacement gain',21)
        if not reference_only and contact_mode=='coarse_physical_sites':
            approximation='Approximate complementarity' if contact_enforcement=='compact_cap_barrier_v1' else 'Approximate boundary'
            centered(d,1068,f'Site spacing {site_resolution_mm:g} mm | {approximation} | NOT SURFACE-COLLISION CERTIFIED',20,'#943c35')
        else:
            centered(d,1068,'Coarse homogenized P1 carrier; display lighting and class colors are not physical measurements.',20)
        centered(d,1106,f'Engineering scale assumption: 1 atlas L0 = {self.scale:g} mm; not measured calibration.',20)
        centered(d,1144,'Fixed orthographic cameras and equal pixel/unit scale. All finite boundary triangles submitted.',19)
        return img,{'seconds':time.perf_counter()-start,'views':stats}

    def metadata(self):
        return {'geometry_sha256':self.g['sha256'],'atlas_sha256':ATLAS_SHA,
            'scale_mm_per_L0':self.scale,'camera':list(map(asdict,self.cameras)),
            'boundary_triangles':len(self.g['triangles']),'native_bone_triangles':len(self.bone_faces),
            'coordinate_operations':'Actual saved q -> unchanged parent-tet P1 barycentric sum; orthographic display rotation only',
            'no_coordinate_fit':True,'no_geometric_smoothing':True,'no_displacement_gain':True,
            'transparency':False,'pixel_filtering':False,'decimation':False}
