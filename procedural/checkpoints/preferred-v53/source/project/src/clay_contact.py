"""Original triangle-BVH contact queries; no external renderer or mesh package."""
from __future__ import annotations
import ctypes,hashlib,subprocess,tempfile,shutil
from pathlib import Path
import numpy as np
_LIBRARY=None

def _library():
 global _LIBRARY
 if _LIBRARY is not None:return _LIBRARY
 src=Path(__file__).with_suffix('.cpp');digest=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
 folder=Path(tempfile.gettempdir())/'original_clay_contacts';folder.mkdir(exist_ok=True)
 output=folder/f'contact_{digest}.so'
 if not output.exists():
  compiler=shutil.which('g++') or shutil.which('clang++')
  if compiler is None:raise RuntimeError('Exact contact refinement requires the installed C++ compiler')
  with tempfile.NamedTemporaryFile(dir=folder,suffix='.so',delete=False) as temp:name=temp.name
  try:
   subprocess.run([compiler,'-O3','-std=c++17','-shared','-fPIC',str(src),'-o',name],check=True,capture_output=True,timeout=90)
   Path(name).replace(output)
  finally:Path(name).unlink(missing_ok=True)
 lib=ctypes.CDLL(str(output));dp=ctypes.POINTER(ctypes.c_double)
 lib.clay_contact_create.argtypes=[dp,ctypes.POINTER(ctypes.c_int64),ctypes.c_uint32];lib.clay_contact_create.restype=ctypes.c_void_p
 lib.clay_contact_destroy.argtypes=[ctypes.c_void_p];lib.clay_contact_destroy.restype=None
 lib.clay_contact_rays.argtypes=[ctypes.c_void_p,dp,dp,ctypes.c_uint32,ctypes.c_double,ctypes.c_double,ctypes.POINTER(ctypes.c_uint8)];lib.clay_contact_rays.restype=None
 _LIBRARY=lib;return lib

class ContactBVH:
 """Retains the original contiguous arrays; nodes/permutation add bounded storage."""
 def __init__(self,vertices,triangles):
  self.vertices=np.ascontiguousarray(vertices,dtype=np.float64);self.triangles=np.ascontiguousarray(triangles,dtype=np.int64)
  if not 0<len(self.triangles)<2**32:raise ValueError('Unsupported triangle count')
  self.lib=_library();self.ptr=self.lib.clay_contact_create(self.vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),self.triangles.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),len(self.triangles))
  if not self.ptr:raise RuntimeError('Unable to construct the exact contact BVH')
 def close(self):
  if getattr(self,'ptr',None):self.lib.clay_contact_destroy(self.ptr);self.ptr=None
 def __del__(self):self.close()
 def occluded(self,points,directions,*,normals=None,normal_offset=.008,tmin=.025,tmax=1e30):
  origins=np.ascontiguousarray(points,dtype=np.float64)
  if normals is not None:origins=np.ascontiguousarray(origins+np.asarray(normals)*normal_offset)
  dirs=np.ascontiguousarray(np.broadcast_to(directions,origins.shape),dtype=np.float64)
  if origins.ndim!=2 or origins.shape[1]!=3 or not np.isfinite(origins).all() or not np.isfinite(dirs).all():raise ValueError('Finite Nx3 rays required')
  out=np.empty(len(origins),np.uint8);dp=ctypes.POINTER(ctypes.c_double)
  self.lib.clay_contact_rays(self.ptr,origins.ctypes.data_as(dp),dirs.ctypes.data_as(dp),len(origins),tmin,tmax,out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
  return out.astype(bool)
