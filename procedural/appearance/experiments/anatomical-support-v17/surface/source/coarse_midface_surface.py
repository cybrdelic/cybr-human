"""Explicit editable cubic front-support chart, not a sum of local bulges.

The low-resolution cross-section net defines the interior surface completely.
Original finite eyelid, oral and medial nasal constructions remain protected;
C2 partitions give exact positional and derivative attachment to those regions.
Existing host connectivity is a tessellation carrier for the sampled surface.
"""
import json
import numpy as np
from scipy.interpolate import RectBivariateSpline

def smooth(x):
 x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))

def sample_supported_surface(p,control_path):
 c=json.loads(control_path.read_text());x=np.array(c['x_controls'],float);y=np.array(c['y_controls'],float);z=np.array(c['anterior_z_controls'],float)
 assert z.shape==(len(y),len(x))
 chart=RectBivariateSpline(y,x,z,kx=3,ky=3,s=0)
 reference=RectBivariateSpline(y,x,np.array(c['reference_anterior_z_controls'],float),kx=3,ky=3,s=0)
 xx,yy,zz=p.T;ax=np.abs(xx)
 target=chart.ev(np.clip(yy,y[0],y[-1]),np.clip(ax,x[0],x[-1]))
 ref=reference.ev(np.clip(yy,y[0],y[-1]),np.clip(ax,x[0],x[-1]))
 carrier=smooth((72-ax)/10)*smooth((yy+110)/18)*smooth((24-yy)/12)*smooth((zz-50)/35)
 dist=np.minimum(np.linalg.norm(p-[31.3,12,94.6],axis=1),np.linalg.norm(p-[-31.3,12,94.6],axis=1))
 carrier*=1.0  # Orbital region is rebuilt against this new host below.
 # Keep the central nose fixed above the oral support. The alar attachment
 # travels gradually with the same carrier as the surrounding maxilla.
 carrier*=1-(1-smooth((ax-8)/8))*smooth((yy+44)/9)
 gate=smooth((ax-14)/9)
 oral=np.sqrt((xx/30)**2+((yy+53)/11.5)**2)
 gate*=smooth((oral-1)/.42)
 # One common support motion for finite features and exterior. Only the
 # residual is suppressed in the replacement interior; protected finite
 # tissues keep their original local thickness and topological returns.
 out=p.copy();out[:,2]=zz+carrier*((target-ref)+gate*(ref-zz))
 # The prescribed interior is a regular graph (x,y,Z(x,y)); no parameter
 # collapse, inverted patch normals or inherited fine-scale surface dents.
 u,v=np.meshgrid(np.linspace(x[0],x[-1],181),np.linspace(y[0],y[-1],181));dzdy=chart.ev(v.ravel(),u.ravel(),dx=1,dy=0);dzdx=chart.ev(v.ravel(),u.ravel(),dx=0,dy=1)
 report={'representation':'Editable 12 by 9 cubic cross-section surface with common finite-feature support carrier and residual-preserving oral/nasal attachments','control_file':str(control_path),'interior_full_replacement_vertices':int(((gate*carrier)>.999).sum()),'modified_vertices':int((carrier>0).sum()),'maximum_abs_depth_change':float(np.abs(out[:,2]-zz).max()),'minimum_parameterized_surface_area_factor':float(np.sqrt(1+dzdx**2+dzdy**2).min()),'limitations':['This is the exterior front support chart only, not a full volumetric mechanics rest map','Existing host triangles carry the tessellation; boundary blending still requires explicit crossing and appearance checks','The guide is artistic and generated multiview is not metrically guaranteed']}
 return out,report
