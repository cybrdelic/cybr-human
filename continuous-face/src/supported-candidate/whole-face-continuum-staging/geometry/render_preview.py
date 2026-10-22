from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
G=Path(__file__).parent;a=np.load(G/'whole-face-boundary-preview.npz');v=a['vertices'];f=a['triangles'];cl=a['boundary_class'];tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=cross/np.linalg.norm(cross,axis=1)[:,None]
W,H=1800,1000;im=Image.new('RGB',(W,H),(248,247,243));dr=ImageDraw.Draw(im)
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
except:font=None;small=None
for k,(yaw,title) in enumerate([(0,'Front: whole facial/head tissue'),(65,'Oblique: source-derived volume')]):
 r=np.deg2rad(yaw);R=np.array([[np.cos(r),0,-np.sin(r)],[0,1,0],[np.sin(r),0,np.cos(r)]]);vv=v@R.T;tt=vv[f];nn=norm@R.T;visible=nn[:,2]>.01;ids=np.flatnonzero(visible);ids=ids[np.argsort(tt[ids,:,2].mean(1))];scale=3.5;center=np.array([450+900*k,520]);xy=np.stack([vv[:,0],-vv[:,1]],axis=1)*scale+center
 light=np.array([-.4,.55,.75]);brightness=.55+.45*np.maximum(0,nn@light);base=np.tile([197.,148.,118.],(len(f),1));base[(cl>=1)&(cl<=5)]=[137,102,91];base[cl==6]=[157,141,122];base[cl==7]=[165,145,122];colors=np.clip(base*brightness[:,None],0,255).astype(np.uint8)
 for idx in ids:dr.polygon([tuple(x) for x in xy[f[idx]]],fill=tuple(colors[idx]))
 dr.text((70+900*k,55),title,fill=(35,40,45),font=font)
dr.text((60,890),'Actual source-only continuum geometry. No solve, fitted skin target, or fabricated globes/ears.',fill=(45,48,52),font=font)
dr.text((60,930),'1-L0 integration boundary shown without smoothing. Coarse envelope, fat and 3-L0 matrix backing are explicit assumptions.',fill=(65,67,70),font=small)
im.save(G/'whole-face-boundary-preview.png')
print(G/'whole-face-boundary-preview.png')
