"""Original anatomical-region skin controls. No images, scans or painted shadows.
Authored RGB absorption/color proxies, not a measured biophysical skin spectrum.
Coordinates are the project's uncalibrated millimetre-like design units.
"""
import numpy as np

def value_noise(v,frequency,seed):
    p=np.asarray(v,dtype=np.float32)*frequency;i=np.floor(p).astype(np.int32);t=p-i;t=t*t*(3-2*t);out=np.zeros(len(v),np.float32)
    for a in (0,1):
      for b in (0,1):
       for c in (0,1):
        h=(i[:,0].astype(np.int64)+a)*73856093^(i[:,1].astype(np.int64)+b)*19349663^(i[:,2].astype(np.int64)+c)*83492791^seed
        h=(h^(h>>13))*1274126177;h=(h^(h>>16))&0xffffff
        q=(h.astype(np.float32)/8388607.5-1)
        w=(t[:,0] if a else 1-t[:,0])*(t[:,1] if b else 1-t[:,1])*(t[:,2] if c else 1-t[:,2]);out+=q*w
    return out

def fields(v):
    x,y,z=v.T;ax=np.abs(x);front=np.clip((z-50)/40,0,1)
    def region(cx,cy,sx,sy):return np.exp(-.5*((ax-cx)/sx)**2-.5*((y-cy)/sy)**2)*front
    cheek=region(37,-14,18,19);nose=region(0,-22,12,17);lid=region(31,11,13,6.5)
    forehead=region(0,62,43,26);ear=np.exp(-.5*((ax-77)/9)**2-.5*((y+2)/29)**2)
    # Independent slow pigmentation and smaller vascular mottling, not shading.
    melanin=.035*value_noise(v,.044,281)+.014*value_noise(v,.17,1307)
    blood_mottle=.035*value_noise(v,.075,937)+.012*value_noise(v,.24,157)
    colors=np.tile(np.array([.43,.245,.158],np.float32),(len(v),1))
    colors*=1-melanin[:,None]*np.array([.7,1.,1.3],np.float32)
    blood=np.clip(.22*cheek+.20*nose+.17*lid+.30*ear+blood_mottle,.0,.48)
    colors=colors*(1-blood[:,None])+np.array([.475,.178,.137],np.float32)*blood[:,None]
    # Mild forehead hue difference, without darkening the socket or creases.
    warm=.10*forehead;colors=colors*(1-warm[:,None])+np.array([.44,.261,.166],np.float32)*warm[:,None]
    rough=.60-.14*forehead-.17*nose-.12*lid+.045*cheek+.025*ear
    rough=np.clip(rough,.40,.67).astype(np.float32)
    return np.column_stack((colors,np.ones(len(v),np.float32))).astype(np.float32),rough,{'method':'Smooth authored forehead/cheek/nose/eyelid/ear fields; exact lip semantics; independent low-amplitude coherent melanin/vascular noise','no_painted_shadow_or_socket_darkening':True,'roughness_range_before_microvariation':[float(rough.min()),float(rough.max())],'roughness_microvariation_amplitude':.02,'parameters_are_artistic_not_measured':True}
