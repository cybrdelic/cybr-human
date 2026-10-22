"""Conservative radius-bucket candidate acceleration; exact AABB final filter.
No exclusions or contact physics here. Inputs must enclose full affine paths.
"""
from itertools import chain
import numpy as np
from scipy.spatial import cKDTree

def candidates(alo,ahi,blo,bhi,clearance=0.):
    alo=np.asarray(alo,float);ahi=np.asarray(ahi,float);blo=np.asarray(blo,float);bhi=np.asarray(bhi,float)
    if clearance<0 or not all(np.isfinite(x).all() for x in (alo,ahi,blo,bhi)) or np.any(ahi<alo) or np.any(bhi<blo):raise ValueError('invalid bounds')
    if not len(alo) or not len(blo):return np.empty((0,2),int)
    # Infinity norm uses AABB halfwidth max; intersecting AABBs necessarily
    # have centre infinity-distance <= both enclosing-cube radii summed.
    ac=alo+(ahi-alo)/2;bc=blo+(bhi-blo)/2
    ar=np.nextafter(np.max(np.maximum(ac-alo,ahi-ac),axis=1),np.inf)
    br=np.nextafter(np.max(np.maximum(bc-blo,bhi-bc),axis=1),np.inf)
    # Additional absolute-coordinate roundoff margin guards centre subtraction.
    eps=32*np.finfo(float).eps*max(1.,np.max(abs(ac)),np.max(abs(bc)))
    bins=np.floor(np.log2(np.maximum(br,np.finfo(float).tiny))).astype(int);out=[]
    for bucket in np.unique(bins):
        ids=np.flatnonzero(bins==bucket);tree=cKDTree(bc[ids]);r=np.nextafter(ar+br[ids].max()+clearance+eps,np.inf)
        hits=tree.query_ball_point(ac,r,p=np.inf,eps=0,workers=1)
        counts=np.fromiter((len(js) for js in hits),dtype=np.intp,count=len(hits))
        prefix=np.r_[0,np.cumsum(counts,dtype=np.intp)]
        start=0
        while start<len(hits):
            end=min(len(hits),max(start+1,int(np.searchsorted(prefix,prefix[start]+100_000,side='right')-1)))
            count=int(prefix[end]-prefix[start])
            if count:
                ai=np.repeat(np.arange(start,end),counts[start:end])
                js=np.fromiter(chain.from_iterable(hits[start:end]),dtype=np.intp,count=count)
                j=ids[js]
                ok=np.all((np.nextafter(ahi[ai]+clearance,np.inf)>=blo[j])&(np.nextafter(alo[ai]-clearance,-np.inf)<=bhi[j]),axis=1)
                if np.any(ok):out.append(np.c_[ai[ok],j[ok]])
            start=end
    if not out:return np.empty((0,2),int)
    p=np.concatenate(out);return p[np.lexsort((p[:,1],p[:,0]))]

def exhaustive(alo,ahi,blo,bhi,clearance=0.,chunk=128):
    out=[]
    for s in range(0,len(alo),chunk):
        k=np.all((np.nextafter(ahi[s:s+chunk,None]+clearance,np.inf)>=blo)&(np.nextafter(alo[s:s+chunk,None]-clearance,-np.inf)<=bhi),axis=2)
        p=np.argwhere(k);p[:,0]+=s;out.append(p)
    return np.concatenate(out) if out else np.empty((0,2),int)
