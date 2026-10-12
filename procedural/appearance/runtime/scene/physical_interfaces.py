"""Original boundary-interface optic, not a validated measured eye.
One surface per boundary, with explicit relative index and region orientation.
No overlapping glass solids, coincident surfaces or artificial air gap.
Material radiance transport follows Cycles' dielectric surface BSDF convention.
The optional shadow-ray passthrough remains an approximation, not physical transport.
"""
import numpy as np
import bpy
from eye_similarity import Frame

def glass(name,ratio,shadow_approx=False):
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;n.clear()
    b=n.new('ShaderNodeBsdfGlass');b.distribution='BECKMANN';b.inputs['Color'].default_value=(1,1,1,1);b.inputs['Roughness'].default_value=0.;b.inputs['IOR'].default_value=ratio
    o=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(b.outputs[0],o.inputs['Surface'])
    if shadow_approx:
        lp=n.new('ShaderNodeLightPath');tr=n.new('ShaderNodeBsdfTransparent');mix=n.new('ShaderNodeMixShader');m.node_tree.links.new(lp.outputs['Is Shadow Ray'],mix.inputs[0]);m.node_tree.links.new(b.outputs[0],mix.inputs[1]);m.node_tree.links.new(tr.outputs[0],mix.inputs[2]);m.node_tree.links.new(mix.outputs[0],o.inputs['Surface'])
    return m

def build(c,R,mesh,transform,oi,use_mnee=True,shadow_approx=False,*,frame=None):
    # Author every boundary in reference units; scale only completed point sets.
    # With an explicit frame its reference radius governs construction, whether
    # the caller supplies the reference or actual radius as the legacy R argument.
    if frame is None:frame=Frame(c,R,R)
    else:R=frame.reference_radius
    scale=frame.scale
    nc=1.376;na=1.3374;nt=192;rr=np.linspace(.035,5.56,40);r,t=np.meshgrid(rr,np.arange(nt)*2*np.pi/nt,indexing='ij')
    top=np.sqrt(R*R-r*r)+.035+.36*np.maximum(0,1-(r/5.56)**2)**2
    thickness=.25+.20*np.maximum(0,1-(r/5.56)**2)
    post=top-thickness;back=np.full_like(top,9.8)
    def points(z):return np.stack((c[0]+r*np.cos(t),c[1]+r*np.sin(t),c[2]+z),axis=-1).reshape(-1,3)
    mats=[glass('Air to cornea | relative IOR 1.376',nc,shadow_approx),glass('Cornea to aqueous | relative IOR 1.376/1.3374',nc/na,shadow_approx),glass('Aqueous to air | relative IOR 1.3374',na,shadow_approx)]
    arrays=[points(top),points(post),points(back)];front=[]
    for j in range(len(rr)-1):
        for k in range(nt):
            a=j*nt+k;b=j*nt+(k+1)%nt;d=(j+1)*nt+k;e=(j+1)*nt+(k+1)%nt;front.extend([(a,d,e),(a,e,b)])
    N=len(arrays[0]);faces=np.array(front,dtype=np.int32);diagnostic_v=[];diagnostic_f=[];offset=0
    for k,(verts,centerz,orientation,mat) in enumerate(zip(arrays,[R+.395,R+.395-.45,9.8],[1,-1,-1],mats)):
        vs=np.r_[verts,[[c[0],c[1],c[2]+centerz]]];fs=np.r_[faces,[(N,i,(i+1)%nt) for i in range(nt)]]
        if orientation<0:fs=fs[:,::-1]
        vs=frame.apply(vs)
        ob=mesh(f'eye{oi}_interface_{k}',transform(vs),fs,mat);ob.cycles.is_caustics_caster=use_mnee
        diagnostic_v.append(vs);diagnostic_f.append(fs+offset);offset+=len(vs)
    # Cylindrical rims join outer envelope. Internal boundary is represented once.
    for k,mat in [(0,mats[0]),(1,mats[2])]:
        va=arrays[k][-nt:];vb=arrays[k+1][-nt:];vs=np.r_[va,vb];fs=[]
        for i in range(nt):j=(i+1)%nt;fs.extend([(i,nt+i,nt+j),(i,nt+j,j)])
        vs=frame.apply(vs)
        fs=np.array(fs);ob=mesh(f'eye{oi}_region_rim_{k}',transform(vs),fs,mat);ob.cycles.is_caustics_caster=use_mnee
        diagnostic_v.append(vs);diagnostic_f.append(fs+offset);offset+=len(vs)
    return {'shadow_ray_passthrough':shadow_approx,'construction':'Single, noncoincident surface per medium boundary; explicit index ratios; no participating-volume absorption. Shared posterior interface is represented only once.','cornea_ior':nc,'aqueous_ior':na,'posterior_relative_ior':nc/na,'corneal_thickness_design_units':[.25*scale,.45*scale],'rear_plane_z_relative':9.8*scale,'posterior_min_z_relative':float(post.min())*scale,'roughness':0,'source_for_index_values':'https://opg.optica.org/josaa/abstract.cfm?uri=josaa-16-8-1881','limits':'Published scalar index examples only. Authored curvature, thickness and eye pose are not calibrated. No lens, retina, dispersion, accommodation or measured tear film.','similarity_frame':frame.report()},np.concatenate(diagnostic_v),np.concatenate(diagnostic_f),frame.apply(arrays[0])
