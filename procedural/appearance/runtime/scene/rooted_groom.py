"""Original deterministic surface-rooted brows/lashes. Native tapered Cycles curves.
No image maps/cards/dots/painted shadows. Geometric semantics and fit are static,
not measured follicle anatomy or a blink-ready groom. Source coordinates are mm-like.
"""
import bpy,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def hair_material(name,melanin,roughness):
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;n.clear()
    h=n.new('ShaderNodeBsdfHairPrincipled');h.model='CHIANG';h.parametrization='MELANIN';h.inputs['Melanin'].default_value=melanin;h.inputs['Melanin Redness'].default_value=.18;h.inputs['Roughness'].default_value=roughness;h.inputs['Radial Roughness'].default_value=.42;h.inputs['IOR'].default_value=1.55;h.inputs['Random Color'].default_value=.12;h.inputs['Random Roughness'].default_value=.12
    out=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(h.outputs[0],out.inputs['Surface']);return m

def create_curves(name,paths,radii,transform,mat):
    c=bpy.data.hair_curves.new(name);c.add_curves([len(p) for p in paths]);c.attributes['position'].data.foreach_set('vector',transform(np.concatenate(paths)).ravel())
    rad=c.attributes.new(name='radius',type='FLOAT',domain='POINT');rad.data.foreach_set('value',np.concatenate(radii).astype(np.float32)*.001)
    # Native curves give the Cycles fiber shader genuine per-strand tangent and randomness.
    obj=bpy.data.objects.new(name,c);bpy.context.scene.collection.objects.link(obj);c.materials.append(mat);return obj

def build_groom(v,f,ids,semantics,transform,eye_centers,eye_radii,seed=2938,eye_scales=None):
    eye_centers=np.asarray(eye_centers,dtype=float);eye_radii=np.asarray(eye_radii,dtype=float)
    eye_scales=np.ones(2) if eye_scales is None else np.asarray(eye_scales,dtype=float)
    assert eye_scales.shape==(2,) and np.all(eye_scales>0)
    assert eye_centers.shape==(2,3) and eye_radii.shape==(2,) and np.all(eye_radii>5.56*eye_scales)
    rng=np.random.default_rng(seed)
    # Only front brow/periorbital skin is needed for attachment queries.
    sf=f[ids==0];p=v[sf];choose=(p[:,:,1].max(1)>-1)&(p[:,:,1].min(1)<48)&(p[:,:,0].max(1)>-62)&(p[:,:,0].min(1)<62)&(p[:,:,2].max(1)>70);sf=sf[choose];used,rev=np.unique(sf,return_inverse=True)
    bvh=BVHTree.FromPolygons(v[used].tolist(),rev.reshape(-1,3).tolist(),all_triangles=True)
    del p,sf,used,rev
    def surface(x,y):
        q,n,face,dist=bvh.ray_cast(Vector((float(x),float(y),190.)),Vector((0,0,-1)),160.)
        return None if q is None else (np.array(q),np.array(n))
    mat_brow=hair_material('Original brown brow fibers | Chiang approximation',.80,.43);mat_lash=hair_material('Original dark lash fibers | Chiang approximation',.90,.40)
    all_stats=[];count_points=0
    for side in [-1,1]:
        paths=[];radii=[];root_errors=[];projected_samples=0
        for i in range(940):
            u=(i+rng.uniform(.05,.95))/940
            t=1-(1-u)**.68
            x=side*(12.8+39.8*t);center=26.2+4.4*np.sin(np.pi*t)-5.2*t*t+.18*side
            width=.35+3.4*(1-t)**.45+1.0*np.sin(np.pi*t)
            across=np.clip(rng.normal(0,.48),-.98,.98);y=center+across*width
            root=surface(x,y)
            if root is None:continue
            direction=np.array([side*(.35+1.25*t),.72-1.18*t-.60*across+rng.normal(0,.09)]);direction/=np.linalg.norm(direction)
            length=(1.65+1.6*np.sin(np.pi*t))*(1-.12*abs(across))*rng.uniform(.75,1.24);jitter=rng.uniform(-.24,.24);radius_scale=rng.uniform(.76,1.15)
            path=[];rr=[]
            for u in np.linspace(0,1,9):
                xy=np.array([x,y])+direction*length*u+np.array([-direction[1],direction[0]])*jitter*np.sin(np.pi*u)
                hit=surface(*xy)
                if hit is None:break
                point,normal=hit;normal=normal/max(np.linalg.norm(normal),1e-8)
                lift=.005+.14*np.sin(np.pi*u)+.075*u
                path.append(point+normal*lift);rr.append(radius_scale*.032*(1-u)**.72+.0015)
                projected_samples+=1
            if len(path)==9:
                path=np.array(path);paths.append(path);radii.append(np.array(rr)*(0.35+0.65*(lambda q:q*q*(3-2*q))(np.clip(t/.18,0,1))));root_errors.append(float(np.linalg.norm(path[0]-root[0])))
        create_curves(f'Original rooted brow {side:+d}',paths,radii,transform,mat_brow);count_points+=sum(map(len,paths))
        all_stats.append({'region':f'brow_{side:+d}','strands':len(paths),'points_per_strand':9,'root_method':'Authored brow-band distribution; every root and path sample projected to actual unchanged skin. This is an authored groom region, not an existing semantic brow band.','maximum_root_offset_input_units':max(root_errors),'root_radius_input_units_range':[min(float(r[0]) for r in radii),max(float(r[0]) for r in radii)],'medial_radius_feather':'0.35+0.65*smoothstep(t/0.18); roots, paths and body/tail unchanged', 'footprint':'Broader medial body tapering to lateral tail; stratified medial-weighted roots and cross-band convergence of growth','tip_radius_input_units':.0015,'seed':seed})
    for label in ['left','right']:
        pts=semantics[label+'_lash_support_points'];norm=semantics[label+'_lash_support_normals'];upper=semantics[label+'_lash_support_upper_half'];vid=semantics[label+'_lash_support_vertex_ids']
        assert np.max(np.abs(v[vid]-pts))<1e-8
        center=pts.mean(0);sign=1 if center[0]>0 else -1
        binding=int(np.argmin(np.linalg.norm(eye_centers-center,axis=1)))
        globe_center=eye_centers[binding];globe_radius=float(eye_radii[binding]);eye_scale=float(eye_scales[binding])
        assert np.all(np.linalg.norm(pts-globe_center,axis=1)<17.5*eye_scale), 'Semantic lash support exceeds intended uniform-similarity region'
        assert np.all((pts[:,0]>-62)&(pts[:,0]<62)&(pts[:,1]>-1)&(pts[:,1]<48)&(pts[:,2]>70)&(pts[:,2]<190)), 'Lash support exceeds groom query coverage'
        for is_upper,count in [(True,83),(False,37)]:
            inds=np.flatnonzero(upper==is_upper);order=np.argsort(sign*pts[inds,0]);inds=inds[order]
            eligible=inds[5:-5]
            side_x=sign*pts[eligible,0];fraction=(side_x-side_x.min())/(side_x.max()-side_x.min())
            # Uneven but deterministic density, sparse at the corners. Exact semantic roots.
            weights=.20+np.sin(np.pi*fraction)**.7
            chosen=rng.choice(eligible,size=count,replace=False,p=weights/weights.sum())
            chosen=chosen[np.argsort(sign*pts[chosen,0])]
            paths=[];radii=[];root_offsets=[];skin_sample_gaps=[];lengths=[];collision_corrections=[]
            for j,idx in enumerate(chosen):
                root=pts[idx];n=norm[idx];n=n/max(np.linalg.norm(n),1e-8)
                frac=float((sign*root[0]-side_x.min())/(side_x.max()-side_x.min()))
                shape=np.sin(np.pi*np.clip(frac,0,1))**.72
                length=((2.0+3.4*shape) if is_upper else (1.05+1.5*shape))*rng.uniform(.77,1.22)*eye_scale
                lateral=sign*((frac-.42)*.72+rng.normal(0,.065))
                curl=rng.uniform(.78,1.23);root_radius=(.032 if is_upper else .020)*rng.uniform(.82,1.14)*eye_scale
                path=[];rr=[]
                for u in np.linspace(0,1,14):
                    # Forward emergence separates the strand from skin; initial upper dip then
                    # upward curl prevents the previous straight tracks along the lid surface.
                    vertical=(-.20*np.sin(np.pi*u)+.42*curl*u*u) if is_upper else (-.12*u-.36*curl*u*u)
                    direction=np.array([lateral*u+.07*sign*np.sin(np.pi*u)*np.sin(j*2.39),vertical,1.0*u-.22*u*u])
                    point=root+n*(.006*eye_scale)+length*(direction+n*.55*u*np.exp(-1.3*u))
                    radius_here=root_radius*(1-u)**.86+.0010*eye_scale
                    original_point=point.copy()
                    if u>0:
                        # Static guide relaxation against the actual unchanged skin. Roots
                        # remain exact; only emergence/curve samples receive collision lift.
                        for _ in range(4):
                            q,normal,face,distance=bvh.find_nearest(Vector(point));normal=np.array(normal)
                            signed=float(np.dot(point-np.array(q),normal))
                            if signed>=radius_here+.018*eye_scale:break
                            point+=normal*(radius_here+.018*eye_scale-signed)
                    collision_corrections.append(float(np.linalg.norm(point-original_point)))
                    path.append(point);rr.append(radius_here)
                    if u>=.15:
                        q,normal,face,distance=bvh.find_nearest(Vector(point))
                        signed=float(np.dot(point-np.array(q),np.array(normal)))
                        skin_sample_gaps.append(signed-rr[-1])
                paths.append(np.array(path));radii.append(np.array(rr));root_offsets.append(float(np.linalg.norm(np.array(path)[0]-root)));lengths.append(length)
            points=np.concatenate(paths);radius=np.concatenate(radii)
            relative=points-globe_center
            clearance=float(np.min(np.linalg.norm(relative,axis=1)-globe_radius-radius))
            rho=np.linalg.norm(relative[:,:2],axis=1);in_cap=rho<5.56*eye_scale
            front=np.sqrt(globe_radius**2-rho[in_cap]**2)+.035*eye_scale+.36*eye_scale*(1-rho[in_cap]**2/(5.56*eye_scale)**2)**2
            optical_gap=float(np.min(relative[in_cap,2]-front-radius[in_cap])) if np.any(in_cap) else None
            assert clearance>0, ('Lash tube intersects original globe at a control point',label,is_upper,clearance)
            assert optical_gap is None or optical_gap>0, ('Lash tube crosses optical cap at a control point',label,is_upper,optical_gap)
            assert min(skin_sample_gaps)>0, ('Lash tube is inside skin away from its root',label,is_upper,min(skin_sample_gaps))
            create_curves(f'Original {label} {"upper" if is_upper else "lower"} rooted lashes',paths,radii,transform,mat_lash);count_points+=sum(map(len,paths))
            all_stats.append({'region':f'{label}_{"upper" if is_upper else "lower"}_lashes','strands':len(paths),'points_per_strand':14,'root_method':'Exact geometry-owner lash_support vertex IDs. Not inner wetline.','curve':'Forward emergence with variable curl and fan; shorter sparse corner fibers','eye_similarity_scale':eye_scale,'actual_globe_binding':{'center_input_units':globe_center.tolist(),'radius_input_units':globe_radius,'binding_method':'Nearest verified source-globe center to exact semantic lid-loop centroid'},'sampled_control_point_tube_globe_clearance_input_units':clearance,'sampled_control_point_tube_cornea_clearance_input_units':optical_gap,'sampled_signed_skin_clearance_away_from_root_input_units':min(skin_sample_gaps),'maximum_control_point_collision_relaxation_input_units':max(collision_corrections),'clearance_scope':'Tube radius included at control points; nearest-surface orientation test beyond u=0.15; not a continuous curve collision proof','length_parameter_range_input_units':[min(lengths),max(lengths)],'source_vertex_ids':vid[chosen].tolist(),'support_loop_indices':chosen.tolist(),'maximum_root_offset_input_units':max(root_offsets),'seed':seed})
    return {'method':'Original deterministic native tapered curves; actual Cycles Chiang fiber shader, no images or painted shadows','seed':seed,'regions':all_stats,'strand_count':sum(s['strands'] for s in all_stats),'control_point_count':count_points,'shader':{'model':'CHIANG','brow_melanin':.80,'lash_melanin':.90,'melanin_redness':.18,'ior':1.55,'roughness_brow':.43,'roughness_lash':.40,'artistic_uncalibrated':True},'brow_binding':'Original authored XY footprint reprojected onto selected skin; deliberately not globally enlarged with the optical assembly','limits':['Static attachment only; no blink/expression validation','Lash paths use artist-authored lengths and curl, not measured follicle anatomy','Brow distribution is authored then projected; only lash roots have exact tissue semantics','No calibrated spectral absorption or population-specific claim']}
