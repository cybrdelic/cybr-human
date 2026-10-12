#!/usr/bin/env python3
"""Rebuild original CYBR HUMAN appearance; execute inside Blender, under run_preview.sh."""
import bpy, numpy as np, math, sys, argparse, json, time, hashlib, gc, resource
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parent
PROFILE_GEOMETRY_UNCHANGED=True
p=argparse.ArgumentParser();p.add_argument('--reference-control',action='store_true');p.add_argument('--eye-camera',choices=['follow_globe','frozen_v8'],default='follow_globe');p.add_argument('--lighting',choices=['studio','neutral_side'],default='studio');p.add_argument('--sss-weight',type=float,default=.35);p.add_argument('--surface-profile',choices=['baseline','pores'],default='baseline');p.add_argument('--optics',choices=['path_only','mnee','shadow_passthrough','no_cornea','mnee_open_cap','mnee_filled_chamber','interfaces_mnee','interfaces_path','interfaces_shadow'],default='path_only');p.add_argument('--groom',type=int,choices=[0,1],default=1);p.add_argument('--raw-only',action='store_true');p.add_argument('--validate-only',action='store_true');p.add_argument('--view',choices=['head','head_front','eye','cheek','forehead'],default='head');p.add_argument('--mode',choices=['appearance','clay'],default='appearance');p.add_argument('--size',type=int,default=560);p.add_argument('--samples',type=int,default=40)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
assert 0 <= a.sss_weight <= 1
LENS_BACK=9.8 if a.optics=='mnee_filled_chamber' else 10.76
START=time.time()
REFERENCE_SOURCE=ROOT/'input-head.npz'
REFERENCE_CAPTURE=ROOT/'reference_eye_frame.json'
reference_capture=json.loads(REFERENCE_CAPTURE.read_text())
REFERENCE_SHA='e568bf8763102e8d9479fc75a295e4d6cdf709c1046d2fd33d0e716e2474ddf6'
SOURCE=REFERENCE_SOURCE if a.reference_control else ROOT/'input-head.npz'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==(REFERENCE_SHA if a.reference_control else '545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e')
sys.path.insert(0,str(ROOT));from eye_similarity import Frame
SEMANTICS=ROOT/'input-semantics.npz'
assert hashlib.sha256(SEMANTICS.read_bytes()).hexdigest()=='d2587be48722543d557a329870f78b29125d1bfa0f0d0aa54d0fc8adc2dce811'
assert not a.reference_control, 'Dedicated candidate runtime; use retained original reference output'
semantics=np.load(SEMANTICS,allow_pickle=False)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=a.samples;scene.cycles.use_denoising=False;scene.cycles.seed=47
scene.cycles.caustics_refractive=True;scene.cycles.max_bounces=8;scene.cycles.diffuse_bounces=3;scene.cycles.glossy_bounces=4;scene.cycles.transmission_bounces=6;scene.cycles.transparent_max_bounces=8
scene.render.threads_mode='FIXED';scene.render.threads=2;scene.render.resolution_x=a.size;scene.render.resolution_y=a.size;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='8';scene.render.film_transparent=False
scene.render.dither_intensity=0
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=-2.5;scene.view_settings.gamma=1
scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.20,.225,.25,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.30

def transform(v):
    v=np.asarray(v);return np.ascontiguousarray(np.column_stack((v[:,0],-v[:,2],v[:,1]))*.001,dtype=np.float32)
def mesh(name,v,f,material=None,colors=None):
    v=np.ascontiguousarray(v,dtype=np.float32);f=np.ascontiguousarray(f,dtype=np.int32)
    m=bpy.data.meshes.new(name);m.vertices.add(len(v));m.vertices.foreach_set('co',v.ravel());m.loops.add(f.size);m.loops.foreach_set('vertex_index',f.ravel());m.polygons.add(len(f));m.polygons.foreach_set('loop_start',np.arange(0,f.size,3,dtype=np.int32));m.polygons.foreach_set('loop_total',np.full(len(f),3,dtype=np.int32));m.polygons.foreach_set('use_smooth',np.ones(len(f),dtype=bool));m.update()
    obj=bpy.data.objects.new(name,m);scene.collection.objects.link(obj)
    if material:m.materials.append(material)
    if colors is not None:
        att=m.color_attributes.new(name='OriginalColor',type='FLOAT_COLOR',domain='POINT');att.data.foreach_set('color',np.ascontiguousarray(colors,dtype=np.float32).ravel())
    return obj

def basic(name,color,rough=.45):
    m=bpy.data.materials.new(name);m.use_nodes=True;bs=m.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(*color,1);bs.inputs['Roughness'].default_value=rough;return m,bs

def node(m,typ):return m.node_tree.nodes.new(typ)
def link(m,x,y):m.node_tree.links.new(x,y)
def mathnode(m,op,x=None,y=None):
    n=node(m,'ShaderNodeMath');n.operation=op
    for i,z in enumerate([x,y]):
        if z is None:continue
        if isinstance(z,(int,float)):n.inputs[i].default_value=z
        else:link(m,z,n.inputs[i])
    return n.outputs[0]
def noise(m,scale,detail=2):
    pos=node(m,'ShaderNodeNewGeometry');n=node(m,'ShaderNodeTexNoise');n.inputs['Scale'].default_value=scale;n.inputs['Detail'].default_value=detail;n.inputs['Roughness'].default_value=.55;link(m,pos.outputs['Position'],n.inputs['Vector']);return n

def skin_colors(v):
    x,y,z=v.T;front=np.clip((z-55)/45,0,1);ax=np.abs(x)
    colors=np.tile(np.array([.43,.245,.158]),(len(v),1))
    def g(cx,cy,sx,sy):return np.exp(-.5*((ax-cx)/sx)**2-.5*((y-cy)/sy)**2)*front
    blush=g(36,-13,20,19)*.15+g(0,-23,15,18)*.13
    colors=colors*(1-blush[:,None])+np.array([.44,.16,.125])*blush[:,None]
    under=g(30,1,18,5)*.09;colors=colors*(1-under[:,None])+np.array([.28,.155,.14])*under[:,None]
    # Smooth vermilion field uses the authored mouth chart. It never changes vertices.
    q=np.clip(ax/27,0,1);line=-54.6+.9*q*q-.45*np.exp(-((ax-8)/6)**2);dy=y-line
    up=np.interp(ax,[0,7,15,22,27],[4.5,8.6,5.5,2.5,.12]);lo=7.2*np.maximum(0,1-q*q)**.65+.08
    dist=np.abs(dy)/np.maximum(np.where(dy>=0,up,lo),.1)
    lip=np.clip((1.08-dist)/.35,0,1)*np.clip((27-ax)/2,0,1)*front*.63
    # Approximate lip chart intentionally not used in v5; exact semantics applied after skin field evaluation.
    ear=np.exp(-.5*((ax-76)/10)**2-.5*((y+3)/28)**2)*.18
    colors=colors*(1-ear[:,None])+np.array([.44,.185,.14])*ear[:,None]
    # Extremely low-amplitude deterministic pigmentation variation, no image maps.
    mott=.012*(np.sin(x*.22+np.sin(y*.12))*np.sin(y*.19+z*.12));colors*=1+mott[:,None]
    return np.column_stack((colors,np.ones(len(v))))

clay,cb=basic('Diagnostic clay | neutral, no texture or SSS',(.35,.365,.37),.63)
skin,sb=basic('Original procedural skin | uncalibrated artistic v1',(.43,.245,.158),.43)
sb.subsurface_method='RANDOM_WALK_SKIN';sb.inputs['Subsurface Weight'].default_value=a.sss_weight;sb.inputs['Subsurface Radius'].default_value=(1.,.48,.24);sb.inputs['Subsurface Scale'].default_value=.0011;sb.inputs['Subsurface IOR'].default_value=1.4;sb.inputs['Subsurface Anisotropy'].default_value=.7;sb.inputs['IOR'].default_value=1.43
vc=node(skin,'ShaderNodeVertexColor');vc.layer_name='OriginalColor';link(skin,vc.outputs['Color'],sb.inputs['Base Color'])
n=noise(skin,150,2);r=mathnode(skin,'MULTIPLY_ADD',n.outputs['Fac'],.04);r.node.inputs[2].default_value=-.02;regional=node(skin,'ShaderNodeAttribute');regional.attribute_name='RegionalRoughness';r=mathnode(skin,'ADD',regional.outputs['Fac'],r);link(skin,r,sb.inputs['Roughness'])
pore=noise(skin,3800,2);bump=node(skin,'ShaderNodeBump');bump.inputs['Strength'].default_value=.24;bump.inputs['Distance'].default_value=.000018;link(skin,pore.outputs['Fac'],bump.inputs['Height']);link(skin,bump.outputs['Normal'],sb.inputs['Normal'])

surface_report={'profile':'baseline: previous Noise bump'}
if a.surface_profile=='pores':
    sys.path.insert(0,str(ROOT));from surface_response import build as surface_build
    surface_report=surface_build(skin,sb,node,link,mathnode)

with np.load(SOURCE) as n:v=n['vertices'];f=n['triangles'];ids=n['object_ids']
source_meta={'path':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'vertices':len(v),'triangles':len(f),'skin_vertices_unchanged':PROFILE_GEOMETRY_UNCHANGED,'skin_topology_unchanged':PROFILE_GEOMETRY_UNCHANGED,'transformation':'(x,-z,y)*0.001; authored units without physical calibration'}
centers=[];radii=[];eye_frames=[];binding_reports=[]
for oi in [1,2]:
    q=v[np.unique(f[ids==oi])];c=(q.min(0)+q.max(0))/2;rad=np.linalg.norm(q-c,axis=1);centers.append(c);radii.append(float(np.median(rad)));assert np.ptp(rad)<.005
    ref=reference_capture['eyes'][oi-1];ref_c=np.array(ref['center_input_units']);ref_R=ref['radius_input_units']
    assert np.allclose(c,ref_c,rtol=0,atol=1e-10), ('This diagnostic expects unchanged eye centers',oi,c,ref_c)
    frame=Frame(c,ref_R,radii[-1]);eye_frames.append(frame)
    assert abs(frame.scale-(1 if a.reference_control else 1.07))<1e-12
    binding_reports.append({'object_id':oi,'frame':frame.report(),'source_sphere_max_radial_residual':float(np.max(np.abs(rad-radii[-1]))),'reference_source_sha256':REFERENCE_SHA,'reference_capture_sha256':hashlib.sha256(REFERENCE_CAPTURE.read_bytes()).hexdigest()})
# Diagnostic removes only anterior globe faces touching the transparent aperture.
removed_caps=0
if a.optics in ('mnee_open_cap','mnee_filled_chamber','interfaces_mnee','interfaces_path'):
    keep=np.ones(len(f),dtype=bool)
    for oi,c in enumerate(centers,1):
        ii=np.flatnonzero(ids==oi);q=(v[f[ii]]-c)/eye_frames[oi-1].scale
        remove=(np.linalg.norm(q[:,:,:2],axis=2).min(1)<5.54)&(q[:,:,2].min(1)>5.)
        keep[ii[remove]]=False
    removed_caps=int((~keep).sum());f=f[keep];ids=ids[keep]
# Frozen source skin is unchanged; optical globe cap handling is recorded separately.
sys.path.insert(0,str(ROOT));from coordinated_tissue import fields,attach_fields,apply_material
colors,regional_roughness,skin_field_report=fields(v)
vv=semantics['vermilion_vertex_ids'];tt=semantics['vermilion_t'];strength=.58*np.clip(tt/.18,0,1)
colors[vv,:3]=colors[vv,:3]*(1-strength[:,None,None])+np.array([.405,.150,.132])*strength[:,None,None]
inner=semantics['inner_labial_return_vertex_ids'];colors[inner,:3]=np.array([.30,.092,.082])
regional_roughness[vv]=regional_roughness[vv]*(1-strength[:,None])+0.34*strength[:,None];regional_roughness[inner]=.30
head=mesh('preferred-v53-integrated-tissue-eye-groom-candidate-v1 | original appearance functions',transform(v),f,clay if a.mode=='clay' else skin,colors if a.mode=='appearance' else None)
if a.mode=='appearance':
    rough_attribute=head.data.attributes.new(name='RegionalRoughness',type='FLOAT',domain='POINT');rough_attribute.data.foreach_set('value',regional_roughness)
    linked_field_report=attach_fields(head,v,semantics)
    linked_shader_report=apply_material(skin,sb,node,link)
del colors,regional_roughness

optics=[]
fit_checks=[];interface_reports=[]
def eye_material(c,R,frame):
    m,s=basic('Original sclera with explicit clear anterior optical aperture',(.69,.665,.59),.24);s.inputs['IOR'].default_value=1.38;s.inputs['Subsurface Weight'].default_value=.35;s.inputs['Subsurface Radius'].default_value=(1,.65,.4);s.inputs['Subsurface Scale'].default_value=.00025*frame.scale
    pos=node(m,'ShaderNodeNewGeometry');sep=node(m,'ShaderNodeSeparateXYZ');link(m,pos.outputs['Position'],sep.inputs[0]);cc=transform(np.array([c]))[0]
    dx=mathnode(m,'SUBTRACT',sep.outputs['X'],float(cc[0]));dz=mathnode(m,'SUBTRACT',sep.outputs['Z'],float(cc[2]))
    if frame.scale!=1.:
        dx=mathnode(m,'DIVIDE',dx,frame.scale);dz=mathnode(m,'DIVIDE',dz,frame.scale)
    r2=mathnode(m,'ADD',mathnode(m,'MULTIPLY',dx,dx),mathnode(m,'MULTIPLY',dz,dz));inside=mathnode(m,'LESS_THAN',r2,.00554**2);anterior=mathnode(m,'LESS_THAN',sep.outputs['Y'],float(cc[1]-.005*frame.scale));gate=mathnode(m,'MULTIPLY',inside,anterior)
    sclera_reports.append(sclera_build(m,s,dx,dz,node,link,mathnode,2711+int(c[0]>0)*137))
    tr=node(m,'ShaderNodeBsdfTransparent');mix=node(m,'ShaderNodeMixShader');link(m,gate,mix.inputs[0]);link(m,s.outputs[0],mix.inputs[1]);link(m,tr.outputs[0],mix.inputs[2]);link(m,mix.outputs[0],m.node_tree.nodes['Material Output'].inputs['Surface']);return m

def ring_surface(c,rr,nt,zfn,colorfn=None,name='ring',mat=None,cap=False,frame=None):
    t=np.arange(nt)*2*np.pi/nt;r,t=np.meshgrid(np.array(rr),t,indexing='ij');z=zfn(r,t);points=np.stack([c[0]+r*np.cos(t),c[1]+r*np.sin(t),c[2]+z],axis=-1).reshape(-1,3);fi=[]
    for j in range(len(rr)-1):
        for k in range(nt):
            aa=j*nt+k;bb=j*nt+(k+1)%nt;dd=(j+1)*nt+k;ee=(j+1)*nt+(k+1)%nt;fi.extend([(aa,dd,ee),(aa,ee,bb)])
    # Order above is +anterior for outer-facing surfaces in the source XY plane.
    col=colorfn(r,t).reshape(-1,4) if colorfn else None
    if frame is not None:points=frame.apply(points)
    return mesh(name,transform(points),np.array(fi),mat,col),points,np.array(fi)

cornea,cs=basic('Ideal clear cornea-aqueous optical approximation',(.98,.99,1),.025);cs.inputs['Transmission Weight'].default_value=1.;cs.inputs['IOR'].default_value=1.38
iris,ib=basic('Original radial iris fibers | no bitmap',(.13,.09,.04),.37);ib.inputs['IOR'].default_value=1.37
iv=node(iris,'ShaderNodeVertexColor');iv.layer_name='OriginalColor';link(iris,iv.outputs['Color'],ib.inputs['Base Color'])
pupil,pb=basic('Recessed opaque pupil cavity',(.0012,.001,.0008),.7)

import iris_stroma
from sclera_fields import build as sclera_build
iris_audits=[];sclera_reports=[]

if a.mode=='appearance':
    mi=np.zeros(len(f),dtype=np.int32)
    for oi,(c,R) in enumerate(zip(centers,radii),1):
        frame=eye_frames[oi-1];Rref=reference_capture['eyes'][oi-1]['radius_input_units'];s_eye=frame.scale
        m=eye_material(c,R,frame);head.data.materials.append(m);mi[ids==oi]=oi
        iris_stroma.configure(2127+oi*101)
        ir,ivv,iff=ring_surface(c,np.linspace(2.08,5.5,80),512,iris_stroma.height,iris_stroma.color,f'eye{oi}_iris_original_stromal_geometry',iris,frame=frame)
        ref_audit=iris_stroma.audit(frame.inverse(ivv),c,Rref)
        actual_audit=dict(ref_audit)
        for key in ('pupillary_margin_mean_z_relative','outer_root_mean_z_relative','minimum_sampled_posterior_cornea_gap_design_units','minimum_sampled_rear_plane_gap_design_units'):actual_audit[key]*=s_eye
        actual_audit['iris_depth_range_relative']=[x*s_eye for x in ref_audit['iris_depth_range_relative']]
        actual_audit['reference_construction_audit']=ref_audit;actual_audit['similarity_scale']=s_eye
        # Independent final-coordinate chamber checks, using the same authored boundaries.
        q=ivv-c;rho=np.linalg.norm(q[:,:2],axis=1);posterior=np.sqrt(R*R-rho*rho)+.035*s_eye+.36*s_eye*np.maximum(0,1-(rho/(5.56*s_eye))**2)**2-(.25+.20*np.maximum(0,1-(rho/(5.56*s_eye))**2))*s_eye
        actual_gaps={'iris_to_posterior_cornea':float(np.min(posterior-q[:,2])),'iris_to_rear_plane':float(np.min(q[:,2]-9.8*s_eye)),'pupil_to_rear_plane':(10.14-9.8)*s_eye,'pupil_recess_behind_iris_minimum':float(np.min(q[:,2])-10.14*s_eye)}
        assert min(actual_gaps.values())>.01*s_eye,actual_gaps
        actual_audit['independent_actual_chamber_gaps_design_units']=actual_gaps;iris_audits.append(actual_audit)
        pu,_,_=ring_surface(c,np.linspace(.001,2.15,14),192,lambda r,t:np.full_like(r,10.14),None,f'eye{oi}_pupil_recess',pupil,frame=frame)
        ir.cycles.is_caustics_receiver=(a.optics in ('mnee','mnee_open_cap','mnee_filled_chamber','interfaces_mnee'));pu.cycles.is_caustics_receiver=(a.optics in ('mnee','mnee_open_cap','mnee_filled_chamber','interfaces_mnee'))
        # Closed optical lens: outer cap + rear disk + edge stitch. Single medium; artistic.
        nt=160;rr=np.linspace(.035,5.56,35);t=np.arange(nt)*2*np.pi/nt;r,t=np.meshgrid(rr,t,indexing='ij')
        z=np.sqrt(Rref*Rref-r*r)+.035+.36*np.maximum(0,1-(r/5.56)**2)**2
        out=np.stack([c[0]+r*np.cos(t),c[1]+r*np.sin(t),c[2]+z],axis=-1).reshape(-1,3)
        back=out.copy();back[:,2]=c[2]+LENS_BACK;points=np.r_[out,back];faces=[];N=len(out)
        for j in range(len(rr)-1):
            for k in range(nt):
                aa=j*nt+k;bb=j*nt+(k+1)%nt;dd=(j+1)*nt+k;ee=(j+1)*nt+(k+1)%nt
                faces.extend([(aa,dd,ee),(aa,ee,bb),(N+aa,N+ee,N+dd),(N+aa,N+bb,N+ee)])
        for k in range(nt):
            aa=(len(rr)-1)*nt+k;bb=(len(rr)-1)*nt+(k+1)%nt;faces.extend([(aa,N+aa,N+bb),(aa,N+bb,bb)])
        # Explicit front and rear center fans close the two former numerical-radius center rings.
        points=np.r_[points,[[c[0],c[1],c[2]+Rref+.395],[c[0],c[1],c[2]+LENS_BACK]]];ct=len(points)-2;cb=len(points)-1
        for k in range(nt):faces.extend([(ct,k,(k+1)%nt),(cb,N+(k+1)%nt,N+k)])
        points=frame.apply(points);out=frame.apply(out)
        edge=np.sort(np.array(faces)[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1);_,counts=np.unique(edge,axis=0,return_counts=True);assert np.all(counts==2)
        lens_volume=float(np.einsum('ij,ij->i',points[np.array(faces)[:,0]],np.cross(points[np.array(faces)[:,1]],points[np.array(faces)[:,2]])).sum()/6);assert lens_volume>0
        co=mesh(f'eye{oi}_closed_original_cornea_lens',transform(points),np.array(faces),cornea)
        co.cycles.is_caustics_caster=(a.optics in ('mnee','mnee_open_cap','mnee_filled_chamber','interfaces_mnee'))
        co.hide_render=(a.optics=='no_cornea' or a.optics.startswith('interfaces_'))
        if a.optics.startswith('interfaces_'):
            sys.path.insert(0,str(ROOT));from physical_interfaces import build
            interface_report,points,faces,out=build(c,Rref,mesh,transform,oi,a.optics=='interfaces_mnee',a.optics=='interfaces_shadow',frame=frame)
            interface_reports.append(interface_report)
        # Broad-phase surface overlap is conservative: zero pairs rules out surface crossings.
        sf=f[ids==0];sv=v[sf];low=c+np.array([-7,-7,9])*s_eye;high=c+np.array([7,7,14])*s_eye
        assert np.all(points.min(0)>=low) and np.all(points.max(0)<=high)
        choose=np.all(sv.max(1)>=low,axis=1)&np.all(sv.min(1)<=high,axis=1);localf=sf[choose];assert len(localf)>0;used,rev=np.unique(localf,return_inverse=True)
        tissue=BVHTree.FromPolygons(transform(v[used]).tolist(),rev.reshape(-1,3).tolist(),all_triangles=True,epsilon=0.0)
        optical=BVHTree.FromPolygons(transform(points).tolist(),faces,all_triangles=True,epsilon=0.0)
        overlap=tissue.overlap(optical);assert len(overlap)==0, ('Optical assembly crosses source skin',oi,len(overlap));dist=[tissue.find_nearest(Vector(pp))[3] for pp in transform(out)[::8]]
        fit_checks.append({'eye':oi,'eye_similarity_scale':s_eye,'query_bounds_source_units':[low.tolist(),high.tolist()],'optical_bounds_enclosed':True,'local_tissue_triangles':len(localf),'conservative_overlap_pairs':len(overlap),'surface_crossing_excluded_by_disjoint_triangle_bounds':len(overlap)==0,'sampled_outer_cap_min_clearance_mm':float(min(dist))*1000,'sample_count':len(dist),'scope':'corneal optical mesh versus frozen skin within enclosing local box; zero overlap rules out surface crossings; nearest distances only sampled'})
        del sf,sv,localf,used,rev,tissue,optical
        optics.append({'object_id':oi,'center_input_units':c.tolist(),'radius_input_units':R,'iris_radius':5.5*s_eye,'pupil_radius':2.08*s_eye,'pupil_cavity_radius':2.15*s_eye,'maximum_cornea_extension':.395*s_eye,'corneal_radius':5.56*s_eye,'similarity_scale':s_eye,'reference_assembly_radius':Rref,'original_globe_cap':{'method':'Shader transparent aperture, source globe mesh unchanged','radius_actual':5.54*s_eye,'anterior_offset_actual':5*s_eye,'field_coordinates':'Inverse similarity to the unchanged reference eye chart'},'corneal_model':'Explicit air-cornea-aqueous boundaries with relative IORs; active interfaces_mnee uses refractive shadow caustics without transparent shadow override','legacy_hidden_lens_boundary_edges':0,'lens_signed_volume_design_units3':lens_volume,'front_center_z_relative':R+.395*s_eye,'lens_back_z_relative':LENS_BACK*s_eye,'minimum_lens_thickness_design_units':float(z.min()-LENS_BACK)*s_eye,'iris_z_range_relative':[float(ivv[:,2].min()-c[2]),float(ivv[:,2].max()-c[2])],'iris_to_lens_back_clearance_design_units':float(LENS_BACK*s_eye-(ivv[:,2].max()-c[2])),'iris_and_pupil_inside_homogeneous_lens_volume':a.optics=='mnee_filled_chamber','iris_triangles_front_facing':bool(np.all(np.cross(ivv[iff[:,1]]-ivv[iff[:,0]],ivv[iff[:,2]]-ivv[iff[:,0]])[:,2]>0))})
    head.data.polygons.foreach_set('material_index',mi)

if a.optics.startswith('interfaces_'):
    for optical_meta,per_eye_interface in zip(optics,interface_reports):
        for key in list(optical_meta):
            if key.startswith('lens_') or key.startswith('legacy_') or key in ('minimum_lens_thickness_design_units','iris_to_lens_back_clearance_design_units','iris_and_pupil_inside_homogeneous_lens_volume'):optical_meta.pop(key)
        optical_meta['interface_construction']=per_eye_interface
        optical_meta['surfaces']='Front, posterior, rear and rim boundaries; each shared medium interface appears once. Individual boundaries are intentionally open, with no overlapping glass volumes.'

groom_report=None
if a.mode=='appearance' and a.groom:
    sys.path.insert(0,str(ROOT))
    from rooted_groom import build_groom
    groom_report=build_groom(v,f,ids,semantics,transform,centers,radii,eye_scales=[frame.scale for frame in eye_frames])
    groom_report['source_sha256']=hashlib.sha256((ROOT/'rooted_groom.py').read_bytes()).hexdigest()
    groom_report['semantics_sha256']=hashlib.sha256(SEMANTICS.read_bytes()).hexdigest()
    from rooted_short_groom import build as build_short_groom
    groom_report['short_facial_groom']=build_short_groom(v,f,ids,semantics,transform,ROOT)

# Same world, light, camera and geometric skin in both appearance and diagnostic clay.
def aim(obj,target):obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
def light(name,loc,target,power,size,color):
    d=bpy.data.lights.new(name,'AREA');d.cycles.is_caustics_light=(a.optics in ('mnee','mnee_open_cap','mnee_filled_chamber','interfaces_mnee'));d.energy=power;d.shape='DISK';d.size=size;d.color=color;o=bpy.data.objects.new(name,d);scene.collection.objects.link(o);o.location=loc;aim(o,target)
scene.world.cycles.is_caustics_light=(a.optics in ('mnee','mnee_open_cap','mnee_filled_chamber','interfaces_mnee'))
if a.optics=='shadow_passthrough':
    lp=node(cornea,'ShaderNodeLightPath');tr=node(cornea,'ShaderNodeBsdfTransparent');mix=node(cornea,'ShaderNodeMixShader');link(cornea,lp.outputs['Is Shadow Ray'],mix.inputs[0]);link(cornea,cs.outputs[0],mix.inputs[1]);link(cornea,tr.outputs[0],mix.inputs[2]);link(cornea,mix.outputs[0],cornea.node_tree.nodes['Material Output'].inputs['Surface'])

if a.lighting=='studio':
    light('large warm key',(-.24,-.35,.31),(0,-.04,0),17,.26,(1,.92,.84))
    light('broad cool fill',(.30,-.16,.08),(0,-.07,-.01),8,.30,(.79,.87,1))
    light('soft shoulder rim',(.05,.18,.23),(0,-.03,0),22,.22,(1,.94,.88))
else:
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(1,1,1,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.045
    light('neutral side window',(-.38,-.24,.26),(0,-.04,0),18,.24,(1,1,1))
    light('neutral broad fill',(.32,-.10,.10),(0,-.07,-.01),3,.50,(1,1,1))
    light('neutral soft rim',(.20,.15,.30),(0,-.03,0),8,.35,(1,1,1))
lighting_report={'preset':a.lighting,'scope':'Original area-light arrangement; neutral RGB whites are not a calibrated daylight spectrum','world_color':list(scene.world.node_tree.nodes['Background'].inputs[0].default_value),'world_strength':scene.world.node_tree.nodes['Background'].inputs[1].default_value,'lights':[{'name':o.name,'location':list(o.location),'power_W':o.data.energy,'size_m':o.data.size,'color':list(o.data.color)} for o in scene.objects if o.type=='LIGHT']}
d=bpy.data.cameras.new('Matched camera');cam=bpy.data.objects.new('Matched camera',d);scene.collection.objects.link(cam);scene.camera=cam;d.type='ORTHO';d.clip_start=.001;d.clip_end=10
if a.view=='head':
    target=np.array([-.004,-.036,-.038]);cam.location=target+np.array([.574,-.819,.025])*.62;d.ortho_scale=.325
elif a.view=='head_front':
    profile=json.loads((ROOT/'fixed_frontal_camera.json').read_text())
    target=np.array(profile['target']);cam.location=profile['location'];d.ortho_scale=profile['ortho_scale']
elif a.view in ('cheek','forehead'):
    bpy.context.view_layer.update();xx,yy=(.032,-.019) if a.view=='cheek' else (.018,.055)
    hit,loc,normal,index=head.ray_cast(Vector((xx,-.20,yy)),Vector((0,1,0)));assert hit
    target=np.array(loc);cam.location=Vector(target)+Vector((.025,-.32,.012));d.ortho_scale=.012
else:
    c=centers[1];target=transform(np.array([[c[0]+1,c[1]-2,c[2]+11]]))[0];cam.location=Vector(target)+Vector((.095,-.43,.024));d.ortho_scale=.049
if a.view=='eye' and a.eye_camera=='frozen_v8':
    profile=json.loads((ROOT/'frozen_eye_camera.json').read_text())['camera']
    target=np.array(profile['target']);cam.location=profile['location'];d.ortho_scale=profile['ortho_scale']
aim(cam,target)
directional_report=None
if a.view in ('cheek','forehead'):
    # Matched directional reflection diagnostic. Relief dimensions are NOT enlarged.
    N=Vector(normal).normalized();V=(cam.location-Vector(target)).normalized()
    assert N.dot(V)>0
    L=(2*N.dot(V)*N-V).normalized()
    for ob in scene.objects:
        if ob.type=='LIGHT':ob.data.energy*=.15
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.03
    light('directional surface response strip',Vector(target)+L*.16,target,1.5,.04,(1.,.97,.93))
    strip=bpy.data.lights['directional surface response strip'];strip.shape='RECTANGLE';strip.size=.04;strip.size_y=.008
    directional_report={'original_lights_power_factor':.15,'world_strength':.03,'strip_power_W':1.5,'strip_distance_m':.16,'strip_dimensions_m':[.04,.008],'strip_direction':list(L),'geometric_target_normal':list(N),'purpose':'Matched directional close lighting, unamplified relief'}

manifest={'description':'Actual Cycles CPU preview of original geometry and procedural appearance, not an AI generated image','source':source_meta,'scene_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'blender':bpy.app.version_string,'renderer':'Cycles CPU','size':a.size,'samples':a.samples,'threads':scene.render.threads,'denoised':False,'color_management':{'view':'AgX','look':'Medium High Contrast','exposure':-2.5},'mode':a.mode,'view':a.view,'path_traced':True,'lighting':lighting_report,'directional_lighting_diagnostic':directional_report,'surface_response':surface_report,'sss':{'method':sb.subsurface_method,'weight':'Per-vertex DermalResponseWeight; nominal CLI value overridden','diffuse_weight':'One minus local dermal weight','model':'Linked regional diffuse/Random Walk Skin approximation, no measured layered biology','radius':[1,.48,.24],'scale_m':'Per-vertex EffectiveTissuePath; authored scene-unit scale','artistic_uncalibrated':True},'skin_detail':{'source':'Original procedural coordinate fields; no maps','active_surface_response':surface_report,'displacement':False},'optical_transport':{'interface_assembly':interface_reports if a.mode=='appearance' and a.optics.startswith('interfaces_') else None,'mode':a.optics,'MNEE_enabled':a.optics in ('mnee','mnee_open_cap','mnee_filled_chamber','interfaces_mnee'),'shadow_ray_passthrough':a.optics in ('shadow_passthrough','interfaces_shadow'),'camera_refraction_and_materials_unchanged':a.optics!='no_cornea','cornea_hidden_control':a.optics=='no_cornea','original_globe_faces_removed':removed_caps,'cap_removal_scope':'diagnostic only; globe faces touching radius5.54 aperture removed; source file and skin faces unchanged','MNEE_note':'Approximate caustic sampling with known curved/rough surface limitations; not validated optics.'},'geometry_revision':'preferred-v53-integrated-tissue-eye-groom-candidate-v1','eye_similarity_bindings':binding_reports,'groom':groom_report,'lip_field':'Exact refreshed candidate vermilion/inner-return vertex semantics','skin_region_fields':skin_field_report,'linked_tissue_response':{'fields':linked_field_report,'shader':linked_shader_report} if a.mode=='appearance' else None,'skin_field_source_sha256':hashlib.sha256((ROOT/'coordinated_tissue.py').read_bytes()).hexdigest(),'eyes':optics,'iris_stroma_audits':iris_audits,'sclera_fields':sclera_reports,'eye_fit_checks':fit_checks,'camera_control':a.eye_camera if a.view=='eye' else ('fixed_frontal_v1' if a.view=='head_front' else 'fixed_head_or_skin_crop'),'camera':{'location':list(cam.location),'rotation':list(cam.rotation_euler),'ortho_scale':d.ortho_scale,'target':target.tolist()},'provenance':{'photos':False,'scans':False,'imported_heads':False,'generated_image_textures':False,'image_reference_sampling':False,'mesh_preserved':True},'limitations':['Artistic, uncalibrated tissue and optics; complete optical similarity preserves authored ratios, not validated human scaling','Unselected coordinated tissue/eye/groom candidate on exact preferred-v53 rest geometry','This candidate uses explicit interfaces_mnee with no shadow passthrough; Cycles caustic sampling remains approximate and unvalidated','Pores and iris are procedural approximations, not measured biology','No tear film or blink rig; brows, lashes and short facial fibers are uncalibrated static groom; source dimensions uncalibrated','No claim of anatomical realism or physical validation']}
if a.mode=='clay':
    manifest['sss']={'method':None,'weight':0}
    manifest['skin_detail']={'source':'Neutral constant clay','bump':False,'displacement':False}
    manifest['surface_response']=None
    manifest['skin_region_fields']=None
    manifest['lip_field']=None
    manifest['optical_transport']={'mode':'Original opaque source globes in clay','shadow_ray_passthrough':False,'interface_assembly':None}
path=ROOT/f'{a.view}_{a.mode}_{a.lighting}_{a.surface_profile}_raw.png';scene.render.filepath=str(path)
sys.path.insert(0,str(ROOT.parent))
from denoise_bridge import setup as setup_guides,run as denoise_guides
guide_paths=setup_guides(scene,ROOT,path.stem) if a.view in ('cheek','forehead') else None
print('CYBR_SCENE_READY '+json.dumps({'output':str(path),'rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'source':source_meta,'centers':[c.tolist() for c in centers],'eye_fit_checks':fit_checks}),flush=True)
if a.validate_only:
    (ROOT/('optical_fit_validation_reference.json' if a.reference_control else 'optical_fit_validation.json')).write_text(json.dumps(manifest,indent=2)+'\n');print('CYBR_VALIDATED '+json.dumps(fit_checks),flush=True);sys.exit(0)
del v,f,ids;gc.collect();bpy.ops.render.render(write_still=True)
(ROOT/f'{a.view}_{a.mode}_{a.lighting}_{a.surface_profile}.json').write_text(json.dumps(manifest,indent=2)+'\n')
if guide_paths:
    from skin_guided_denoise import run as guided_skin
    manifest['guided_skin_denoise']=guided_skin(scene,guide_paths,path,ROOT/f'{a.view}_{a.mode}_{a.lighting}_{a.surface_profile}_guided.png')
manifest['elapsed_seconds']=round(time.time()-START,2);manifest['peak_rss_mib']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)
(ROOT/f'{a.view}_{a.mode}_{a.lighting}_{a.surface_profile}.json').write_text(json.dumps(manifest,indent=2)+'\n');print('CYBR_DONE '+str(path),flush=True)
