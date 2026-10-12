#!/usr/bin/env python3
"""Optional original CPU broad-source clay renderer.

Uses the unchanged clay_render camera rasterizer and geometric shadow mapper.
Integrates a distant angular disk light with deterministic equal-solid-angle
sunflower samples. Every sample independently reprojects the actual mesh into a
light-space depth map. No image blur, painted masks, textures, or mesh edits.
This is finite-sample shadow-map integration, not a path-traced finite emitter.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from clay_render import (PALETTES, validate_mesh, vertex_normals, make_framing,
                         _rotation, _rasterize, _shadow_visibility)


def disk_light_samples(count=24, angular_radius=30., direction=(-.55,.75,1.)):
    """Equal-solid-angle samples across a distant circular emitter's cone."""
    if not isinstance(count,(int,np.integer)) or not 1 <= count <= 128:
        raise ValueError('sample count must be an integer in [1,128]')
    if not math.isfinite(angular_radius) or not 0 < angular_radius < 80:
        raise ValueError('angular_radius must be between 0 and 80 degrees')
    axis=np.asarray(direction,dtype=float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9:
        raise ValueError('light direction must be a finite nonzero 3-vector')
    axis/=np.linalg.norm(axis)
    up=np.array([0.,1.,0.]) if abs(axis[1])<.95 else np.array([1.,0.,0.])
    right=np.cross(up,axis);right/=np.linalg.norm(right)
    up=np.cross(axis,right)
    i=np.arange(count,dtype=float)
    cos_theta=1-(i+.5)/count*(1-math.cos(math.radians(angular_radius)))
    sin_theta=np.sqrt(1-cos_theta**2)
    phi=i*(math.pi*(3-math.sqrt(5)))
    return (cos_theta[:,None]*axis + sin_theta[:,None]*
            (np.cos(phi)[:,None]*right+np.sin(phi)[:,None]*up))


def render_mesh_soft(vertices,triangles,material_ids=None,*,angle=-35.,size=720,
                     framing=None,normals=None,material='gray',supersample=2,
                     samples=24,angular_radius=30.,light_direction=(-.55,.75,1.),
                     shadow_resolution=1536,shadow_bias=.18,ambient=.25,
                     light_energy=.80,background=(239,238,234),backend='auto',
                     return_buffers=False,progress=None,contact_refine=False,
                     contact_normal_offset=.008,contact_tmin=.025):
    """Return a PIL RGB raw-mesh render, optionally with diagnostic buffers.

    Old unshadowed/three-lobe rendering lives unchanged in clay_render.py.
    `ambient` is an explicitly unoccluded studio fill approximation, not GI.
    Smooth normals and shared orthographic projection are identical to that path.
    The only image resampling is final supersample antialiasing, never a blur.
    Optional contact_refine uses original exact BVH triangle rays at positive-
    diffuse pixels whose shadow-map PCF taps disagree. Other samples retain
    the old shadow-map path. Ray self-hit tolerances are in mesh units.
    """
    vertices,triangles,material_ids=validate_mesh(vertices,triangles,material_ids)
    if not isinstance(size,(int,np.integer)) or not 16<=size<=4096:
        raise ValueError('size must be an integer in [16,4096]')
    if supersample not in (1,2,3,4) or backend not in ('auto','native','numpy'):
        raise ValueError('invalid supersampling or raster backend')
    if not isinstance(shadow_resolution,(int,np.integer)) or not 32<=shadow_resolution<=4096:
        raise ValueError('invalid shadow map resolution')
    if not all(math.isfinite(x) and x>=0 for x in (shadow_bias,ambient,light_energy)):
        raise ValueError('lighting and bias values must be finite and nonnegative')
    if not math.isfinite(angle) or material not in PALETTES:
        raise ValueError('invalid camera angle or material')
    if not all(math.isfinite(x) and x>=0 for x in (contact_normal_offset,contact_tmin)):
        raise ValueError("Contact ray offsets must be finite and nonnegative")
    lights=disk_light_samples(samples,angular_radius,light_direction)
    framing=make_framing(vertices,angles=[angle]) if framing is None else framing
    center=np.asarray(framing['center'],dtype=float);span=float(framing['span'])
    if center.shape!=(3,) or not np.isfinite(center).all() or not math.isfinite(span) or span<=0:
        raise ValueError('invalid shared framing')
    normals=vertex_normals(vertices,triangles) if normals is None else np.asarray(normals,dtype=float)
    if normals.shape!=vertices.shape or not np.isfinite(normals).all():
        raise ValueError('invalid normals')
    n=size*supersample;rotation=_rotation(angle)
    camera_vertices=np.ascontiguousarray((vertices-center)@rotation.T)
    camera_normals=np.ascontiguousarray(normals@rotation.T)
    screen=camera_vertices.copy()
    screen[:,0]=screen[:,0]*n/span+n/2
    screen[:,1]=-screen[:,1]*n/span+n/2
    depth,normal_buffer,material_buffer=_rasterize(screen,camera_normals,triangles,material_ids,n,backend)
    mask=material_buffer>=0
    illumination=np.full(depth.shape,ambient,dtype=np.float32)
    visibility_average=np.zeros_like(depth)
    highlight=np.zeros_like(depth)
    contact_bvh=None;contact_rays=0;contact_changed=0
    if contact_refine:
        from clay_contact import ContactBVH
        contact_bvh=ContactBVH(camera_vertices,triangles)
    for index,light in enumerate(lights):
        visibility=_shadow_visibility(camera_vertices,camera_normals,triangles,material_ids,
                     depth,normal_buffer,mask,span,light,shadow_resolution,shadow_bias,backend)
        cosine=np.maximum(normal_buffer@light,0)
        if contact_bvh is not None:
            # Mixed PCF taps are geometric visibility uncertainty, not an anatomy mask.
            ambiguous=mask&(visibility>1e-6)&(visibility<1-1e-6)&(cosine>0)
            iy,ix=np.nonzero(ambiguous)
            if len(ix):
                points=np.column_stack([(ix+.5-n/2)*span/n,-(iy+.5-n/2)*span/n,depth[iy,ix]])
                blocked=contact_bvh.occluded(points,light,normals=normal_buffer[iy,ix],
                            normal_offset=contact_normal_offset,tmin=contact_tmin)
                exact=(~blocked).astype(np.float32)
                contact_rays+=len(ix);contact_changed+=int(np.count_nonzero(abs(exact-visibility[iy,ix])>1e-6))
                visibility[iy,ix]=exact
        illumination+=light_energy/samples*cosine*visibility
        visibility_average+=visibility/samples
        half_vector=light+np.array([0.,0.,1.]);half_vector/=np.linalg.norm(half_vector)
        highlight+=.014/samples*np.maximum(normal_buffer@half_vector,0)**24*visibility
        if progress is not None:progress(index+1,samples)
    if contact_bvh is not None:contact_bvh.close()
    base=PALETTES[material][np.maximum(material_buffer,0)]**2.2
    rgb=np.clip(base*illumination[...,None]+highlight[...,None],0,1)**(1/2.2)
    background=np.asarray(background,dtype=float)
    if background.shape!=(3,) or not np.isfinite(background).all() or background.min()<0 or background.max()>255:
        raise ValueError('invalid RGB background')
    rgb[~mask]=background/255
    image=Image.fromarray(np.round(rgb*255).astype(np.uint8))
    if supersample!=1:image=image.resize((size,size),Image.Resampling.LANCZOS)
    if return_buffers:
        return {'image':image,'depth':depth,'normals':normal_buffer,'mask':mask,
                'material_ids':material_buffer,'shadow_visibility':visibility_average,
                'framing':framing,'light_directions':lights.tolist(),
                'contact_refinement':{'enabled':bool(contact_refine),'rays':contact_rays,'refined_visibility_samples':contact_changed,
                    'normal_offset':contact_normal_offset,'tmin':contact_tmin,'selection':'positive-diffuse pixels with mixed PCF taps'}}
    return image



def render_views_soft(vertices,triangles,material_ids=None,*,angles=(0,-35,-90),framing=None,**kwargs):
    """Return matched-scale images from one unchanged mesh and shared world frame."""
    if framing is None:framing=make_framing(vertices,angles=angles)
    if kwargs.get('normals') is None:kwargs['normals']=vertex_normals(vertices,triangles)
    return [render_mesh_soft(vertices,triangles,material_ids,angle=angle,framing=framing,**kwargs)
            for angle in angles]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--angle',type=float,default=-35);p.add_argument('--size',type=int,default=720)
    p.add_argument('--angles',type=float,nargs='+',help='optional batch of matched views; overrides --angle')
    p.add_argument('--center',type=float,nargs=3,default=[0,-8,30]);p.add_argument('--span',type=float,default=250)
    p.add_argument('--samples',type=int,default=24);p.add_argument('--angular-radius',type=float,default=30)
    p.add_argument('--supersample',type=int,default=2)
    p.add_argument('--shadow-resolution',type=int,default=1536)
    p.add_argument('--shadow-bias',type=float,default=.18)
    p.add_argument('--ambient',type=float,default=.25);p.add_argument('--light-energy',type=float,default=.80)
    p.add_argument('--preserve-materials',action='store_true')
    p.add_argument('--contact-refine',action='store_true',help='exact triangle rays at ambiguous PCF visibility edges')
    args=p.parse_args(argv);start=time.perf_counter()
    with np.load(args.input,allow_pickle=False) as m:
        v=m['vertices'];f=m['triangles']
        ids=m['material_ids'] if args.preserve_materials and 'material_ids' in m else np.zeros(len(f),dtype=np.int32)
        normals=m['normals'] if 'normals' in m else None
    framing={'center':args.center,'span':args.span}
    args.out.mkdir(parents=True,exist_ok=True)
    angles=args.angles if args.angles is not None else [args.angle]
    images=[];views=[]
    if normals is None:normals=vertex_normals(v,f)
    for angle in angles:
        result=render_mesh_soft(v,f,ids,normals=normals,angle=angle,size=args.size,
                  framing=framing,supersample=args.supersample,samples=args.samples,
                  angular_radius=args.angular_radius,shadow_resolution=args.shadow_resolution,
                  shadow_bias=args.shadow_bias,ambient=args.ambient,light_energy=args.light_energy,
                  return_buffers=True,contact_refine=args.contact_refine,progress=lambda i,n:print(f'Yaw {angle:g}: geometric shadow sample {i}/{n}',flush=True))
        name={0:'front',-35:'threequarter',35:'threequarter',-90:'profile',90:'profile'}.get(angle,f'yaw_{angle:g}')
        filename=f'{name}.png' if args.angles is not None else 'actual_threequarter_soft.png'
        result['image'].save(args.out/filename)
        images.append(result['image']);views.append({'name':name,'angle_degrees':angle,'file':filename})
    if len(images)>1:
        sheet=Image.new('RGB',(args.size*len(images),args.size+55),(239,238,234))
        draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=17)
        for i,(im,view) in enumerate(zip(images,views)):
            sheet.paste(im,(i*args.size,0))
            draw.text((i*args.size+18,args.size+10),f"{view['name'].upper()} / {view['angle_degrees']:g} DEG",font=font,fill=(65,65,62))
        sheet.save(args.out/'actual_studio_checkpoint.png')
    manifest={'renderer':'Original CPU triangle rasterizer and geometric shadow-map integration',
              'input_mesh':str(args.input),'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'vertices':len(v),'triangles':len(f),'mesh_modified':False,
              'views':views,'framing':framing,'size':args.size,'supersample':args.supersample,
              'light_model':'Distant circular angular emitter; equal-solid-angle deterministic sunflower quadrature',
              'shadow_samples':args.samples,'angular_radius_degrees':args.angular_radius,
              'light_directions_camera_space':result['light_directions'],
              'shadow_resolution':args.shadow_resolution,'shadow_bias_mesh_units':args.shadow_bias,
              'shadow_filter':'Receiver-plane-corrected 3x3 PCF per independent sample',
              'ambient_unoccluded_fill':args.ambient,'integrated_direct_light_energy':args.light_energy,
              'uniform_clay':not args.preserve_materials,'ray_traced':bool(args.contact_refine),'path_traced':False,'finite_distance_emitter':False,
              'ray_tracing_scope':'mixed-PCF contact shadow visibility only' if args.contact_refine else None,
              'hybrid_exact_contact_rays':args.contact_refine,'contact_refinement':result['contact_refinement'],
              'ambient_occlusion':False,'global_illumination':False,'textures':False,'image_blur':False,
              'cavity_paint':False,'reference_image_used_as_texture':False,
              'antialiasing':'Supersampled raster with Lanczos resolve only',
              'limitations':'Finite source quadrature can leave bands. Fully classified but undersampled map contacts are not refined. Not path tracing; no subsurface scattering or indirect-light transport.',
              'elapsed_seconds':round(time.perf_counter()-start,3)}
    (args.out/'render_settings.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(args.out/('actual_studio_checkpoint.png' if len(images)>1 else views[0]['file']),flush=True)

if __name__=='__main__':main()
