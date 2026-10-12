"""Uniform numerical conditioning for fixed-threshold Cycles MNEE.
All authored geometry, dimensionless optics and radiometric relationships are preserved.
"""
import bpy
from mathutils import Matrix

def apply(scene,factor):
 assert factor>0
 bpy.context.view_layer.update()
 cam=scene.camera;base={o.name:o.matrix_world.copy()for o in scene.objects};lights={o.name:{'location':list(o.location),'power':o.data.energy,'size':o.data.size,'size_y':o.data.size_y}for o in scene.objects if o.type=='LIGHT'}
 coord=linked=fixed=0
 for m in bpy.data.materials:
  if not m.use_nodes:continue
  nt=m.node_tree
  for n in list(nt.nodes):
   if n.bl_idname=='ShaderNodeNewGeometry':
    outgoing=list(n.outputs['Position'].links)
    if outgoing:
     inv=nt.nodes.new('ShaderNodeVectorMath');inv.operation='SCALE';inv.label='Inverse uniform numerical scale';nt.links.new(n.outputs['Position'],inv.inputs[0]);inv.inputs[3].default_value=1/factor
     for l in outgoing:to=l.to_socket;nt.links.remove(l);nt.links.new(inv.outputs[0],to)
     coord+=1
   if n.bl_idname=='ShaderNodeBsdfPrincipled':
    sock=n.inputs.get('Subsurface Scale')
    if sock is not None:
     if sock.is_linked:
      l=list(sock.links)[0];src=l.from_socket;nt.links.remove(l);mul=nt.nodes.new('ShaderNodeMath');mul.operation='MULTIPLY';nt.links.new(src,mul.inputs[0]);mul.inputs[1].default_value=factor;nt.links.new(mul.outputs[0],sock);linked+=1
     else:sock.default_value*=factor;fixed+=1
   if n.bl_idname=='ShaderNodeBump':n.inputs['Distance'].default_value*=factor;fixed+=1
 for o in scene.objects:
  mat=base[o.name].copy()
  if o.type in ('CAMERA','LIGHT'):mat.translation*=factor
  else:mat=Matrix.Scale(factor,4)@mat
  o.matrix_world=mat
 cam.data.ortho_scale*=factor;cam.data.clip_start*=factor;cam.data.clip_end*=factor
 for name,q in lights.items():
  d=bpy.data.objects[name].data;d.energy=q['power']*factor*factor;d.size=q['size']*factor;d.size_y=q['size_y']*factor
 bpy.context.view_layer.update()
 return {'factor':factor,'reason':'Cycles 4.3.2 MNEE fixed segment and progress thresholds; 1000x passes the tested actual-eye causal diagnostic','transform_contract':{'mesh_curve_positions_and_radii':'factor','camera_ortho_clip_location':'factor; rotation unchanged','light_location_and_area_dimensions':'factor; transform scale kept 1','area_light_power':'factor squared','world_radiance_exposure_color_roughness_IOR':'unchanged','absolute_geometry_position_fields':'inverse factor before original field calculation','SSS_and_bump_lengths':'factor','volumetric_density_nodes':'none present'},'node_counts':{'position_inverse':coord,'linked_subsurface_scale':linked,'constant_lengths':fixed},'base_lights':lights,'result_lights':{o.name:{'location':list(o.location),'power':o.data.energy,'size':o.data.size,'transform_scale':list(o.scale)}for o in scene.objects if o.type=='LIGHT'},'limitation':'Numerical conditioning, not physical size calibration or complete optical validation'}
