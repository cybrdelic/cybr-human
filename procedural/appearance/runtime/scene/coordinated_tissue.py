"""Original linked region response: authored absorption/path/oil fields, no maps.
No spectral/biological calibration is claimed. All texture coordinates are rest geometry.
"""
import numpy as np

def regions(v):
 x,y,z=np.asarray(v).T;ax=np.abs(x);front=np.clip((z-45)/45,0,1)
 def g(cx,cy,sx,sy):return np.exp(-.5*((ax-cx)/sx)**2-.5*((y-cy)/sy)**2)*front
 return {'cheek':g(37,-14,18,19),'nose':g(0,-22,10,17),'lid':g(31,11,13,6.5),'forehead':g(0,62,43,26),'ear':np.exp(-.5*((ax-77)/9)**2-.5*((y+2)/29)**2),'chin':g(0,-81,30,18),'neck':g(0,-121,48,24)}

def fields(v):
 q=regions(v);n=len(v)
 # Broad smooth biological proxies share the same anatomical fields. No noise.
 melanin=.030*q['forehead']+.022*q['chin']+.012*q['neck']-.012*q['lid']
 blood=.20*q['cheek']+.23*q['nose']+.19*q['lid']+.31*q['ear']+.035*q['chin']
 path=np.clip(.00060+.00022*q['cheek']-.00024*q['lid']-.00012*q['forehead']+.00009*q['ear'],.00030,.00090)
 # Beer-Lambert-style relative pigment attenuation; RGB coefficients are authored proxies.
 color=np.tile(np.array([.438,.252,.168]),(n,1));color*=np.exp(-melanin[:,None]*np.array([.65,1.,1.45]))
 color=color*(1-blood[:,None])+np.array([.475,.174,.136])*blood[:,None]
 oil=np.clip(.63*q['forehead']+.88*q['nose']+.35*q['lid']+.18*q['chin'],0,1)
 rough=np.clip(.585-.22*oil+.015*q['cheek'],.355,.615)
 report={'model':'Linked smooth anatomical pigment, vascular fraction, effective subsurface path and oil fields','random_mottle_added':False,'independent_color_noise':False,'global_tint_fit':False,'effective_path_scene_units_range':[float(path.min()),float(path.max())],'roughness_range':[float(rough.min()),float(rough.max())],'oil_fraction_range':[float(oil.min()),float(oil.max())],'limits':'RGB absorption/path proxies and mixture weights are authored, uncalibrated; no measured multilayer biology or physical unit claim.'}
 return np.column_stack((color,np.ones(n))).astype(np.float32),rough.astype(np.float32),report

def attach_fields(head,v,semantics):
 q=regions(v);path=np.clip(.00060+.00022*q['cheek']-.00024*q['lid']-.00012*q['forehead']+.00009*q['ear'],.00030,.00090)
 weight=np.clip(.58+.08*q['cheek']-.10*q['lid'],.46,.68)
 vv=semantics['vermilion_vertex_ids'];path[vv]=.00040;weight[vv]=.63
 for name,values in [('EffectiveTissuePath',path),('DermalResponseWeight',weight)]:
  a=head.data.attributes.new(name=name,type='FLOAT',domain='POINT');a.data.foreach_set('value',values.astype(np.float32))
 # Rest coordinates are an actual point attribute, ready to advect with a later qualified pose.
 a=head.data.attributes.new(name='PreferredRestPosition',type='FLOAT_VECTOR',domain='POINT');a.data.foreach_set('vector',np.asarray(v,dtype=np.float32).ravel())
 return {'rest_coordinate_attribute':'PreferredRestPosition','geometry_changed':False,'scattering_path_and_weight_linked_to_same_region_fields':True}

def apply_material(skin,sb,node,link):
 p=node(skin,'ShaderNodeAttribute');p.attribute_name='EffectiveTissuePath';link(skin,p.outputs['Fac'],sb.inputs['Subsurface Scale'])
 w=node(skin,'ShaderNodeAttribute');w.attribute_name='DermalResponseWeight';link(skin,w.outputs['Fac'],sb.inputs['Subsurface Weight'])
 sb.inputs['Subsurface Radius'].default_value=(1.,.48,.24);sb.inputs['Subsurface Anisotropy'].default_value=.7;sb.inputs['IOR'].default_value=1.43
 # The physical-size pores are far below one whole-head pixel. Filter their normal modulation;
 # do not enlarge pores or add paint-like spots. Cycles samples actual groom coverage.
 for n in skin.node_tree.nodes:
  if n.bl_idname=='ShaderNodeBump':n.inputs['Strength'].default_value*=.22
 return {'added_coat':False,'surface_ior':1.43,'baseline_pore_dimensions_preserved':True,'pore_normal_modulation_filter':.22,'display_context':'560-pixel whole-head cameras; authored filtering, not a calibrated spatial-frequency model'}
