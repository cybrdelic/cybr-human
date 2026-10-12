"""Soft pore wells + low-amplitude elongated microrelief, never Voronoi edges.
All amplitudes and chart densities are authored static appearance parameters.
No displacement, image texture, measured pore claim or cellular/leather network.
"""
import bpy

def build(skin,sb,node,link,mathnode):
    p=node(skin,'ShaderNodeNewGeometry');sep=node(skin,'ShaderNodeSeparateXYZ');link(skin,p.outputs['Position'],sep.inputs[0]);chart=node(skin,'ShaderNodeCombineXYZ');link(skin,sep.outputs['X'],chart.inputs['X']);link(skin,sep.outputs['Z'],chart.inputs['Y'])
    vor=node(skin,'ShaderNodeTexVoronoi');vor.voronoi_dimensions='2D';vor.feature='F1';vor.distance='EUCLIDEAN';vor.inputs['Scale'].default_value=4500.;vor.inputs['Randomness'].default_value=1;link(skin,chart.outputs[0],vor.inputs['Vector'])
    # Randomized soft radius, no hard circular rim and no Voronoi cell boundaries.
    csep=node(skin,'ShaderNodeSeparateColor');link(skin,vor.outputs['Color'],csep.inputs[0]);radius=mathnode(skin,'MULTIPLY_ADD',csep.outputs[0],.08);radius.node.inputs[2].default_value=.13
    q=mathnode(skin,'DIVIDE',vor.outputs['Distance'],radius);q=mathnode(skin,'MULTIPLY',q,q);q=mathnode(skin,'MULTIPLY',q,-1.);pit=mathnode(skin,'EXPONENT',q);pit=mathnode(skin,'MULTIPLY',pit,-.85)
    # Smooth authored horizontal forehead to oblique cheek orientation.
    fade=node(skin,'ShaderNodeMapRange');fade.clamp=True;fade.inputs['From Min'].default_value=.020;fade.inputs['From Max'].default_value=.045;fade.inputs['To Min'].default_value=1.;fade.inputs['To Max'].default_value=0.;link(skin,sep.outputs['Z'],fade.inputs['Value']);angle=mathnode(skin,'MULTIPLY',fade.outputs['Result'],.38);sign=mathnode(skin,'SIGN',sep.outputs['X']);angle=mathnode(skin,'MULTIPLY',angle,sign)
    rot=node(skin,'ShaderNodeVectorRotate');rot.rotation_type='AXIS_ANGLE';rot.inputs['Axis'].default_value=(0,0,1);link(skin,chart.outputs[0],rot.inputs['Vector']);link(skin,angle,rot.inputs['Angle'])
    stretch=node(skin,'ShaderNodeVectorMath');stretch.operation='MULTIPLY';stretch.inputs[1].default_value=(330,3300,1);link(skin,rot.outputs['Vector'],stretch.inputs[0]);crease=node(skin,'ShaderNodeTexNoise');crease.inputs['Scale'].default_value=1;crease.inputs['Detail'].default_value=1;link(skin,stretch.outputs[0],crease.inputs['Vector']);fine=mathnode(skin,'MULTIPLY_ADD',crease.outputs['Fac'],.16);fine.node.inputs[2].default_value=-.08
    height=mathnode(skin,'ADD',pit,fine);b=node(skin,'ShaderNodeBump');b.inputs['Distance'].default_value=.000012;b.inputs['Strength'].default_value=1;link(skin,height,b.inputs['Height']);link(skin,b.outputs['Normal'],sb.inputs['Normal'])
    return {'method':'Soft Gaussian nearest-point pore wells plus low-amplitude elongated Noise microrelief','voronoi_edges_used':False,'projected_pore_cell_spacing_m':1/4500,'pore_characteristic_radius_m_range':[.13/4500,.21/4500],'maximum_pore_depression_m':.85*.000012,'microrelief_peak_height_m':.08*.000012,'direction_field':'Authored horizontal forehead to mirrored oblique cheek, blended across source Y20-45 design units','bump_only_no_mesh_displacement':True,'limits':'Static authored frontal chart; not measured skin or a deformation-ready UV parameterization.'}
