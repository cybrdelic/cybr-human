"""Original gentle scleral color and sub-resolution peripheral vessel fields.
No image map, shadow painting, UV photograph or injected normal change.
"""
import numpy as np

def build(m,s,dx,dy,node,link,M,seed):
    x=M(m,'MULTIPLY',dx,1000.);y=M(m,'MULTIPLY',dy,1000.);ax=M(m,'ABSOLUTE',x);shift=M(m,'SUBTRACT',ax,6.)
    p=node(m,'ShaderNodeMapRange');p.clamp=True;p.inputs['From Min'].default_value=5.8;p.inputs['From Max'].default_value=10.8;link(m,ax,p.inputs['Value'])
    gate=node(m,'ShaderNodeMapRange');gate.clamp=True;gate.inputs['From Min'].default_value=6.6;gate.inputs['From Max'].default_value=8.;link(m,ax,gate.inputs['Value'])
    rng=np.random.default_rng(seed);veins=None
    for j,base in enumerate([-3.2,-1.1,1.2,3.1]):
        slope=float(rng.uniform(-.18,.18));phase=float(rng.uniform(0,6.28));w=.055
        arg=M(m,'MULTIPLY_ADD',shift,float(rng.uniform(.65,1.05)));arg.node.inputs[2].default_value=phase
        wave=M(m,'MULTIPLY',M(m,'SINE',arg),.18);line=M(m,'MULTIPLY_ADD',shift,slope);line.node.inputs[2].default_value=base;line=M(m,'ADD',line,wave)
        dist=M(m,'DIVIDE',M(m,'SUBTRACT',y,line),w);sq=M(m,'MULTIPLY',dist,dist);v=M(m,'EXPONENT',M(m,'MULTIPLY',sq,-1.))
        # One faint fork on alternate vessels; no dense uniformly repeated web.
        if j%2==0:
            fork=M(m,'MAXIMUM',M(m,'SUBTRACT',ax,8.3),0.);offset=M(m,'MULTIPLY',fork,float(rng.choice([-1,1]))*.48);branch=M(m,'ADD',line,offset);bd=M(m,'DIVIDE',M(m,'SUBTRACT',y,branch),w*.75);bv=M(m,'EXPONENT',M(m,'MULTIPLY',M(m,'MULTIPLY',bd,bd),-1.));v=M(m,'MAXIMUM',v,M(m,'MULTIPLY',bv,.55))
        veins=v if veins is None else M(m,'MAXIMUM',veins,v)
    veins=M(m,'MULTIPLY',veins,gate.outputs['Result']);factor=M(m,'ADD',M(m,'MULTIPLY',p.outputs['Result'],.12),M(m,'MULTIPLY',veins,.16))
    mix=node(m,'ShaderNodeMixRGB');mix.blend_type='MIX';mix.inputs[1].default_value=(.64,.64,.60,1);mix.inputs[2].default_value=(.65,.26,.23,1);link(m,factor,mix.inputs[0]);link(m,mix.outputs[0],s.inputs['Base Color'])
    return {'seed':seed,'base_linear_rgb':[.64,.64,.60],'peripheral_mix_max':.12,'vessel_mix_max':.16,'vessel_characteristic_radius_design_units':.055,'primary_paths':4,'small_forks':2,'scope':'Original intrinsic scleral coloration only; same roughness/SSS/geometry and no image textures','limits':'Authored static vessel paths, not a measured vascular network or tissue transport model'}
