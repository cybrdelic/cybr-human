// Unreduced implicit tetrahedral FEM. Stable Neo-Hookean growth energy,
// assembled sparse block PCG, energy/Jacobian guarded Newton steps. No pose interpolation.
struct Node { rest:vec4f, x:vec4f, old:vec4f, velocity:vec4f, predict:vec4f, gradient:vec4f, diagonal:vec4f, direction:vec4f, r:vec4f, z:vec4f, p:vec4f, ap:vec4f, trial:vec4f, normal:vec4f, attachment:vec4f, incidence:vec4u }
struct Tet { ids:vec4u, grad:array<vec4f,4>, material:vec4f, hydro:array<vec4f,4>, force:array<vec4f,4>, F:array<vec4f,3>, metric:vec4f, growth:vec4f }
struct Params { counts:vec4u, settings:vec4f, grab:vec4f, grabMeta:vec4u, integration:vec4u }
@group(0) @binding(0) var<storage,read_write> nodes:array<Node>;
@group(0) @binding(1) var<storage,read_write> tets:array<Tet>;
@group(0) @binding(2) var<storage,read> adjacency:array<u32>;
@group(0) @binding(3) var<storage,read_write> partial:array<vec4f>;
struct Solver {values:array<f32,16>}
@group(0) @binding(4) var<storage,read_write> solver:Solver;
@group(0) @binding(5) var<storage,read_write> output:array<vec4f>;
struct Edge {edgeInfo:vec4u, c0:vec4f,c1:vec4f,c2:vec4f}
@group(0) @binding(6) var<storage,read_write> edges:array<Edge>;
@group(0) @binding(7) var<storage,read> edgeCodes:array<u32>;
@group(0) @binding(8) var<uniform> params:Params;
@group(0) @binding(9) var<storage,read_write> dispatchArgs:array<vec4u>;
var<workgroup> controlEnabled:u32;
var<workgroup> sums:array<vec4f,128>;
fn cgDispatch(enabled:bool){let yes=select(0u,1u,enabled);dispatchArgs[0]=vec4u(yes*((params.counts.x+127u)/128u),1u,1u,0u);dispatchArgs[1]=vec4u(yes*((params.counts.x+127u)/128u),1u,1u,0u);dispatchArgs[2]=vec4u(yes,1u,1u,0u);dispatchArgs[3]=vec4u(yes*((params.grabMeta.z+127u)/128u),1u,1u,0u);}
fn newtonDispatch(enabled:bool){let yes=select(0u,1u,enabled);dispatchArgs[8]=vec4u(yes*((params.counts.x+127u)/128u),1u,1u,0u);dispatchArgs[9]=vec4u(yes*((params.counts.y+127u)/128u),1u,1u,0u);dispatchArgs[10]=vec4u(yes*params.counts.w,1u,1u,0u);}
fn lineDispatch(enabled:bool){let yes=select(0u,1u,enabled);dispatchArgs[4]=vec4u(yes*((params.counts.x+127u)/128u),1u,1u,0u);dispatchArgs[5]=vec4u(yes*((params.counts.y+127u)/128u),1u,1u,0u);dispatchArgs[6]=vec4u(yes*params.counts.w,1u,1u,0u);dispatchArgs[7]=vec4u(yes,1u,1u,0u);}
fn grabStiffness(n:u32)->f32 {let picked=params.grabMeta.x;if(picked==0xffffffffu||n>=params.counts.z||nodes[n].attachment.w>.5){return 0.;}if(nodes[n].attachment.y!=nodes[picked].attachment.y){return 0.;}let delta=nodes[n].rest.xyz-nodes[picked].rest.xyz;let d2=dot(delta,delta);if(d2>9.*.006*.006){return 0.;}return params.grab.w*nodes[n].attachment.x*exp(-d2/(2.*.006*.006))*bitcast<f32>(params.grabMeta.w);}
fn mass(n:u32)->f32{return nodes[n].rest.w+params.settings.z*nodes[n].normal.w;}
fn inertialWeight(n:u32)->f32 {if(params.integration.x!=0u){return 0.;}return mass(n)/(params.settings.x*params.settings.x);}
fn coefficients(t:u32)->vec4f {return tets[t].growth;}
@compute @workgroup_size(128) fn updateMaterials(@builtin(global_invocation_id) id:vec3u){let t=id.x;if(t>=params.counts.y){return;}let m=tets[t].material;if(m.w<0.){return;}let ratio=1.+params.settings.z*m.w;let grow=pow(ratio,1./3.);let softness=params.settings.w;let mu=m.y*softness;let lam=m.z-m.y+mu;tets[t].growth=vec4f(mu*grow,lam/ratio,ratio,grow);}

fn contactNormal(n:u32)->vec3f {return nodes[n].normal.xyz;}

fn contactGap(n:u32,p:vec3f)->f32 {
 let gap=bitcast<f32>(nodes[n].incidence.z);
 return dot(p,nodes[n].normal.xyz)+gap-.00001;
}
fn precondition(n:u32,r:vec3f)->vec3f {
 if(params.grabMeta.y==0u){return r/nodes[n].diagonal.xyz;}
 let d=nodes[n].diagonal.xyz;let xy=nodes[n].diagonal.w;let xz=nodes[n].gradient.w;let yz=nodes[n].predict.w;
 let l0=sqrt(d.x);let l1=xy/l0;let l2=xz/l0;let l3=sqrt(max(d.y-l1*l1,d.y*1e-7));let l4=(yz-l1*l2)/l3;let l5=sqrt(max(d.z-l2*l2-l4*l4,d.z*1e-7));
 let y0=r.x/l0;let y1=(r.y-l1*y0)/l3;let y2=(r.z-l2*y0-l4*y1)/l5;let z2=y2/l5;let z1=(y1-l4*z2)/l3;let z0=(y0-l1*z1-l2*z2)/l0;return vec3f(z0,z1,z2);
}
fn externalEnergy(n:u32,p:vec3f)->f32 {let inertial=inertialWeight(n);let u=p-nodes[n].predict.xyz;var e=.5*inertial*dot(u,u)+mass(n)*9.81*params.settings.y*p.y;if(nodes[n].attachment.w<.5){let gap=min(0.,contactGap(n,p));e+=10000.*gap*gap;let spring=grabStiffness(n);if(spring>0.){let v=p-(params.grab.xyz-nodes[params.grabMeta.x].rest.xyz);e+=.5*spring*dot(v,v);}}return e;}
// Use one tetrahedral origin; rigid nodal translation must cancel before
// multiplying by large gradients in thin cells, rather than after summation.
fn tetGradient(t:u32,a:u32)->vec3f {if(a==0u){return -(tets[t].grad[1].xyz+tets[t].grad[2].xyz+tets[t].grad[3].xyz);}return tets[t].grad[a].xyz;}
fn matrix(t:u32,trial:bool)->mat3x3f {var f=mat3x3f(0.,0.,0.,0.,0.,0.,0.,0.,0.);let origin=tets[t].ids[0];let anchor=select(nodes[origin].x.xyz,nodes[origin].trial.xyz,trial);for(var a=1u;a<4u;a++){let n=tets[t].ids[a];let u=select(nodes[n].x.xyz,nodes[n].trial.xyz,trial)-anchor;let grad=tets[t].grad[a].xyz;f[0]+=u*grad.x;f[1]+=u*grad.y;f[2]+=u*grad.z;}return f;}
fn identity()->mat3x3f {return mat3x3f(1.,0.,0.,0.,1.,0.,0.,0.,1.);}
fn trace(u:mat3x3f)->f32 {return u[0].x+u[1].y+u[2].z;}
fn minorSum(u:mat3x3f)->f32 {return u[0].x*u[1].y-u[0].y*u[1].x+u[0].x*u[2].z-u[0].z*u[2].x+u[1].y*u[2].z-u[1].z*u[2].y;}
fn cofU(u:mat3x3f)->mat3x3f {return mat3x3f(cross(u[1],u[2]),cross(u[2],u[0]),cross(u[0],u[1]));}
// Numerical orientation barrier: not a calibrated tissue constitutive term.
fn compressionBarrier(j:f32)->vec3f {if(j>=.6){return vec3f(0.);}if(j<=.2){return vec3f(1e30,0.,0.);}let d=j-.2;let q=j-.6;let l=log(d/.4);return vec3f(-q*q*l,-2.*q*l-q*q/d,-2.*l-4.*q/d+q*q/(d*d));}
fn stableLog1p(x:f32)->f32 {if(abs(x)<.01){return x*(1.+x*(-.5+x*(1./3.+x*(-.25+x*.2))));}return log(1.+x);}
fn compressionDifference(j:f32,dj:f32)->f32 {let b=j+dj;if(b<=.2){return 1e30;}if(j>=.6||b>=.6){return compressionBarrier(b).x-compressionBarrier(j).x;}let q=j-.6;let d=j-.2;return -(2.*q*dj+dj*dj)*log(d/.4)-(q+dj)*(q+dj)*stableLog1p(dj/d);}
fn energy(t:u32,u:mat3x3f,c:vec4f)->vec2f {let tr=trace(u);let minors=minorSum(u)+determinant(u);let delta=tr+minors-(c.z-1.);let square=dot(u[0],u[0])+dot(u[1],u[1])+dot(u[2],u[2]);let constant=1.5*c.x*(1.-c.w*c.w)+c.x/c.w*(c.z-1.);let e=constant+c.x*((1.-1./c.w)*tr+.5*square-minors/c.w)+.5*c.y*delta*delta+tets[t].material.y*compressionBarrier(1.+tr+minors).x;return vec2f(tets[t].material.x*e,1.+tr+minors);}
// Algebraically factored E(u+du)-E(u), without growth constants.
fn matrixIncrement(t:u32)->mat3x3f {
 var d=mat3x3f(0.,0.,0.,0.,0.,0.,0.,0.,0.);let origin=tets[t].ids[0];let anchor=nodes[origin].trial.xyz-nodes[origin].x.xyz;
 for(var a=1u;a<4u;a++){let n=tets[t].ids[a];let du=(nodes[n].trial.xyz-nodes[n].x.xyz)-anchor;let g=tets[t].grad[a].xyz;d[0]+=du*g.x;d[1]+=du*g.y;d[2]+=du*g.z;}return d;
}
fn volumeEnergyDifference(t:u32,a:mat3x3f,d:mat3x3f,c:vec4f)->f32 {
 let b=a+d;let tr=trace(d);let sum=a+b;
 let quadratic=.5*(tr*trace(sum)-trace(d*sum));
 let cubic=dot(d[0],cross(a[1],a[2]))+dot(b[0],cross(d[1],a[2]))+dot(b[0],cross(b[1],d[2]));
 let minors=quadratic+cubic;let dj=tr+minors;
 let ja=trace(a)+minorSum(a)+determinant(a)-(c.z-1.);
 let square=dot(d[0],sum[0])+dot(d[1],sum[1])+dot(d[2],sum[2]);
 return tets[t].material.x*(c.x*((1.-1./c.w)*tr+.5*square-minors/c.w)+.5*c.y*dj*(2.*ja+dj)+tets[t].material.y*compressionDifference(ja+c.z,dj));
}
fn externalEnergyDifference(n:u32,a:vec3f,b:vec3f)->f32 {
 let d=b-a;let inertial=inertialWeight(n);
 var e=.5*inertial*dot(d,a+b-2.*nodes[n].predict.xyz)+mass(n)*9.81*params.settings.y*d.y;
 if(nodes[n].attachment.w<.5){let ga=min(0.,contactGap(n,a));let gb=min(0.,contactGap(n,b));e+=10000.*(gb-ga)*(gb+ga);let k=grabStiffness(n);if(k>0.){let springGoal=params.grab.xyz-nodes[params.grabMeta.x].rest.xyz;e+=.5*k*dot(d,a+b-2.*springGoal);}}
 return e;
}
fn reduce(v:vec4f,l:u32)->vec4f {sums[l]=v;workgroupBarrier();var width=64u;loop{if(l<width){let a=sums[l];let b=sums[l+width];sums[l]=vec4f(a.x+b.x,min(a.y,b.y),a.z+b.z,a.w+b.w);}workgroupBarrier();if(width==1u){break;}width/=2u;}return sums[0];}
@compute @workgroup_size(128) fn predict(@builtin(global_invocation_id) id:vec3u){let n=id.x;if(n>=params.counts.x){return;}if(n==0u){solver.values[11]=0.;solver.values[12]=1.;newtonDispatch(true);}nodes[n].old=nodes[n].x;nodes[n].predict=vec4f(select(nodes[n].x.xyz+params.settings.x*nodes[n].velocity.xyz,nodes[n].x.xyz,params.integration.x!=0u),0.);if(nodes[n].attachment.w>.5){nodes[n].x=vec4f(nodes[n].attachment.xyz-nodes[n].rest.xyz,0.);}nodes[n].direction=vec4f(0.);}

struct BendState {q:array<vec3f,4>,angle:f32,valid:f32}
fn bendingState(t:u32,trial:bool)->BendState {
 var p:array<vec3f,4>;let origin=tets[t].ids[0];let offset=select(nodes[origin].x.xyz,nodes[origin].trial.xyz,trial);for(var a=0u;a<4u;a++){let n=tets[t].ids[a];p[a]=(nodes[n].rest.xyz-nodes[origin].rest.xyz)+(select(nodes[n].x.xyz,nodes[n].trial.xyz,trial)-offset);}
 let e=p[1]-p[0];let e2=dot(e,e);let length=sqrt(e2);let n1=cross(e,p[2]-p[0]);let n2=cross(p[3]-p[0],e);let s1=dot(n1,n1);let s2=dot(n2,n2);var state:BendState;
 if(e2<1e-20||min(s1,s2)<1e-24){state.valid=0.;return state;}
 state.valid=1.;state.angle=atan2(dot(cross(n1,n2),e)/length,dot(n1,n2));state.q[2]=-length*n1/s1;state.q[3]=-length*n2/s2;
 state.q[0]=dot(p[2]-p[1],e)/e2*state.q[2]+dot(p[3]-p[1],e)/e2*state.q[3];state.q[1]=-state.q[0]-state.q[2]-state.q[3];return state;
}
// Signed hinge-angle changes evaluated from factored normal differences.
// Avoid adding nanometre motion to head-sized positions or subtracting atan2 angles.
fn hingeAngleChange(p:array<vec3f,4>,d:array<vec3f,4>)->f32 {
 let e=p[1]-p[0];let de=d[1]-d[0];let q2=p[2]-p[0];let dq2=d[2]-d[0];let q3=p[3]-p[0];let dq3=d[3]-d[0];
 let n1=cross(e,q2);let n2=cross(q3,e);
 let dn1=cross(de,q2)+cross(e,dq2)+cross(de,dq2);let dn2=cross(dq3,e)+cross(q3,de)+cross(dq3,de);
 let edgeLength=length(e);let nextLength=length(e+de);
 let dl=(2.*dot(e,de)+dot(de,de))/max(edgeLength+nextLength,1e-20);
 let cross0=cross(n1,n2);let dcross=cross(dn1,n2)+cross(n1,dn2)+cross(dn1,dn2);
 let sa=dot(cross0,e)/max(edgeLength,1e-20);let ca=dot(n1,n2);
 let ds=(dot(dcross,e)+dot(cross0+dcross,de)-sa*dl)/max(nextLength,1e-20);
 let dc=dot(dn1,n2)+dot(n1,dn2)+dot(dn1,dn2);
 let scale=max(max(abs(sa),abs(ca)),1e-30);
 let a=sa/scale;let b=ca/scale;let da=ds/scale;let db=dc/scale;
 return atan2(da*b-db*a,a*a+b*b+da*a+db*b);
}
fn bendingStrain(t:u32,trial:bool)->f32 {
 var p:array<vec3f,4>;var d:array<vec3f,4>;let origin=tets[t].ids[0];let offset=select(nodes[origin].x.xyz,nodes[origin].trial.xyz,trial);
 for(var a=0u;a<4u;a++){let n=tets[t].ids[a];p[a]=nodes[n].rest.xyz-nodes[origin].rest.xyz;d[a]=select(nodes[n].x.xyz,nodes[n].trial.xyz,trial)-offset;}
 return hingeAngleChange(p,d);
}
fn bendingIncrement(t:u32)->f32 {
 var p:array<vec3f,4>;var d:array<vec3f,4>;let origin=tets[t].ids[0];let dx=nodes[origin].trial.xyz-nodes[origin].x.xyz;
 for(var a=0u;a<4u;a++){let n=tets[t].ids[a];p[a]=(nodes[n].rest.xyz-nodes[origin].rest.xyz)+(nodes[n].x.xyz-nodes[origin].x.xyz);d[a]=(nodes[n].trial.xyz-nodes[n].x.xyz)-dx;}
 return hingeAngleChange(p,d);
}
fn bendingEnergy(t:u32,trial:bool)->vec2f {let b=bendingState(t,trial);let delta=bendingStrain(t,trial);return vec2f(select(1e10,.5*tets[t].material.z*delta*delta,b.valid>.5),select(-1.,1.,b.valid>.5));}

@compute @workgroup_size(128) fn evaluateTets(@builtin(global_invocation_id) id:vec3u){let t=id.x;if(t>=params.counts.y){return;}if(tets[t].material.w<0.){let b=bendingState(t,false);let delta=bendingStrain(t,false);let k=tets[t].material.z;for(var a=0u;a<4u;a++){tets[t].hydro[a]=vec4f(b.q[a],0.);tets[t].force[a]=vec4f(k*delta*b.q[a],0.);}tets[t].metric=vec4f(bendingEnergy(t,false),0.,k);return;}let u=matrix(t,false);let f=identity()+u;let linearC=trace(u)*identity()-transpose(u);let quadraticC=cofU(u);let cof=identity()+linearC+quadraticC;let c=coefficients(t);let delta=trace(u)+minorSum(u)+determinant(u)-(c.z-1.);let barrier=compressionBarrier(delta+c.z);let kBarrier=tets[t].material.y;let stress=c.x*((1.-1./c.w)*identity()+u-(1./c.w)*(linearC+quadraticC))+(c.y*delta+kBarrier*barrier.y)*cof;let V=tets[t].material.x;for(var a=0u;a<4u;a++){tets[t].force[a]=vec4f(V*(stress*tetGradient(t,a)),0.);tets[t].hydro[a]=vec4f(cof*tetGradient(t,a),0.);}for(var k=0u;k<3u;k++){tets[t].F[k]=vec4f(f[k],0.);}let e=energy(t,u,c);tets[t].metric=vec4f(e,c.x,c.y+kBarrier*barrier.z);}
@compute @workgroup_size(128) fn gatherGradient(@builtin(global_invocation_id) id:vec3u){let n=id.x;if(n>=params.counts.x){return;}if(nodes[n].attachment.w>.5){nodes[n].gradient=vec4f(0.);nodes[n].diagonal=vec4f(1.);return;}let inertial=inertialWeight(n);var g=inertial*(nodes[n].x.xyz-nodes[n].predict.xyz)+vec3f(0.,mass(n)*9.81*params.settings.y,0.);var diagonal=vec3f(inertial);var off=vec3f(0.);for(var i=nodes[n].incidence.x;i<nodes[n].incidence.y;i++){let code=adjacency[i];let t=code/4u;let a=code%4u;g+=tets[t].force[a].xyz;let grad=tetGradient(t,a);let h=tets[t].hydro[a].xyz;let m=tets[t].metric;diagonal+=tets[t].material.x*(vec3f(m.z*dot(grad,grad))+m.w*h*h);off+=tets[t].material.x*m.w*vec3f(h.x*h.y,h.x*h.z,h.y*h.z);}let gap=contactGap(n,nodes[n].x.xyz);nodes[n].velocity.w=gap;if(gap<0.){let normal=contactNormal(n);g+=20000.*gap*normal;diagonal+=20000.*normal*normal;off+=20000.*vec3f(normal.x*normal.y,normal.x*normal.z,normal.y*normal.z);}let spring=grabStiffness(n);if(spring>0.){g+=spring*(nodes[n].x.xyz-(params.grab.xyz-nodes[params.grabMeta.x].rest.xyz));diagonal+=vec3f(spring);}nodes[n].gradient=vec4f(g,off.y);nodes[n].diagonal=vec4f(max(diagonal,vec3f(1e-12)),off.x);nodes[n].predict.w=off.z;}
@compute @workgroup_size(128) fn cgInit(@builtin(global_invocation_id) id:vec3u){let n=id.x;if(n>=params.counts.x){return;}let r=-nodes[n].gradient.xyz;let z=precondition(n,r);nodes[n].r=vec4f(r,0.);nodes[n].z=vec4f(z,0.);nodes[n].p=vec4f(z,0.);nodes[n].direction=vec4f(0.);}
@compute @workgroup_size(128) fn rzPartial(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(workgroup_id) group:vec3u){let n=id.x;var v=vec4f(0.,1e30,0.,0.);if(n<params.counts.x){v.x=dot(nodes[n].r.xyz,nodes[n].z.xyz);v.z=dot(nodes[n].r.xyz,nodes[n].r.xyz);v.w=dot(nodes[n].gradient.xyz,nodes[n].gradient.xyz);}let r=reduce(v,lid.x);if(lid.x==0u){partial[group.x]=r;}}
fn sumPartial(l:u32,limit:u32)->vec4f {var v=vec4f(0.,1e30,0.,0.);for(var i=l;i<limit;i+=128u){let p=partial[i];v.x+=p.x;v.y=min(v.y,p.y);v.z+=p.z;v.w+=p.w;}return reduce(v,l);}
// Independent static-force and all-layer motion checks at the committed state.
@compute @workgroup_size(128) fn validationPartial(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(workgroup_id) group:vec3u){
 let n=id.x;var v=vec4f(0.,0.,0.,0.);
 if(n<params.counts.x&&nodes[n].attachment.w<.5){
  let inertia=inertialWeight(n)*(nodes[n].x.xyz-nodes[n].predict.xyz);
  let force=nodes[n].gradient.xyz-inertia;let speed2=dot(nodes[n].velocity.xyz,nodes[n].velocity.xyz);
  v=vec4f(dot(force,force),-sqrt(speed2),.5*mass(n)*speed2,0.);
 }
 let r=reduce(v,lid.x);if(lid.x==0u){partial[group.x]=r;}
}
@compute @workgroup_size(128) fn validationFinish(@builtin(local_invocation_id) lid:vec3u){
 let v=sumPartial(lid.x,(params.counts.x+127u)/128u);
 if(lid.x==0u){solver.values[13]=sqrt(v.x);solver.values[14]=-v.y;solver.values[15]=v.z;}
}
@compute @workgroup_size(128) fn cgStart(@builtin(local_invocation_id) lid:vec3u){let v=sumPartial(lid.x,(params.counts.x+127u)/128u);if(lid.x==0u){solver.values[0]=v.x;solver.values[3]=select(1.,0.,v.z<1e-8);solver.values[4]=v.w;solver.values[10]=sqrt(v.w);cgDispatch(solver.values[3]>.5);solver.values[12]=solver.values[3];newtonDispatch(solver.values[12]>.5);if(solver.values[12]<.5){lineDispatch(false);}}}

@compute @workgroup_size(128) fn cgAlpha(@builtin(local_invocation_id) lid:vec3u){let v=sumPartial(lid.x,(params.counts.x+127u)/128u);if(lid.x==0u){solver.values[1]=select(0.,solver.values[0]/max(v.x,1e-30),solver.values[3]>.5&&v.x>1e-30);}}

@compute @workgroup_size(128) fn cgBeta(@builtin(local_invocation_id) lid:vec3u){if(lid.x==0u){controlEnabled=select(0u,1u,solver.values[3]>.5);}if(workgroupUniformLoad(&controlEnabled)==0u){return;}let v=sumPartial(lid.x,(params.counts.x+127u)/128u);if(lid.x==0u){solver.values[2]=v.x/max(solver.values[0],1e-30);solver.values[0]=v.x;if(v.z<=solver.values[4]*.0001){solver.values[3]=0.;}solver.values[10]=sqrt(v.z);solver.values[11]+=1.;cgDispatch(solver.values[3]>.5);}}
@compute @workgroup_size(128) fn cgDirection(@builtin(global_invocation_id) id:vec3u){let n=id.x;if(n>=params.counts.x){return;}nodes[n].p=select(vec4f(0.),nodes[n].z+solver.values[2]*nodes[n].p,solver.values[3]>.5);}
fn energyPart(n:u32,trial:bool)->vec4f {var v=vec4f(0.,1e30,0.,0.);if(n<params.counts.y){v.x=tets[n].metric.x;v.y=tets[n].metric.y;}if(n<params.counts.x){v.x+=select(externalEnergy(n,nodes[n].x.xyz),externalEnergyDifference(n,nodes[n].x.xyz,nodes[n].trial.xyz),trial);v.z=dot(nodes[n].gradient.xyz,nodes[n].direction.xyz);}return v;}
@compute @workgroup_size(128) fn currentPartial(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(workgroup_id) group:vec3u){let r=reduce(energyPart(id.x,false),lid.x);if(lid.x==0u){partial[group.x]=r;}}
@compute @workgroup_size(128) fn currentEnergy(@builtin(local_invocation_id) lid:vec3u){if(lid.x==0u){controlEnabled=select(0u,1u,solver.values[12]>.5);}if(workgroupUniformLoad(&controlEnabled)==0u){return;}let v=sumPartial(lid.x,params.counts.w);if(lid.x==0u){solver.values[5]=v.x;solver.values[6]=v.z;solver.values[7]=1.;solver.values[8]=0.;solver.values[9]=v.y;lineDispatch(true);}}
@compute @workgroup_size(128) fn trialNodes(@builtin(global_invocation_id) id:vec3u){let n=id.x;if(n>=params.counts.x||solver.values[8]>.5){return;}nodes[n].trial=nodes[n].x+solver.values[7]*nodes[n].direction;}
@compute @workgroup_size(128) fn trialTets(@builtin(global_invocation_id) id:vec3u){let t=id.x;if(t>=params.counts.y||solver.values[8]>.5){return;}var e:vec2f;if(tets[t].material.w<0.){let a=bendingState(t,false);let b=bendingState(t,true);let da=bendingStrain(t,false);let change=bendingIncrement(t);e=vec2f(.5*tets[t].material.z*change*(2.*da+change),select(-1.,1.,b.valid>.5));}else{let a=matrix(t,false);let b=matrix(t,true);let c=coefficients(t);e=vec2f(volumeEnergyDifference(t,a,matrixIncrement(t),c),energy(t,b,c).y);}tets[t].metric.x=e.x;tets[t].metric.y=e.y;}
@compute @workgroup_size(128) fn trialPartial(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(workgroup_id) group:vec3u){let r=reduce(energyPart(id.x,true),lid.x);if(lid.x==0u){partial[group.x]=r;}}
@compute @workgroup_size(128) fn acceptTrial(@builtin(local_invocation_id) lid:vec3u){if(lid.x==0u){controlEnabled=select(0u,1u,solver.values[8]<.5&&solver.values[12]>.5);}if(workgroupUniformLoad(&controlEnabled)==0u){return;}let v=sumPartial(lid.x,params.counts.w);if(lid.x==0u&&solver.values[8]<.5){if(v.y>.2&&v.x<=.0001*solver.values[7]*solver.values[6]){solver.values[8]=1.;solver.values[9]=v.y;}else{solver.values[7]*=.5;}lineDispatch(solver.values[8]<.5);}}
@compute @workgroup_size(128) fn commit(@builtin(global_invocation_id) id:vec3u){let n=id.x;if(n>=params.counts.x){return;}if(solver.values[8]>.5){nodes[n].x=nodes[n].trial;}}
@compute @workgroup_size(128) fn finish(@builtin(global_invocation_id) id:vec3u){let n=id.x;if(n>=params.counts.x){return;}nodes[n].velocity=vec4f(select(exp(log(.98)*60.*params.settings.x)*(nodes[n].x.xyz-nodes[n].old.xyz)/params.settings.x,vec3f(0.),params.integration.x!=0u),0.);}
@compute @workgroup_size(128) fn pack(@builtin(global_invocation_id) id:vec3u){let n=id.x;if(n>=params.counts.x){return;}output[n]=vec4f(nodes[n].rest.xyz+nodes[n].x.xyz,0.);if(n>=params.counts.z){return;}var f0=vec3f(0.);var f1=vec3f(0.);var f2=vec3f(0.);var weight=0.;for(var i=nodes[n].incidence.x;i<nodes[n].incidence.y;i++){let t=adjacency[i]/4u;if(tets[t].material.w<0.){continue;}let V=tets[t].material.x;f0+=V*tets[t].F[0].xyz;f1+=V*tets[t].F[1].xyz;f2+=V*tets[t].F[2].xyz;weight+=V;}let offset=params.counts.x;output[offset+n]=vec4f(f0/weight,0.);output[offset+params.counts.z+n]=vec4f(f1/weight,0.);output[offset+2u*params.counts.z+n]=vec4f(f2/weight,0.);}

@compute @workgroup_size(128) fn assembleBlocks(@builtin(global_invocation_id) id:vec3u){let e=id.x;if(e>=params.grabMeta.z||solver.values[3]<.5){return;}let edgeInfo=edges[e].edgeInfo;var h=mat3x3f(vec3f(0.),vec3f(0.),vec3f(0.));for(var i=edgeInfo.y;i<edgeInfo.z;i++){let code=edgeCodes[i];let t=code/16u;let a=(code%16u)/4u;let b=code%4u;let ga=tetGradient(t,a);let gb=tetGradient(t,b);let ha=tets[t].hydro[a].xyz;let hb=tets[t].hydro[b].xyz;let m=tets[t].metric;h+=tets[t].material.x*(m.z*dot(ga,gb)*identity()+m.w*mat3x3f(ha*hb.x,ha*hb.y,ha*hb.z));}edges[e].c0=vec4f(h[0],0.);edges[e].c1=vec4f(h[1],0.);edges[e].c2=vec4f(h[2],0.);}

// Optional synchronous bridge captures the actual final substep velocity at
// the same command/version as positions; .w contact scratch is excluded.
@compute @workgroup_size(128) fn packCheckpointVelocity(@builtin(global_invocation_id) id:vec3u){let n=id.x;if(n>=params.counts.x){return;}output[6u*params.counts.z+n]=vec4f(nodes[n].velocity.xyz,0.);}
fn sparseProduct(n:u32)->vec3f {if(nodes[n].attachment.w>.5){return vec3f(0.);}var ap=(vec3f(inertialWeight(n))+1e-5*nodes[n].diagonal.xyz)*nodes[n].p.xyz;let end=select(params.grabMeta.z,nodes[min(n+1u,params.counts.x-1u)].incidence.w,n+1u<params.counts.x);for(var e=nodes[n].incidence.w;e<end;e++){let p=nodes[edges[e].edgeInfo.x].p.xyz;ap+=edges[e].c0.xyz*p.x+edges[e].c1.xyz*p.y+edges[e].c2.xyz*p.z;}if(nodes[n].velocity.w<0.){let normal=contactNormal(n);ap+=20000.*dot(normal,nodes[n].p.xyz)*normal;}ap+=(grabStiffness(n))*nodes[n].p.xyz;return ap;}
@compute @workgroup_size(128) fn multiplyReduce(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(workgroup_id) group:vec3u){let n=id.x;var v=vec4f(0.,1e30,0.,0.);if(n<params.counts.x){let ap=sparseProduct(n);nodes[n].ap=vec4f(ap,0.);v.x=dot(nodes[n].p.xyz,ap);}let r=reduce(v,lid.x);if(lid.x==0u){partial[group.x]=r;}}
@compute @workgroup_size(128) fn updateReduce(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(workgroup_id) group:vec3u){let n=id.x;var v=vec4f(0.,1e30,0.,0.);if(n<params.counts.x){nodes[n].direction+=solver.values[1]*nodes[n].p;nodes[n].r-=solver.values[1]*nodes[n].ap;let z=precondition(n,nodes[n].r.xyz);nodes[n].z=vec4f(z,0.);v.x=dot(nodes[n].r.xyz,z);v.z=dot(nodes[n].r.xyz,nodes[n].r.xyz);v.w=dot(nodes[n].gradient.xyz,nodes[n].gradient.xyz);}let r=reduce(v,lid.x);if(lid.x==0u){partial[group.x]=r;}}
