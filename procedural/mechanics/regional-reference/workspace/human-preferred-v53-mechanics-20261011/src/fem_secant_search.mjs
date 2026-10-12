/** Limited-memory inverse-BFGS acceleration of the existing FEM search metric.
 * Energy, analytic force, Dirichlet constraints, line search and collision
 * admission are unchanged. Secants only select a descent direction.
 */
export function installSecantSearch(s,{history=12}={}){
 const original=s.pcg.bind(s),records=[];let previousX=null,previousG=null;
 const dot=(a,b)=>{let z=0;for(let i=0;i<a.length;i++)z+=a[i]*b[i];return z;};
 s.secantSearchReport={acceptedPairs:0,skippedPairs:0,descentFallbacks:0};
 s.pcg=function(dt,limit,tolerance){
  const g=this.g.slice(),x=this.x.slice();
  if(previousX){const dx=x.map((v,i)=>v-previousX[i]),dy=g.map((v,i)=>v-previousG[i]),sy=dot(dx,dy),ss=dot(dx,dx),yy=dot(dy,dy);
   if(sy>1e-10*Math.sqrt(ss*yy)&&sy>1e-28){records.push({dx,dy,rho:1/sy});if(records.length>history)records.shift();this.secantSearchReport.acceptedPairs++;}
   else this.secantSearchReport.skippedPairs++;
  }
  previousX=x;previousG=g;
  const q=g.slice(),alpha=new Float64Array(records.length);
  for(let j=records.length-1;j>=0;j--){const z=records[j];alpha[j]=z.rho*dot(z.dx,q);for(let i=0;i<q.length;i++)q[i]-=alpha[j]*z.dy[i];}
  this.g.set(q);let iterations=original(dt,limit,tolerance);this.g.set(g);
  const r=this.direction.map(v=>-v);
  for(let j=0;j<records.length;j++){const z=records[j],beta=z.rho*dot(z.dy,r);for(let i=0;i<r.length;i++)r[i]+=z.dx[i]*(alpha[j]-beta);}
  this.direction.set(r.map(v=>-v));
  if(!(dot(this.direction,g)<0)){this.secantSearchReport.descentFallbacks++;records.length=0;iterations+=original(dt,limit,tolerance);}
  return iterations;
 };
 return s;
}
