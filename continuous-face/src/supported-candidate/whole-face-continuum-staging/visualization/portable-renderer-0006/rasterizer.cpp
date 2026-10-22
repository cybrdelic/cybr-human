// Auditable, serial, opaque orthographic z-buffer. No mesh edits, culling,
// decimation, transparency, interpolated normals, or third-party dependencies.
// All finite triangles are submitted. Larger view-z means closer to camera.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" int rasterize(
    const double* p, int64_t nv, const int32_t* faces, int64_t nf,
    const uint8_t* classes, const double* palette, int palette_size,
    int width, int height, double cx, double cy, double pixels_per_unit,
    int lighting, double depth_low, double depth_high, int depth_cue,
    uint8_t* rgb, double* zbuffer, int64_t* owner, int64_t owner_offset,
    int64_t* counts) {
  if (!p || !faces || !classes || !palette || !rgb || !zbuffer || !owner ||
      !counts || nv < 0 || nf < 0 || width < 1 || height < 1 ||
      !std::isfinite(cx) || !std::isfinite(cy) ||
      !std::isfinite(pixels_per_unit) || pixels_per_unit <= 0) return 1;
  for (int64_t i=0; i<nv*3; ++i) if (!std::isfinite(p[i])) return 2;
  for (int64_t i=0; i<nf*3; ++i) if (faces[i]<0 || faces[i]>=nv) return 3;
  for (int64_t i=0; i<nf; ++i) if (classes[i]>=palette_size) return 4;
  for (int i=0; i<5; ++i) counts[i]=0;
  const double ll=std::sqrt(.25*.25+.45*.45+1.0);
  for (int64_t t=0; t<nf; ++t) {
    ++counts[0];
    const double *a=p+3*faces[3*t], *b=p+3*faces[3*t+1], *c=p+3*faces[3*t+2];
    double x0=(a[0]-cx)*pixels_per_unit+width*.5;
    double y0=height*.5-(a[1]-cy)*pixels_per_unit;
    double x1=(b[0]-cx)*pixels_per_unit+width*.5;
    double y1=height*.5-(b[1]-cy)*pixels_per_unit;
    double x2=(c[0]-cx)*pixels_per_unit+width*.5;
    double y2=height*.5-(c[1]-cy)*pixels_per_unit;
    double area=(x1-x0)*(y2-y0)-(y1-y0)*(x2-x0);
    if (!std::isfinite(area)) return 5;
    // Zero projected area has no finite pixel coverage. No winding culling.
    if (area==0) { ++counts[1]; continue; }
    double lo_x=std::min({x0,x1,x2}), hi_x=std::max({x0,x1,x2});
    double lo_y=std::min({y0,y1,y2}), hi_y=std::max({y0,y1,y2});
    if (hi_x<.5 || lo_x>width-.5 || hi_y<.5 || lo_y>height-.5) {
      ++counts[2]; continue;
    }
    int minx=int(std::ceil(std::max(.5,lo_x)-.5));
    int maxx=int(std::floor(std::min(width-.5,hi_x)-.5));
    int miny=int(std::ceil(std::max(.5,lo_y)-.5));
    int maxy=int(std::floor(std::min(height-.5,hi_y)-.5));
    const double ux=b[0]-a[0], uy=b[1]-a[1], uz=b[2]-a[2];
    const double vx=c[0]-a[0], vy=c[1]-a[1], vz=c[2]-a[2];
    const double nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
    const double norm=std::sqrt(nx*nx+ny*ny+nz*nz);
    double shade=lighting ? .60+.40*(norm>0 ? std::max(0.,(-.25*nx+.45*ny+nz)/(norm*ll)) : 0) : 1.;
    if (depth_cue) {
      double d=(a[2]+b[2]+c[2])/3;
      shade *= .40+.60*std::clamp((d-depth_low)/std::max(depth_high-depth_low,1e-12),0.,1.);
    }
    uint8_t color[3];
    for (int k=0;k<3;++k) color[k]=uint8_t(std::lround(std::clamp(palette[3*classes[t]+k]*shade,0.,255.)));
    // Signed barycentrics work for either triangle winding. Orthographic z
    // is affine over each triangle; mean-depth sorting is deliberately absent.
    const double inv=1./area;
    for (int y=miny;y<=maxy;++y) for (int x=minx;x<=maxx;++x) {
      const double px=x+.5, py=y+.5;
      double w0=((x1-px)*(y2-py)-(y1-py)*(x2-px))*inv;
      double w1=((x2-px)*(y0-py)-(y2-py)*(x0-px))*inv;
      double w2=1.-w0-w1;
      if (w0< -1e-12 || w1< -1e-12 || w2< -1e-12) continue;
      ++counts[3];
      double z=w0*a[2]+w1*b[2]+w2*c[2];
      if (!std::isfinite(z)) return 6;
      const int64_t i=int64_t(y)*width+x, id=owner_offset+t;
      // Exact depth ties choose lower global triangle ID deterministically.
      if (z>zbuffer[i] || (z==zbuffer[i] && id<owner[i])) {
        zbuffer[i]=z; owner[i]=id;
        for(int k=0;k<3;++k) rgb[3*i+k]=color[k];
        ++counts[4];
      }
    }
  }
  return 0;
}
