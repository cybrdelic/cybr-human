#!/usr/bin/env python3
"""Original, asset-free CPU clay preview renderer.

Geometry coordinates: X lateral, Y superior, Z anterior; millimetres are
recommended. The front camera looks from +Z. A positive yaw makes the nose point
right on the image. Projection is orthographic, and larger camera-Z wins depth.
Triangle winding must be outward; it is never silently repaired. Smooth normals
are area-weighted unless supplied. No backface culling is performed.

Dependencies: numpy and Pillow. An optional original C++ rasterizer is compiled
with the installed C++ compiler for speed; a numpy implementation is the fallback.
No imported model, mesh package, renderer, texture, or image asset is used.

CLI: python src/clay_render.py input.npz --out output --size 640
API: image = render_mesh(vertices, triangles, material_ids, angle=35)
For matched cameras, pass framing=make_framing(vertices) to every call.
Material IDs: 0 clay, 1 eye white, 2 muted iris, 3 dark aperture, 4 darker clay.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTES = {
    "gray": np.array([[177, 178, 176], [213, 214, 208], [102, 114, 110],
                      [46, 45, 43], [145, 144, 140]], dtype=np.float64) / 255.,
    "warm": np.array([[185, 163, 143], [214, 208, 194], [111, 103, 86],
                      [48, 43, 37], [157, 134, 116]], dtype=np.float64) / 255.,
}
_NATIVE = None
_NATIVE_ATTEMPTED = False


def validate_mesh(vertices, triangles, material_ids=None):
    """Validate and return contiguous vertex, index, and material arrays."""
    v = np.asarray(vertices, dtype=np.float64)
    source_triangles = np.asarray(triangles)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError("vertices must be a nonempty finite (N, 3) array")
    if source_triangles.ndim != 2 or source_triangles.shape[1] != 3 or not len(source_triangles):
        raise ValueError("triangles must be a nonempty (M, 3) integer array")
    if not np.issubdtype(source_triangles.dtype, np.integer):
        raise ValueError("triangle indices must use an integer dtype")
    t = np.ascontiguousarray(source_triangles, dtype=np.int64)
    if t.min() < 0 or t.max() >= len(v):
        raise ValueError("triangle index outside the vertices array")
    if material_ids is None:
        m = np.zeros(len(t), dtype=np.int32)
    else:
        source_materials = np.asarray(material_ids)
        if source_materials.shape != (len(t),) or not np.issubdtype(source_materials.dtype, np.integer):
            raise ValueError("material_ids must be an integer (M,) array")
        if source_materials.min() < 0 or source_materials.max() >= len(PALETTES["gray"]):
            raise ValueError("material_ids must be in the documented range 0 through 4")
        m = np.ascontiguousarray(source_materials, dtype=np.int32)
    return np.ascontiguousarray(v), t, m


def vertex_normals(vertices, triangles):
    """Area-weighted smooth normals; preserves supplied triangle orientation."""
    face = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                    vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    result = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(result, triangles[:, corner], face)
    length = np.linalg.norm(result, axis=1, keepdims=True)
    return np.divide(result, length, out=np.zeros_like(result), where=length > 1e-15)


def _rotation(angle):
    a = math.radians(float(angle))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_framing(vertices, angles=(0, 35, 90), margin=0.08, center=None):
    """One world center and square span shared by all orthographic views.

    margin is a fraction of the output on each edge. The span accounts for every
    listed view, so all views share scale rather than being independently fitted.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError("vertices must be a nonempty finite (N, 3) array")
    if not 0 <= margin < 0.45 or not len(angles):
        raise ValueError("margin must be in [0, .45) and angles must be nonempty")
    center = (v.min(axis=0) + v.max(axis=0)) * .5 if center is None else np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center must contain three finite coordinates")
    half_span = 0.
    for angle in angles:
        camera = (v - center) @ _rotation(angle).T
        half_span = max(half_span, float(np.abs(camera[:, :2]).max()))
    if half_span <= 1e-12:
        raise ValueError("mesh has no visible extent")
    return {"center": center.tolist(), "span": 2 * half_span / (1 - 2 * margin)}


def _native_rasterizer():
    global _NATIVE, _NATIVE_ATTEMPTED
    if _NATIVE_ATTEMPTED:
        return _NATIVE
    _NATIVE_ATTEMPTED = True
    source = Path(__file__).with_name("clay_raster.cpp")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not source.exists() or compiler is None:
        return None
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        directory = Path(tempfile.gettempdir()) / "original_clay_rasterizer"
        directory.mkdir(exist_ok=True)
        library = directory / f"raster_{digest}.so"
        if not library.exists():
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".so", delete=False) as temporary:
                temporary_name = temporary.name
            try:
                subprocess.run([compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
                                str(source), "-o", temporary_name], check=True,
                               capture_output=True, timeout=90)
                Path(temporary_name).replace(library)
            finally:
                Path(temporary_name).unlink(missing_ok=True)
        lib = ctypes.CDLL(str(library))
        func = lib.clay_rasterize
        double_ptr = ctypes.POINTER(ctypes.c_double)
        float_ptr = ctypes.POINTER(ctypes.c_float)
        int_ptr = ctypes.POINTER(ctypes.c_int32)
        func.argtypes = [double_ptr, double_ptr, ctypes.POINTER(ctypes.c_int64), int_ptr,
                         ctypes.c_int64, ctypes.c_int, ctypes.c_int, float_ptr, float_ptr, int_ptr]
        func.restype = None
        _NATIVE = (lib, func)
    except (OSError, subprocess.SubprocessError) as error:
        warnings.warn(f"Native rasterizer unavailable; using numpy fallback: {error}")
    return _NATIVE


def _rasterize_numpy(screen, normals, triangles, materials, depth, normal_buffer, material_buffer):
    """Reference barycentric rasterizer, used if a C++ compiler is unavailable."""
    height, width = depth.shape
    for face, material in zip(triangles, materials):
        a, b, c = screen[face]
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-12:
            continue
        lo = np.maximum(np.floor(np.minimum(np.minimum(a[:2], b[:2]), c[:2])), 0).astype(int)
        hi = np.minimum(np.ceil(np.maximum(np.maximum(a[:2], b[:2]), c[:2])), [width-1, height-1]).astype(int)
        if np.any(lo > hi):
            continue
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        x, y = xx + .5, yy + .5
        wa = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / denominator
        wb = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / denominator
        wc = 1 - wa - wb
        z = wa*a[2] + wb*b[2] + wc*c[2]
        inside = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9) & (z > depth[yy, xx])
        iy, ix = yy[inside], xx[inside]
        depth[iy, ix] = z[inside]
        material_buffer[iy, ix] = material
        normal_buffer[iy, ix] = (wa[inside, None]*normals[face[0]] + wb[inside, None]*normals[face[1]] + wc[inside, None]*normals[face[2]])


def _rasterize(screen, normals, triangles, materials, size, backend="auto"):
    depth = np.full((size, size), -np.inf, dtype=np.float32)
    normal_buffer = np.zeros((size, size, 3), dtype=np.float32)
    material_buffer = np.full((size, size), -1, dtype=np.int32)
    native = _native_rasterizer() if backend != "numpy" else None
    if backend == "native" and native is None:
        raise RuntimeError("Native rasterizer requested, but it could not be loaded")
    if native is None:
        _rasterize_numpy(screen, normals, triangles, materials, depth, normal_buffer, material_buffer)
    else:
        ptr = lambda a, dtype: a.ctypes.data_as(ctypes.POINTER(dtype))
        native[1](ptr(screen, ctypes.c_double), ptr(normals, ctypes.c_double),
                  ptr(triangles, ctypes.c_int64), ptr(materials, ctypes.c_int32),
                  len(triangles), size, size, ptr(depth, ctypes.c_float),
                  ptr(normal_buffer, ctypes.c_float), ptr(material_buffer, ctypes.c_int32))
    lengths = np.linalg.norm(normal_buffer, axis=2, keepdims=True)
    np.divide(normal_buffer, lengths, out=normal_buffer, where=lengths > 1e-12)
    return depth, normal_buffer, material_buffer


def _diagnostic_depth_ao(depth, normals, mask, units_per_pixel, strength=.30):
    """Small tangent-plane depth AO, a diagnostic cue, not physical ray tracing.

    Neighbours occlude only when closer than the current tangent plane. This
    avoids shading a convex sphere like a concavity. Missing/background samples
    do not occlude, and large depth discontinuities are attenuated.
    """
    height, width = depth.shape
    occlusion = np.zeros_like(depth)
    count = np.zeros_like(depth)
    safe_depth = np.where(mask, depth, 0.)
    nz = np.maximum(normals[..., 2], .20)
    dx_depth = -normals[..., 0] / nz * units_per_pixel
    dy_depth = normals[..., 1] / nz * units_per_pixel
    for radius in (max(1, round(width*.004)), max(2, round(width*.010)), max(3, round(width*.020))):
        for dx, dy in ((radius,0), (-radius,0), (0,radius), (0,-radius),
                       (radius,radius), (radius,-radius), (-radius,radius), (-radius,-radius)):
            y0, y1 = max(0,-dy), min(height,height-dy)
            x0, x1 = max(0,-dx), min(width,width-dx)
            center = np.s_[y0:y1,x0:x1]
            neighbour = np.s_[y0+dy:y1+dy,x0+dx:x1+dx]
            valid = mask[center] & mask[neighbour]
            scale = max(units_per_pixel * radius, 1e-10)
            delta = safe_depth[neighbour] - safe_depth[center]
            excess = delta - dx_depth[center]*dx - dy_depth[center]*dy
            amount = np.clip((excess - scale*.08) / (scale*.70), 0, 1)
            amount *= np.clip(1 - np.abs(delta)/(scale*6), 0, 1)
            occlusion[center] += amount * valid
            count[center] += valid
    return 1 - strength * occlusion / np.maximum(count, 1)



def _shadow_visibility(camera_vertices, camera_normals, triangles, materials,
                       depth, normals, mask, span, light, resolution, bias, backend):
    """Geometric light-space z-buffer visibility with receiver-plane 3x3 PCF.

    Each shadow sample compares actual mesh depth from a light camera with the
    visible receiving surface. No image-space cavity mask or painted darkness
    is used. Receiver-plane correction prevents sloped surfaces self-shadowing
    merely because PCF samples fall on neighbouring map pixels.
    """
    up = np.array([0.,1.,0.]) if abs(light[1]) < .95 else np.array([1.,0.,0.])
    right = np.cross(up,light); right /= np.linalg.norm(right)
    light_up = np.cross(light,right)
    basis = np.stack([right,light_up,light])
    light_vertices = camera_vertices @ basis.T
    bounds_min, bounds_max = light_vertices[:,:2].min(axis=0), light_vertices[:,:2].max(axis=0)
    light_center = (bounds_min+bounds_max)*.5
    light_span = max(float((bounds_max-bounds_min).max())*1.04, 1e-8)
    scale = resolution/light_span
    light_screen = np.ascontiguousarray(light_vertices.copy())
    light_screen[:,0] = (light_screen[:,0]-light_center[0])*scale+resolution/2
    light_screen[:,1] = -(light_screen[:,1]-light_center[1])*scale+resolution/2
    light_normals = np.ascontiguousarray(camera_normals @ basis.T)
    shadow_depth, _, _ = _rasterize(light_screen,light_normals,triangles,materials,resolution,backend)
    iy,ix = np.nonzero(mask)
    size = depth.shape[0]
    points = np.column_stack([(ix+.5-size/2)*span/size,
                              -(iy+.5-size/2)*span/size, depth[iy,ix]]) @ basis.T
    qx = (points[:,0]-light_center[0])*scale+resolution/2
    qy = -(points[:,1]-light_center[1])*scale+resolution/2
    receiver_normals = normals[iy,ix] @ basis.T
    denominator = np.maximum(receiver_normals[:,2], .12)
    slope_x = np.clip(-receiver_normals[:,0]/denominator,-5.,5.)
    slope_y = np.clip(-receiver_normals[:,1]/denominator,-5.,5.)
    center_x,center_y = np.floor(qx).astype(int),np.floor(qy).astype(int)
    visibility = np.zeros(len(ix),dtype=np.float32)
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            sx,sy = center_x+dx,center_y+dy
            valid = (sx>=0)&(sx<resolution)&(sy>=0)&(sy<resolution)
            sampled = shadow_depth[np.clip(sy,0,resolution-1),np.clip(sx,0,resolution-1)]
            plane_z = points[:,2] + slope_x*(sx+.5-qx)/scale - slope_y*(sy+.5-qy)/scale
            visibility += ((sampled <= plane_z+bias)|~valid)/9.
    result = np.ones(depth.shape,dtype=np.float32)
    result[iy,ix] = visibility
    return result


def render_mesh(vertices, triangles, material_ids=None, *, angle=0., size=640,
                framing=None, material="gray", palette=None, normals=None,
                background=(239, 238, 234), supersample=2, ssao=True,
                backend="auto", return_buffers=False, shadows=False,
                shadow_resolution=None, shadow_bias=.18):
    """Render a PIL RGB image, optionally with full-resolution diagnostic buffers.

    `return_buffers=True` returns a dict: image, depth, normals, material_ids,
    mask, framing, angle. Buffers are at supersampled resolution; background
    depth is -infinity. `palette` overrides the five linear-independent sRGB
    swatches, using a (5,3) float array in [0,1]. Shared vertex normals smooth
    across material boundaries; duplicate vertices at intended hard boundaries.
    `shadows=True` enables three geometric light-space shadow maps with 3x3 PCF.
    `shadow_bias` is in mesh units (0.18 mm by default for millimetre inputs).
    """
    vertices, triangles, material_ids = validate_mesh(vertices, triangles, material_ids)
    if not isinstance(size, (int, np.integer)) or size < 16 or size > 4096:
        raise ValueError("size must be an integer from 16 through 4096")
    if supersample not in (1, 2, 3, 4) or backend not in ("auto", "native", "numpy"):
        raise ValueError("supersample must be 1 to 4; backend must be auto, native, or numpy")
    if material not in PALETTES:
        raise ValueError("material must be gray or warm")
    if not math.isfinite(float(angle)):
        raise ValueError("angle must be finite")
    framing = make_framing(vertices) if framing is None else framing
    center = np.asarray(framing["center"], dtype=float)
    span = float(framing["span"])
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(span) or span <= 0:
        raise ValueError("framing must contain a finite center and positive span")
    normals = vertex_normals(vertices, triangles) if normals is None else np.asarray(normals, dtype=float)
    if normals.shape != vertices.shape or not np.isfinite(normals).all():
        raise ValueError("normals must be a finite (N,3) array")
    render_size = size * supersample
    shadow_resolution = min(2048, max(256,render_size)) if shadow_resolution is None else shadow_resolution
    if not isinstance(shadow_resolution,(int,np.integer)) or not 32 <= shadow_resolution <= 4096:
        raise ValueError("shadow_resolution must be an integer from 32 through 4096")
    if not np.isfinite(shadow_bias) or shadow_bias < 0:
        raise ValueError("shadow_bias must be finite and nonnegative")
    rotation = _rotation(angle)
    camera_vertices = np.ascontiguousarray((vertices - center) @ rotation.T)
    screen = camera_vertices.copy()
    screen[:, 0] = screen[:, 0] * render_size / span + render_size / 2
    screen[:, 1] = -screen[:, 1] * render_size / span + render_size / 2
    camera_normals = np.ascontiguousarray(normals @ rotation.T)
    depth, normal_buffer, material_buffer = _rasterize(screen, camera_normals, triangles,
                                                      material_ids, render_size, backend)
    mask = material_buffer >= 0
    colors = PALETTES[material] if palette is None else np.asarray(palette, dtype=float)
    if colors.shape != (5,3) or not np.isfinite(colors).all() or colors.min() < 0 or colors.max() > 1:
        raise ValueError("palette must be five sRGB swatches in [0,1]")
    # Three diffuse lobes approximate one broad neutral studio light upper left.
    illumination = np.full(depth.shape, .23 if shadows else .31, dtype=np.float32)
    shadow_average = np.zeros(depth.shape,dtype=np.float32) if shadows else None
    key_visibility = 1.
    lobes = (((-.55,.75,1.), .51 if shadows else .43),
             ((-.85,.25,1.), .18 if shadows else .16), ((.60,.20,1.), .10))
    for light_index,(direction,weight) in enumerate(lobes):
        light = np.asarray(direction, dtype=float)
        light /= np.linalg.norm(light)
        visibility = _shadow_visibility(camera_vertices,camera_normals,triangles,material_ids,
                      depth,normal_buffer,mask,span,light,shadow_resolution,shadow_bias,backend) if shadows else 1.
        illumination += weight * np.maximum(normal_buffer @ light, 0) * visibility
        if shadows:
            shadow_average += weight*visibility/sum(lobe[1] for lobe in lobes)
            if light_index == 0:
                key_visibility = visibility
    illumination += .04 * np.clip(normal_buffer[...,1]*.5+.5, 0, 1)
    if ssao:
        illumination *= _diagnostic_depth_ao(depth, normal_buffer, mask, span/render_size)
    base = colors[np.maximum(material_buffer,0)] ** 2.2
    rgb = base * illumination[...,None]
    light = np.array([-.45,.65,1.]); light /= np.linalg.norm(light)
    half_vector = light + np.array([0,0,1.]); half_vector /= np.linalg.norm(half_vector)
    highlight = .020 * np.maximum(normal_buffer @ half_vector, 0) ** 24 * key_visibility
    rgb += highlight[...,None]
    rgb = np.clip(rgb,0,1) ** (1/2.2)
    background = np.asarray(background,dtype=float)
    if background.shape != (3,) or not np.isfinite(background).all() or background.min() < 0 or background.max() > 255:
        raise ValueError("background must contain three RGB values from 0 to 255")
    rgb[~mask] = background / 255.
    image = Image.fromarray(np.round(rgb*255).astype(np.uint8))
    if supersample != 1:
        image = image.resize((size,size), Image.Resampling.LANCZOS)
    if return_buffers:
        return {"image": image, "depth": depth, "normals": normal_buffer,
                "material_ids": material_buffer, "mask": mask, "framing": framing,
                "angle": float(angle), "shadow_visibility": shadow_average}
    return image


def render_views(vertices, triangles, material_ids=None, *, angles=(0,35,90), **kwargs):
    """Return ordered matched-scale PIL images for the requested camera angles."""
    framing = kwargs.pop("framing", None)
    if framing is None:
        framing = make_framing(vertices, angles=angles)
    return [render_mesh(vertices, triangles, material_ids, angle=a, framing=framing, **kwargs) for a in angles]


def _view_name(angle):
    return {0:"front",35:"threequarter",90:"profile"}.get(float(angle), f"yaw_{float(angle):g}".replace("-","minus_"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="NPZ: vertices, triangles, optional material_ids and normals")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--angles", type=float, nargs="+", default=[0,35,90])
    parser.add_argument("--material", choices=list(PALETTES), default="gray")
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--no-ssao", action="store_true", help="disable diagnostic screen-space depth occlusion")
    parser.add_argument("--backend", choices=["auto","native","numpy"], default="auto")
    parser.add_argument("--shadows", action="store_true", help="enable geometric self-shadow maps for the three broad-light samples")
    parser.add_argument("--shadow-resolution", type=int, help="light-space map width; default follows supersampled image up to 2048")
    parser.add_argument("--shadow-bias", type=float, default=.18, help="shadow depth bias in mesh units, normally mm")
    parser.add_argument("--uniform-clay", action="store_true", help="render every triangle as clay, ignoring per-face material IDs")
    parser.add_argument("--buffers", action="store_true", help="also save full-resolution depth/normal/material diagnostic arrays")
    args = parser.parse_args(argv)
    with np.load(args.input, allow_pickle=False) as mesh:
        vertices = mesh["vertices"]
        triangles = mesh["triangles"]
        material_ids = mesh["material_ids"] if "material_ids" in mesh else None
        normals = mesh["normals"] if "normals" in mesh else None
    if args.uniform_clay:
        material_ids = np.zeros(len(triangles), dtype=np.int32)
    framing = make_framing(vertices, angles=args.angles)
    args.out.mkdir(parents=True, exist_ok=True)
    images, views = [], []
    for angle in args.angles:
        result = render_mesh(vertices,triangles,material_ids, angle=angle,size=args.size,
                             framing=framing, material=args.material,normals=normals,
                             supersample=args.supersample,ssao=not args.no_ssao,
                             backend=args.backend,return_buffers=True,shadows=args.shadows,
                             shadow_resolution=args.shadow_resolution,shadow_bias=args.shadow_bias)
        name = _view_name(angle)
        result["image"].save(args.out / f"{name}.png")
        if args.buffers:
            buffers = {k:result[k] for k in ("depth","normals","material_ids","mask")}
            if args.shadows:
                buffers["shadow_visibility"] = result["shadow_visibility"]
            np.savez_compressed(args.out / f"{name}_buffers.npz", **buffers)
        images.append(result["image"])
        views.append({"name":name,"yaw_degrees":angle,"file":f"{name}.png"})
        print(f"Rendered {name}: {args.out / (name+'.png')}", flush=True)
    sheet = Image.new("RGB",(args.size*len(images),args.size+44),(239,238,234))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for i,(img,view) in enumerate(zip(images,views)):
        sheet.paste(img,(i*args.size,0))
        draw.text((i*args.size+18,args.size+12),f"{view['name'].replace('_',' ').upper()}  /  {view['yaw_degrees']:g} DEG",font=font,fill=(75,75,72))
    sheet.save(args.out / "contact_sheet.png")
    manifest = {"renderer":"original CPU barycentric rasterizer", "projection":"orthographic",
                "coordinates":"X lateral, Y superior, Z anterior; larger camera-Z wins", "framing":framing,
                "size":args.size,"supersample":args.supersample,"material":args.material,
                "lighting":"broad neutral upper-left diffuse, weak fill, small clay highlight",
                "diagnostic_depth_ao":not args.no_ssao,"ray_traced":False,
                "geometric_self_shadows":args.shadows,"shadow_samples":3 if args.shadows else 0,
                "shadow_filter":"3x3 receiver-plane PCF" if args.shadows else None,
                "shadow_bias_mesh_units":args.shadow_bias,"uniform_clay":args.uniform_clay,
                "shadow_map_resolution":args.shadow_resolution or min(2048,max(256,args.size*args.supersample)),
                "imported_models_or_assets":False,"views":views,
                "vertex_count":len(vertices),"triangle_count":len(triangles)}
    (args.out / "render_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
