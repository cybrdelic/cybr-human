"""Source origin clipping for the literal positive-volume voxel continuum.

Only actual occupied source-triangle polygons receive constraints. No void
origin point is snapped, and no new connector volume or exterior target is made.
"""
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import save_npz
from audit_origin_occupancy import VoxelDomain
from clipped_origin_support import OFFSETS, PERMUTATIONS, clip_halfspace, area_centroid, prepare_clipped_support
from source_transfer import load_geometry_with_source_fibres


class VoxelFineDomain(VoxelDomain):
    description = 'literal occupied1-L0 voxel union, each cube split into6 aligned Freudenthal tetrahedra; positive finite volume, not fitted anatomical boundary'

    def __init__(self, path, solid, carrier_spacing=6.):
        super().__init__(path)
        self.solid, self.H = solid, float(carrier_spacing)
        self.carrier_origin = np.floor(self.origin/self.H)*self.H
        self.unit_inverse = np.array([np.linalg.inv(np.stack([v[1]-v[0], v[2]-v[0], v[3]-v[0]], axis=1)) for v in OFFSETS])/self.h
        self.perm_index = {tuple(p): i for i, p in enumerate(PERMUTATIONS)}
        points = solid.X[solid.T].mean(1)/solid.mm_per_L0
        uv = (points-self.carrier_origin)/self.H
        cells = np.floor(uv).astype(int)
        perms = np.argsort(-(uv-cells), axis=1)
        self.parents = {tuple(c)+(self.perm_index[tuple(p)],): i for i, (c, p) in enumerate(zip(cells, perms))}
        if len(self.parents) != len(solid.T):
            raise ValueError('Carrier key mapping is not unique')

    def pieces(self, triangle):
        low = np.maximum(0, np.floor((triangle.min(0)-self.origin)/self.h).astype(int))
        high = np.minimum(self.shape-1, np.floor((triangle.max(0)-self.origin)/self.h).astype(int))
        for cell in itertools.product(*(range(low[k], high[k]+1) for k in range(3))):
            if not self.occ[cell]:
                continue
            base = self.origin+self.h*np.asarray(cell)
            for perm in range(6):
                inv = self.unit_inverse[perm]
                normals = np.r_[inv, -inv.sum(0)[None]]
                offsets = np.r_[-inv@base, 1+inv.sum(0)@base]
                poly = triangle.copy()
                for normal, offset in zip(normals, offsets):
                    poly = clip_halfspace(poly, normal, offset)
                    if len(poly) < 3:
                        break
                area, centroid = area_centroid(poly)
                if area <= 1e-12:
                    continue
                middle = base+self.h*OFFSETS[perm].mean(0)
                uv = (middle-self.carrier_origin)/self.H
                cc = np.floor(uv).astype(int)
                pi = self.perm_index[tuple(np.argsort(-(uv-cc)))]
                par = self.parents[tuple(cc)+(pi,)]
                fid = int(np.ravel_multi_index(cell, self.shape))*6+perm
                yield fid, par, poly, area, centroid


def main():
    here = Path(__file__).resolve().parent
    out = here/'coarse-physical-site-v1/supports'
    if out.exists():
        raise FileExistsError('Preserve occupied source-support audit')
    out.mkdir(parents=True)
    start = time.perf_counter()
    geom = here.parent/'geometry'
    solid = load_geometry_with_source_fibres(geom/'whole-face-continuum-final.npz', geom/'whole-face-muscle-phases-final.npz')
    path = geom/'whole-face-integration-cells.npz'
    domain = VoxelFineDomain(path, solid)
    constraints, arrays, report = prepare_clipped_support(solid, path, physical_domain=domain)
    report['seconds'] = time.perf_counter()-start
    np.savez_compressed(out/'clipped-origin-support.npz', **arrays)
    save_npz(out/'constraint-matrix.npz', constraints.C)
    (out/'clipped-origin-support.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(rank=report['rank_audit'], seconds=report['seconds'], source_areas=[dict(family=r['family'], part=r['muscle_source_part'], area=r['retained_constraint_area_L0_2']) for r in report['source_patch_records']]), indent=2))


if __name__ == '__main__':
    main()
