"""Append the verified native cavity annulus to the frozen bilateral case.

Preserves all existing tissue coordinates, cells and material records. The old
deep mucosal cut is replaced by the new exact row-24 cut; other omitted-tissue
constraints remain. This is input construction, not a solved anatomical claim.
"""
from pathlib import Path
import argparse, copy, hashlib, json, os
import numpy as np
from build_bilateral_mechanics_case import boundary, surface_audit

R = Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_arrays(case, d):
    raw = (case / d['binary']).read_bytes()
    return {k: np.frombuffer(raw, dtype=q['dtype'], count=np.prod(q['shape']),
                            offset=q['byte_offset']).reshape(q['shape']).copy()
            for k, q in d['arrays'].items()}

def write_arrays(out, arrays):
    contract = {}
    with (out / 'arrays.bin').open('xb') as h:
        for k, a in arrays.items():
            if h.tell() % 8: h.write(b'\0' * (8 - h.tell() % 8))
            q = np.ascontiguousarray(a)
            contract[k] = dict(byte_offset=h.tell(), dtype=q.dtype.str,
                               shape=list(q.shape), bytes=q.nbytes)
            h.write(q.tobytes())
    return contract

def build(base, mesh, binding, out, passive_families):
    base, mesh, binding, out = (p.resolve() for p in (base, mesh, binding, out))
    d = json.loads((base / 'domain.json').read_text())
    old = read_arrays(base, d)
    a = np.load(mesh); b = np.load(binding)
    v = a['vertices']; t = a['tetrahedra']; nn = len(old['rest']); nt = len(old['nodes'])
    if not np.array_equal(v[:nn] * .001, old['rest']):
        raise ValueError('Old coordinates changed')
    if not np.array_equal(t[:nt], old['nodes']):
        raise ValueError('Old cells changed')
    if str(b['source_geometry_sha256']) != sha(mesh): raise ValueError('Binding mesh mismatch')
    if len(v) != int(b['required_gradient_recovery_count']): raise ValueError('Binding node count mismatch')
    x = v * .001; newt = t[nt:]
    dm = np.stack([x[newt[:, j]] - x[newt[:, 0]] for j in (1, 2, 3)], -1)
    det = np.linalg.det(dm)
    if det.min() <= 0: raise ValueError('Nonpositive appended cell')
    g = np.zeros((len(newt), 4, 3)); g[:, 1:] = np.linalg.inv(dm); g[:, 0] = -g[:, 1:].sum(1)
    vol = det / 6; young = 3200.; nu = .43; mu = young / (2 * (1 + nu))
    lam = young * nu / ((1 + nu) * (1 - 2 * nu)) + mu
    mat = np.r_[old['material'], np.c_[vol, np.full(len(vol), mu), np.full(len(vol), lam)]]
    mass = np.r_[old['mass'], np.zeros(len(v) - nn)]
    for j in range(4): np.add.at(mass, newt[:, j], 1000 * vol / 4)
    arrays = dict(rest=x.astype('<f8'), nodes=t.astype('<u4'),
                  gradients=np.r_[old['gradients'], g].astype('<f8'),
                  material=mat.astype('<f8'), mass=mass.astype('<f8'))
    old_cut = np.unique(a['oral_node_grid'][:, -1, -1])
    new_cut = np.unique(a['cavity_extension_native_cut_nodes'])
    if not np.array_equal(new_cut, np.sort(b['deep_mucosal_cut_support_node_ids'])):
        raise ValueError('New cavity cut differs from exact effective binding supports')
    old_frozen = np.asarray(d['frozen_transfer_gradient_nodes'])
    if not np.isin(old_cut, old_frozen).all(): raise ValueError('Old cut provenance mismatch')
    frozen = np.unique(np.r_[np.setdiff1d(old_frozen, old_cut), new_cut])
    fixed_star = np.unique(t[np.isin(t, frozen).any(1)])
    origin_pins = np.unique(np.concatenate([q['nodes'] for q in d['bone_origin_supports']]))
    fixed = np.unique(np.r_[fixed_star, origin_pins])
    if not np.isin(b['cut_recovery_incident_node_ids'], fixed).all():
        raise ValueError('An exact native cut gradient support is not fixed')
    # These old deep faces now have real tissue on both sides. They cannot retain
    # an omitted-tissue support spring just because their node IDs were reused.
    internalized = np.unique(a['cavity_extension_base_node_ids'])
    springs = np.setdiff1d(d['omitted_deep_tissue_spring_nodes'], np.r_[internalized, fixed])
    bf = boundary(t); audit = surface_audit(bf)
    if audit['open_or_nonmanifold_edges'] or audit['inconsistent_winding_edges'] or audit['nonmanifold_vertex_count']:
        raise ValueError('Invalid appended volume boundary')
    muscles = copy.deepcopy(d['muscles'])
    if passive_families:
        p = json.loads(passive_families.read_text())['muscles']
        if any(q.get('activation') != 0 for q in p): raise ValueError('Incoming proposal is not passive')
        if any(max(q['tetrahedra']) >= nt for q in p): raise ValueError('Passive family left its exact preserved cells')
        muscles += p
    occupancy = np.zeros(len(t))
    for q in muscles:
        ids = np.asarray(q['tetrahedra']); w = np.asarray(q.get('volumeFractions', np.full(len(ids), q.get('volumeFraction', 1.))))
        if w.shape != ids.shape or np.any(w <= 0): raise ValueError('Invalid mixture fractions')
        np.add.at(occupancy, ids, w)
    if occupancy.max() > 1 + 1e-12: raise ValueError('Mixture occupancy exceeds one')
    out.mkdir(parents=True, exist_ok=False)
    contract = write_arrays(out, arrays)
    os.link(base / d['bones']['binary'], out / 'bones.bin')
    new = copy.deepcopy(d)
    new.update(source_geometry_sha256=sha(mesh), model=dict(schema='cybr-layered-fem-v1', nodes=len(v), tetrahedra=len(t)),
               arrays=contract, fixed_nodes=fixed.tolist(), frozen_transfer_gradient_nodes=frozen.tolist(),
               omitted_deep_tissue_spring_nodes=springs.tolist(), boundary_triangles=bf.tolist(), muscles=muscles,
               material_family_model='interdigitated' if passive_families else 'single-family',
               native_binding_sha256=sha(binding),
               parent_case_domain_sha256=sha(base / 'domain.json'),
               cavity_continuation=dict(old_cut_native_row=16, new_cut_native_row=24,
                    added_nodes=len(v)-nn, added_tetrahedra=len(t)-nt,
                    shared_attachment_faces=a['cavity_extension_attachment_faces'].tolist(),
                    internalized_old_deep_node_ids=internalized.tolist(),
                    source_binding=str(binding.relative_to(R))))
    new['limits'] += ['Cavity annulus is passive matrix; native row-24 remainder is still an omitted fixed interface',
                      'Original finite-offset origin pins remain until the separate distributed attachment is accepted',
                      'New passive fibre directions and stiffness are authored, uncalibrated; no extra activation']
    (out/'domain.json').write_text(json.dumps(new, separators=(',', ':'))+'\n')
    report = dict(source_domain_sha256=sha(base/'domain.json'), domain_sha256=sha(out/'domain.json'),
                  arrays_sha256=sha(out/'arrays.bin'), bones_sha256=sha(out/'bones.bin'),
                  geometry_sha256=sha(mesh), binding_sha256=sha(binding),
                  original_rest_and_cells_exact=True, original_gradients_and_material_exact=True,
                  nodes=len(v), tetrahedra=len(t), boundary=audit,
                  old_fixed_nodes=len(d['fixed_nodes']), fixed_nodes=len(fixed),
                  released_old_fixed_nodes=np.setdiff1d(d['fixed_nodes'], fixed).tolist(),
                  added_fixed_nodes=np.setdiff1d(fixed, d['fixed_nodes']).tolist(),
                  new_deep_cut_nodes=new_cut.tolist(),
                  removed_internal_deep_springs=np.intersect1d(d['omitted_deep_tissue_spring_nodes'], internalized).tolist(),
                  minimum_new_volume_solver_units_cubed=float(vol.min()),
                  passive_family_count=len(muscles)-len(d['muscles']), maximum_fibre_mixture_fraction=float(occupancy.max()),
                  status='Input construction only; no solved state or attachment acceptance')
    (out/'construction_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['domain_sha256','nodes','tetrahedra','fixed_nodes','passive_family_count']}, indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,default=R/'output/bilateral-mechanics-v1')
    p.add_argument('--mesh',type=Path,required=True); p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--passive-families',type=Path); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); build(a.base,a.mesh,a.binding,a.output,a.passive_families)
