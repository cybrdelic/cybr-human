"""Complete exterior finite-triangle branches of the EXISTING contact energy.

For one exterior site/owner, decreasing quadratic penalty of the minimum finite
distance equals the maximum of its finite-triangle penalties. These are energy
epigraph branches, NOT extra summed triangle forces. Interior signed-gap and
cap-barrier terms are deliberately outside this representation.
"""
from pathlib import Path
import time
import numpy as np
from scipy.sparse import coo_matrix
from coarse_physical_site_contact import sha


def build_contact_branches(guard, q):
    started = time.perf_counter()
    state = guard._state(q)
    if not state['valid'].all() or not np.isfinite(state['energy']):
        raise ValueError('Finite feasible native query required before branch construction')
    native = getattr(guard.rigid, 'native', guard.rigid)
    # Positive winning gap proves this owner is exterior under the unchanged
    # complete per-owner native occupancy query. Another owner may be interior.
    selected = np.flatnonzero((state['force_gap'] > 0) & (state['force_gap'] < guard.h))
    group_sites = state['force_ids'][selected]
    group_owners = state['force_owner'][selected]
    owner_count = len(native.owner_ids)
    group_keys = group_sites*owner_count+np.searchsorted(native.owner_ids, group_owners)
    order = np.argsort(group_keys)
    selected, group_sites, group_owners, group_keys = [x[order] for x in (selected, group_sites, group_owners, group_keys)]
    sites = np.unique(group_sites)
    raw_group, raw_face, raw_distance2, raw_witness = [], [], [], []
    for start in range(0, len(sites), 256):
        site_ids = sites[start:start+256]
        p = state['points'][site_ids]
        pad = 1024*np.finfo(float).eps*(np.linalg.norm(p, axis=1)+np.max(abs(native.vertices))+guard.h+1)
        for face_ids, tree, radius in native.distance_bins:
            choices = tree.query_ball_point(p, radius+guard.h+pad)
            count = np.fromiter(map(len, choices), int, len(p))
            if not count.sum():
                continue
            ii = np.repeat(np.arange(len(p)), count)
            jj = face_ids[np.concatenate(choices).astype(int)]
            for begin in range(0, len(ii), 16384):
                pi, fi = ii[begin:begin+16384], jj[begin:begin+16384]
                key = site_ids[pi]*owner_count+native.owner_slot[fi]
                gi = np.searchsorted(group_keys, key)
                eligible = gi < len(group_keys)
                eligible[eligible] &= group_keys[gi[eligible]] == key[eligible]
                pi, fi, gi = pi[eligible], fi[eligible], gi[eligible]
                if not len(pi):
                    continue
                tri = native.triangles[fi]
                delta = np.maximum(np.maximum(tri.min(1)-p[pi], p[pi]-tri.max(1)), 0.)
                eligible = np.sum(delta**2, axis=1) <= (guard.h+pad[pi])**2
                pi, fi, gi = pi[eligible], fi[eligible], gi[eligible]
                witness, distance2 = native._finite_witness(p[pi], fi)
                distance = np.sqrt(distance2)
                eligible = distance < guard.h
                if np.any(eligible):
                    raw_group.append(gi[eligible]);raw_face.append(fi[eligible])
                    raw_distance2.append(distance2[eligible]);raw_witness.append(witness[eligible])
    if raw_group:
        rg, rf, rd2, rw = map(np.concatenate, (raw_group, raw_face, raw_distance2, raw_witness))
        order = np.lexsort((rf, rd2, rg))
        rg, rf, rd2, rw = [x[order] for x in (rg, rf, rd2, rw)]
        rd = np.sqrt(rd2)
    else:
        rg, rf = np.empty(0, int), np.empty(0, int)
        rd, rw = np.empty(0), np.empty((0, 3))
    branch_sites=[];branch_owners=[];branch_triangles=[];branch_energy=[]
    branch_normal=[];branch_local_gradient=[];branch_witness=[];branch_group=[]
    equivalent_faces=[];base_branch=[];retained_sites=[];retained_owners=[]
    retained_force_rows=[];group_energy=[];exact_duplicates=0
    offsets=np.searchsorted(rg, np.arange(len(group_keys)+1))
    for group in range(len(group_keys)):
        lo, hi = offsets[group:group+2]
        if lo == hi:
            raise AssertionError('Complete finite branch search missed the existing winner')
        site, owner = group_sites[group], group_owners[group]
        p = state['points'][site]
        distance, witness, faces = rd[lo:hi], rw[lo:hi], rf[lo:hi]
        normal = (p-witness)/distance[:, None]
        normal[distance <= 1e-13] = native.face_normals[faces[distance <= 1e-13]]
        energy = .5*guard.k*guard.weights[site]*(guard.h-distance)**2
        local_gradient = -(guard.k*guard.weights[site]*(guard.h-distance))[:, None]*normal
        old = selected[group]
        # Existing native tie-break is minimum distance then minimum native ID.
        if (distance[0] != state['force_gap'][old]
                or not np.array_equal(normal[0], state['normal'][old])):
            raise AssertionError('Exact current winning branch was not reproduced')
        unique=[];known={};sources=[]
        for j in range(len(distance)):
            key=np.asarray([energy[j], *local_gradient[j]], dtype=np.float64).tobytes()
            if key in known:
                sources[known[key]].append(int(faces[j]));exact_duplicates+=1
            else:
                known[key]=len(unique);unique.append(j);sources.append([int(faces[j])])
        # A singleton is already handled by the smooth unchanged owner term.
        if len(unique) < 2:
            continue
        gid=len(base_branch);base_branch.append(len(branch_energy))
        retained_sites.append(site);retained_owners.append(owner)
        retained_force_rows.append(old);group_energy.append(energy[0])
        for j, source_faces in zip(unique, sources):
            branch_sites.append(site);branch_owners.append(owner);branch_triangles.append(faces[j])
            branch_energy.append(energy[j]);branch_normal.append(normal[j]);branch_local_gradient.append(local_gradient[j])
            branch_witness.append(witness[j]);branch_group.append(gid);equivalent_faces.append(source_faces)
        # All omitted farther triangles have zero energy; preserve that branch.
        branch_sites.append(site);branch_owners.append(owner);branch_triangles.append(-1)
        branch_energy.append(0.);branch_normal.append(np.zeros(3));branch_local_gradient.append(np.zeros(3))
        branch_witness.append(p);branch_group.append(gid);equivalent_faces.append([])
    bsites=np.asarray(branch_sites,int);bowners=np.asarray(branch_owners,int)
    local=np.asarray(branch_local_gradient,float).reshape(-1,3)
    normal=np.asarray(branch_normal,float).reshape(-1,3)
    A=guard.C[bsites].tocoo()
    rr=np.repeat(A.row,3);cc=(3*A.col[:,None]+np.arange(3)).ravel()
    shape=(len(bsites),3*len(guard.solid.X))
    gradient=coo_matrix(((A.data[:,None]*local[A.row]).ravel(),(rr,cc)),shape=shape).tocsr()
    normals=coo_matrix(((A.data[:,None]*normal[A.row]).ravel(),(rr,cc)),shape=shape).tocsr()
    bases=np.asarray(base_branch,int)
    chosen=np.asarray(gradient[bases].sum(axis=0)).reshape(guard.solid.X.shape)
    return dict(branch_gradient_N=gradient,branch_energy_N_mm=np.asarray(branch_energy),
        branch_group=np.asarray(branch_group,int),group_base_branch=bases,
        group_site=np.asarray(retained_sites,int),group_owner=np.asarray(retained_owners,int),
        group_energy_N_mm=np.asarray(group_energy),group_original_force_row=np.asarray(retained_force_rows,int),
        branch_site=bsites,branch_owner=bowners,branch_triangle=np.asarray(branch_triangles,int),
        branch_closest_point_mm=np.asarray(branch_witness,float).reshape(-1,3),branch_normal=normal,
        branch_equivalent_native_faces=equivalent_faces,branch_normal_jacobian=normals,
        branch_normal_stiffness_N_per_mm=guard.k*guard.weights[bsites],
        smooth_contact_gradient_N=state['gradient']-chosen,
        smooth_contact_energy_N_mm=state['energy']-float(np.sum(group_energy)),
        original_contact_energy_N_mm=state['energy'],original_contact_gradient_N=state['gradient'],
        provenance=dict(scope='Complete positive-gap exterior finite-feature energy maxima; not summed triangle contact',
            enumeration='All source triangles within unchanged physical clearance using existing radius/AABB finite queries',
            clearance_mm=guard.h,interior_and_cap_terms_unchanged=True,
            native_face_index_scope='Index in the unchanged filtered NativeRigidQuery face inventory',
            zero_branch_included=True,exact_value_gradient_duplicates_removed=exact_duplicates,
            nearly_equal_branches_retained=True,finite_witnesses_valid=True,
            branch_source_sha256=sha(Path(__file__)),construction_seconds=time.perf_counter()-started))
