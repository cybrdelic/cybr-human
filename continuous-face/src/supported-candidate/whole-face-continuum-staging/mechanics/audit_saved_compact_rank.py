"""One read-only comparison on the saved actual QP, never a motion solve."""
import importlib.util, json, time
from pathlib import Path
import numpy as np
from scipy.sparse import vstack
from scipy.linalg import lstsq
from source_transfer import load_geometry_with_source_fibres, sha
from clipped_origin_support import load_support_bundle
from compact_seam_rank import independent_seams
from coarse_physical_site_contact_epigraph import create_guard
root=Path(__file__).resolve().parent
case=root/'coarse-barrier-v1/first-step-v7'
meta=json.loads((case/'passive-step0001-NONPHYSICAL-checkpoint.json').read_text())
cp=meta['failed_inner_report']['history'][-1]['epigraph_qp_checkpoint']
q=np.load(case/'passive-step0001-NONPHYSICAL-checkpoint.npz')['nonphysical_iterate_mm']
solid=load_geometry_with_source_fibres(root.parent/'geometry/whole-face-continuum-final.npz',root.parent/'geometry/whole-face-muscle-phases-final.npz')
constraints,_=load_support_bundle(solid,root/'coarse-physical-site-v1/supports')
guard=create_guard(solid,constraints);guard.set_barrier_scale(meta['barrier_scale'])
b=guard.contact_branches(q);G=b['branch_gradient_N']; E=b['branch_energy_N_mm']; groups=b['branch_group']
d=np.array(cp['direction']);t=np.array(cp['epigraph_values']);active=cp['active_branches']
violation=float(np.max(E+G@d-t[groups]))
assert violation<1e-11,violation
A=vstack([G[j]-G[a[0]] for a in active for j in a[1:]],format='csr');C=constraints.C3
spec=importlib.util.spec_from_file_location('frozen_full_rank',root/'coarse-barrier-v1/first-step-v7-source-snapshot/exterior_epigraph.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
a=time.perf_counter(); old_ids=old._independent_seams(A,np.zeros(A.shape[0]),C);old_s=time.perf_counter()-a
a=time.perf_counter(); new_ids=independent_seams(A,np.zeros(A.shape[0]),C);new_s=time.perf_counter()-a
cols=np.union1d(A.indices,C.indices);Ac=A[:,cols].toarray();Cc=C[:,cols].toarray()
# Orthogonal projection only in the exact nonzero union, no physical solve.
from scipy.sparse.linalg import splu
P=Ac-(C[:,cols].T@splu((C@C.T).tocsc()).solve(C[:,cols]@Ac.T)).T
coeff,_,_,_=lstsq(P[new_ids].T,P[old_ids].T,cond=1e-12)
error=float(np.max(abs(P[new_ids].T@coeff-P[old_ids].T)))
report=dict(checkpoint_npz_sha256=sha(case/'passive-step0001-NONPHYSICAL-checkpoint.npz'),
    checkpoint_json_sha256=sha(case/'passive-step0001-NONPHYSICAL-checkpoint.json'),
    compact_rank_sha256=sha(root/'compact_seam_rank.py'),full_columns=A.shape[1],union_columns=len(cols),
    seam_rows=A.shape[0],full_rank=len(old_ids),compact_rank=len(new_ids),full_seconds=old_s,
    compact_seconds=new_s,retained_projected_row_span_error_N=error,
    saved_qp_branch_feasibility_error_N_mm=violation,physical_solve_run=False)
assert len(old_ids)==len(new_ids) and error<1e-10,report
out=root/'coarse-barrier-v1/factor-reuse-qualification/saved-compact-rank.json'
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
