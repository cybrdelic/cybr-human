"""Disclosed machine-precision representation repair, never geometric clearance.

Creates a separate boundary packet, preserving the raw source. All vertices
are assigned exact dyadic carrier weights supported in the intersection of
incident facet parents. Any non-microscopic correction fails closed. This
checks mapping/topology/triangle resolution, not missing native-clearance or
full reference nonintersection evidence.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import csr_matrix
from continuum_contact import sha,topology_report
from carrier_partition import exact_unit_sum


def canonicalize(source,output,*,bits=48,coefficient_limit=1e-12,coordinate_limit=1e-10):
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('new output path required; preserve prior evidence')
    started=time.perf_counter()
    with np.load(source,allow_pickle=False) as z:
        X,T,V,F,P,W,Q,K=[z[k].copy() for k in ('nodes','tets','vertices','triangles',
            'exterior_tet','exterior_barycentric','boundary_parent_tet','boundary_class')]
    oldsha=sha(source)
    if np.any(Q<0) or np.any(Q>=len(T)):raise ValueError('invalid explicit facet parent')
    allowed=np.ones(W.shape,bool)
    vertex_nodes=T[P];face_nodes=T[Q]
    for k in range(3):
        ids=F[:,k]
        membership=(vertex_nodes[ids,:,None]==face_nodes[:,None,:]).any(2)
        np.logical_and.at(allowed,ids,membership)
    outside=np.abs(W[~allowed])
    dropped_max=float(outside.max(initial=0))
    negative_max=float((-W[W<0]).max(initial=0))
    if max(dropped_max,negative_max)>coefficient_limit:
        raise ValueError('reference carrier correction exceeds declared machine-scale bound')
    if np.any(~allowed.any(1)):raise ValueError('incident facet parents have no common vertex support')
    clean=np.where(allowed,np.maximum(W,0),0)
    denominator=1<<bits
    integers=np.rint(clean*denominator).astype(np.int64)
    largest=np.argmax(clean,axis=1)
    integers[np.arange(len(W)),largest]+=denominator-integers.sum(1)
    if np.any(integers<0) or np.any(integers.sum(1)!=denominator):
        raise ValueError('invalid dyadic simplex partition')
    canonical=integers/denominator
    if not all(exact_unit_sum(row) for row in canonical):raise ValueError('binary partition proof failed')
    C=csr_matrix((canonical.ravel(),(np.repeat(np.arange(len(V)),4),vertex_nodes.ravel())),shape=(len(V),len(X)))
    C.eliminate_zeros();reference=C@X
    shift=np.linalg.norm(reference-V,axis=1)
    if shift.max(initial=0)>coordinate_limit:raise ValueError('canonical reference shift exceeds declared machine-scale bound')
    old_cross=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
    new_cross=np.cross(reference[F[:,1]]-reference[F[:,0]],reference[F[:,2]]-reference[F[:,0]])
    edge_scale=np.maximum(np.linalg.norm(reference[F[:,1]]-reference[F[:,0]],axis=1),np.linalg.norm(reference[F[:,2]]-reference[F[:,0]],axis=1))
    rank=np.linalg.norm(new_cross,axis=1)/edge_scale**2
    if np.any(rank<=1e-12) or np.any(np.einsum('ij,ij->i',old_cross,new_cross)<=0):
        raise ValueError('reference representation repair degenerates or reverses a facet')
    for k in range(3):
        ids=F[:,k]
        membership=(vertex_nodes[ids,:,None]==face_nodes[:,None,:]).any(2)
        if np.any((canonical[ids]!=0)&~membership):raise ValueError('whole-facet parent support not exact')
    topology=topology_report(F,len(V))
    # A construction-only derived map may preserve a failed source topology.
    # Record it; never let map repair manufacture a geometry certificate.
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,vertices=reference,triangles=F,exterior_tet=P,
        exterior_barycentric=canonical,boundary_parent_tet=Q,boundary_class=K,
        source_domain_sha256=np.array(oldsha),canonical_dyadic_bits=np.array(bits))
    report=dict(status='machine_precision_boundary_representation_checked',
        source_path=str(source.resolve()),source_sha256=oldsha,output_path=str(output.resolve()),
        output_sha256=sha(output),vertices=len(V),triangles=len(F),dyadic_bits=bits,
        max_coefficient_change=float(np.max(abs(canonical-W))),
        max_discarded_off_common_simplex_coefficient=dropped_max,
        max_original_negative_coefficient_magnitude=negative_max,
        coefficient_correction_limit=coefficient_limit,
        maximum_reference_shift_L0=float(shift.max()),coordinate_shift_limit_L0=coordinate_limit,
        all_vertex_partitions_exact_binary_rational=True,all_facets_exactly_in_declared_parent=True,
        minimum_scaled_triangle_rank=float(rank.min()),reversed_or_degenerate_triangles=0,
        topology=topology,source_file_unchanged=sha(source)==oldsha,
        native_collision_reaudit_performed=False,reference_embedding_certified=False,
        motion_certified=False,actual_facial_frames=0,
        remaining='No native-clearance or full reference-embedding certificate is supplied. Mapping repair does not qualify geometric defects or accepted motion.',
        seconds=time.perf_counter()-started)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(canonicalize(args.source,args.output),indent=2))
