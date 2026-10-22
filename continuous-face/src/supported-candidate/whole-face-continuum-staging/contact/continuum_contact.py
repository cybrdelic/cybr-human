"""Finite physical-boundary contact adapter for one P1 tissue continuum.

Only this staging file is new: triangle geometry, finite intersections,
outward-rounded swept certificates, sparse broadphase, barrier gradient, and
PSD normal/Gauss-Newton tangent are reused unchanged from qualified modules.
Bonded phase interfaces are not surfaces. Shared topological boundary features
have geometry guards, not a fitted clearance or artificial local repulsion.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix, hstack, eye

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT/'unified-contact-staging', ROOT/'deformable-contact-staging',
                  ROOT, HERE.parent/'mechanics'):
    sys.path.insert(0,str(directory))
from accelerated_triangle_contact import AcceleratedTriangleContact
from unified_moving_surface_contact import NonsharedSelfContact, material_ids, adjacency_pairs
from shared_feature_contact import SharedFeatureContact
from fixed_geometry_contact import FixedGeometryContact
from moving_triangle_contact import InfeasibleContactError
from finite_closed_contact import ClosedTriangleOwner, UnsignedTriangleSheetOwner
from volume_guard import certify_positive_volume
from carrier_partition import CarrierLocalPartition, CarrierPartitionedSelfContact, unresolved_adjacency_pairs
from implicit_dynamics import REQUIRED_GUARDS
from grid_injectivity import RegularGridInjectivity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary_carrier(solid, vertices, faces, parent, bary, *, coordinates_in_L0=True):
    """No independent surface DOFs, nearest-node projection, or extrapolation."""
    V,F,P,W = np.asarray(vertices,float),np.asarray(faces),np.asarray(parent),np.asarray(bary,float)
    if (V.ndim != 2 or V.shape[1] != 3 or P.shape != (len(V),) or W.shape != (len(V),4)
            or not np.issubdtype(P.dtype,np.integer) or not np.isfinite(V).all()
            or not np.isfinite(W).all() or np.any(W < 0)
            or not np.allclose(W.sum(1),1,atol=1e-12,rtol=0)
            or P.min(initial=0)<0 or P.max(initial=0)>=len(solid.T)):
        raise ValueError('finite in-parent partition-of-unity boundary map required')
    if F.ndim != 2 or F.shape[1] != 3 or not len(F) or not np.issubdtype(F.dtype,np.integer) or F.min()<0 or F.max()>=len(V):
        raise ValueError('nonempty referenced integer physical-boundary triangles required')
    C = csr_matrix((W.ravel(),(np.repeat(np.arange(len(V)),4),solid.T[P].ravel())),shape=(len(V),len(solid.X)))
    C.sum_duplicates(); C.eliminate_zeros(); C.sort_indices()
    reference = V*solid.mm_per_L0 if coordinates_in_L0 else V
    error = float(np.max(abs(C@solid.X-reference)))
    if error > 1e-8*max(1,solid.mm_per_L0):
        raise ValueError(f'physical-boundary/carrier mismatch: {error}')
    return C,F.astype(np.int64),dict(vertices=len(V),triangles=len(F),carrier_nodes=len(solid.X),
        sparse_entries=C.nnz,max_reconstruction_error_mm=error,independent_boundary_dofs=0)


def topology_report(faces, vertex_count):
    """Closed oriented edge manifold AND connected circular vertex links."""
    F = np.asarray(faces,int)
    E = np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]])
    _,inverse,count = np.unique(np.sort(E,axis=1),axis=0,return_inverse=True,return_counts=True)
    sense = np.bincount(inverse,weights=np.where(E[:,0]<E[:,1],1,-1))
    duplicates = len(F)-len(np.unique(np.sort(F,axis=1),axis=0))
    bad_links = []
    incident = [[] for _ in range(vertex_count)]
    for a,b,c in F:
        incident[a].append((int(b),int(c)));incident[b].append((int(c),int(a)));incident[c].append((int(a),int(b)))
    for vertex,edges in enumerate(incident):
        if not edges:
            bad_links.append(vertex);continue
        graph = {}
        for a,b in edges:
            graph.setdefault(a,[]).append(b);graph.setdefault(b,[]).append(a)
        if any(len(v)!=2 for v in graph.values()):
            bad_links.append(vertex);continue
        seen=set();pending=[next(iter(graph))]
        while pending:
            a=pending.pop()
            if a not in seen:seen.add(a);pending.extend(graph[a])
        if len(seen)!=len(graph):bad_links.append(vertex)
    valid = bool(np.all(count==2) and np.all(sense==0) and not duplicates and not bad_links)
    return dict(certified=valid,open_edges=int(np.sum(count==1)),
        nonmanifold_edges=int(np.sum(count>2)),inconsistent_oriented_edges=int(np.sum(sense!=0)),
        duplicate_facets=int(duplicates),nonmanifold_vertex_links=len(bad_links),
        bad_vertex_examples=bad_links[:8])


def signed_winding(point, triangles):
    """One finite containment witness, not a sampled collision substitute."""
    a,b,c = (triangles-np.asarray(point)).transpose(1,0,2)
    la,lb,lc = [np.linalg.norm(x,axis=1) for x in (a,b,c)]
    numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
    denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
    return float(np.sum(2*np.arctan2(numerator,denominator))/(4*np.pi))


class ContinuumContactGuard:
    def __init__(self, solid, C, faces, *, boundary_class, rigid_owners=(),
                 clearance_mm=.02, barrier_stiffness_N_per_mm=.001,
                 initial_evidence=None, attachment_evidence=None,
                 fixture_only=False, jacobian_floor=.1, grid_spacing_mm=None):
        self.solid,self.C,self.faces=solid,csr_matrix(C),np.asarray(faces,int)
        self.fixture_only=bool(fixture_only)
        self.jacobian_floor=float(jacobian_floor)
        self.classes=np.asarray(boundary_class,int)
        if self.C.shape[1]!=len(solid.X) or self.classes.shape!=(len(self.faces),):
            raise ValueError('boundary namespace or class mismatch')
        if not np.all(np.isin(self.classes,np.arange(9))):
            raise ValueError('internal phase interfaces must never enter the physical boundary')
        if not np.isfinite(barrier_stiffness_N_per_mm) or barrier_stiffness_N_per_mm<=0:
            raise ValueError('explicit positive discrete barrier stiffness required')
        self.stiffness=float(barrier_stiffness_N_per_mm)
        self.topology=topology_report(self.faces,self.C.shape[0])
        if not self.topology['certified']:
            raise ValueError('physical boundary is not a closed oriented two-manifold: '+str(self.topology))
        self.ids=material_ids(self.C)
        self.initial_evidence=dict(initial_evidence or {})
        self.partition=CarrierLocalPartition(solid,self.C,self.faces,reference_embedding_certified=
            self.initial_evidence.get("reference_boundary_embedding_certified") is True)
        self.injectivity=None
        self.injectivity_report=dict(enabled=False,reason='regular-grid proof not supplied')
        if grid_spacing_mm is not None:
            if self.partition.report['certified_faces']!=len(self.faces):
                self.injectivity_report=dict(enabled=False,reason='exact whole-facet/reference embedding preconditions incomplete')
            else:
                try:
                    self.injectivity=RegularGridInjectivity(solid,grid_spacing_mm)
                    self.injectivity_report=self.injectivity.report
                except ValueError as exc:
                    self.injectivity_report=dict(enabled=False,reason=str(exc))
        pairs,self.adjacency_report=unresolved_adjacency_pairs(self.faces,self.ids,self.partition)
        self.nonshared=CarrierPartitionedSelfContact(self.C,self.faces,self.ids,self.partition,clearance=clearance_mm)
        self.shared=SharedFeatureContact(self.C,self.faces,self.faces,pairs,clearance=clearance_mm)
        self.rigid_terms=[];self.rigid_owners=[]
        n=len(solid.X)
        for owner in rigid_owners:
            if not isinstance(owner,(ClosedTriangleOwner,UnsignedTriangleSheetOwner)):
                raise ValueError('unchanged native finite rigid owner required')
            if isinstance(owner,UnsignedTriangleSheetOwner) and not getattr(owner,'initial_side_qualified',False):
                raise ValueError('open rigid sheet requires explicit qualified initial-side evidence; no signed-solid fiction')
            m=len(owner.vertices)
            A=hstack([self.C,csr_matrix((self.C.shape[0],m))],format='csr')
            B=hstack([csr_matrix((m,n)),eye(m,format='csr')],format='csr')
            term=AcceleratedTriangleContact(A,self.faces,B,owner.faces,clearance=clearance_mm)
            self.rigid_terms.append(FixedGeometryContact(term,n,owner.vertices))
            self.rigid_owners.append(owner)
        self.initial_evidence=dict(initial_evidence or {})
        self.attachment_evidence=dict(attachment_evidence or {})
        required_initial=('source_geometry_qualified','native_rigid_inventory_complete',
                          'cavity_inventory_complete','boundary_is_physical_material_boundary',
                          'rigid_source_self_intersections_checked','reference_boundary_embedding_certified')
        if self.fixture_only and not solid.provenance.get('kind','').startswith('synthetic'):
            raise ValueError('fixture-only guard cannot be constructed for actual anatomy')
        reference=self.endpoint(solid.X)
        initial=bool(reference['feasible'] and all(self.initial_evidence.get(k) is True for k in required_initial))
        self.certification={k:False for k in REQUIRED_GUARDS}
        self.certification.update(source_initial_state=initial,
            rigid_bone_cartilage=initial,lumens=initial,separated_exterior_sheets=initial,
            self_collision=initial,finite_contact_forces=True,whole_segment_collision=True,
            finite_source_origin_attachments=self.attachment_evidence.get('finite_material_attachment_qualified') is True)
        self.report=dict(fixture_only=self.fixture_only,certification=self.certification,
            reference_endpoint=reference,topology=self.topology,
            global_injectivity_shortcut=self.injectivity_report,
            adjacent_pairs=len(pairs), adjacency_construction=self.adjacency_report, carrier_local_partition=self.partition.report, boundary_faces=len(self.faces),
            rigid_owners=len(self.rigid_owners), native_rigid_triangles=sum(len(x.faces) for x in self.rigid_owners),
            boundary_class_counts={str(k):int(np.sum(self.classes==k)) for k in np.unique(self.classes)},
            contact_clearance_mm=float(clearance_mm),barrier_stiffness_N_per_mm=self.stiffness,
            energy_model='unchanged activated finite-triangle log barrier with unit facet pair factors; explicitly mesh-dependent numerical assumption, not calibrated pressure',
            tangent_model='PSD normal Gauss-Newton approximation, not exact contact Hessian; mechanics tangent remains exact',
            adjacency='only exact common material vertices; shared-feature forbidden intersections and whole-path guards retained; no local adjacency barrier energy',
            bone_pair_exclusions=0,source_origin_bone_aliases=0,
            candidate_strategy='conservative parent-group radius-bucket cKDTree, then bounded ragged Cartesian finite-facet expansion; unchanged outward AABB/material filters and sparse material incidence',
            initial_evidence=self.initial_evidence,attachment_evidence=self.attachment_evidence)

    def _contains_rigid(self,V):
        records=[]
        triangles=V[self.faces]
        for owner in self.rigid_owners:
            eligible=np.flatnonzero(np.all((V>=owner.low)&(V<=owner.high),axis=1))
            closed=isinstance(owner,ClosedTriangleOwner)
            if closed:
                signed=owner.query(V[eligible])
                invalid=~signed['valid'] | (signed['signed_gap']<=0)
            else:
                # No inferred inside/outside for an open native sheet. Its
                # finite facets remain in endpoint and full-path contact.
                eligible=np.empty(0,int);invalid=np.empty(0,bool)
            # Since both closed boundaries were finitely intersection-tested,
            # winding is constant on each connected rigid owner. One surface
            # representative excludes the opposite full-containment case.
            winding=signed_winding(owner.vertices[owner.faces[0,0]],triangles)
            good=not invalid.any() and np.isfinite(winding) and abs(winding)<1e-7
            records.append(dict(name=owner.name,feasible=bool(good),
                mode='closed_signed' if closed else 'explicit_unsigned_native_sheet',
                inside_sign_supplied=closed,initial_side_qualified=closed or bool(owner.initial_side_qualified),
                signed_vertex_queries=len(eligible),inside_or_ambiguous_vertices=int(invalid.sum()),
                rigid_surface_witness_tissue_winding=winding))
        return records

    def endpoint(self,q):
        volume=certify_positive_volume(self.solid,q,q,floor=self.jacobian_floor)
        if not volume['certified']:
            return dict(feasible=False,reason='positive-volume endpoint',volume=volume)
        injectivity=None if self.injectivity is None else self.injectivity.bound(q)
        states=[] if injectivity is not None and injectivity['certified'] else [('nonlocal_self',self.nonshared.evaluate(q)),('topological_shared',self.shared.evaluate(q))]
        states.extend((owner.name,term.evaluate(q)) for owner,term in zip(self.rigid_owners,self.rigid_terms))
        surfaces=[dict(name=name,feasible=bool(s.feasible),reason=s.reason,
                    candidate_pairs=len(s.pairs) if hasattr(s,'pairs') else len(self.shared.pairs)) for name,s in states]
        if not all(s.feasible for _,s in states):
            return dict(feasible=False,reason='finite boundary intersection or unresolved separation',volume=volume,surfaces=surfaces)
        containment=self._contains_rigid(np.asarray(self.C@q))
        return dict(feasible=all(x['feasible'] for x in containment),volume=volume,
                    surfaces=surfaces,rigid_containment=containment,self_injectivity=injectivity)

    def energy(self,q):
        e,g=self.nonshared(q)
        for term in self.rigid_terms:
            ee,gg=term(q);e+=ee;g+=gg
        return self.stiffness*e,self.stiffness*g

    def tangent(self,q):
        K=self.nonshared.normal_tangent(q)
        for term in self.rigid_terms:K+=term.normal_tangent(q)
        return (self.stiffness*K).tocsr()

    def _surface_path(self,q,d,q1=None):
        injectivity=None if self.injectivity is None else (self.injectivity.direction_path(q,d) if q1 is None else self.injectivity.path(q,q1))
        if injectivity is not None and injectivity['certified']:
            reports=[dict(name='global_carrier_injectivity',**injectivity)]
        else:
            reports=[dict(name='nonlocal_self',**self.nonshared.certify_segment(q,d,1.)),
                     dict(name='topological_shared',**self.shared.certify_segment(q,d,1.))]
        for owner,term in zip(self.rigid_owners,self.rigid_terms):
            reports.append(dict(name=owner.name,**term.term.certify_segment(term._q(q),term._direction(d),1.)))
        return reports

    def path(self,q0,q1):
        volume=certify_positive_volume(self.solid,q0,q1,floor=self.jacobian_floor)
        if not volume['certified']:
            return dict(feasible=False,whole_segment_certified=False,volume=volume)
        start=self.endpoint(q0);end=self.endpoint(q1)
        if not start['feasible'] or not end['feasible']:
            return dict(feasible=False,whole_segment_certified=False,volume=volume,start=start,end=end)
        try:
            reports=self._surface_path(q0,np.asarray(q1)-np.asarray(q0),q1=np.asarray(q1))
            good=all(x['certified'] for x in reports)
        except (InfeasibleContactError,ValueError,FloatingPointError) as exc:
            reports=[dict(certified=False,reason=str(exc))];good=False
        return dict(feasible=good,whole_segment_certified=good,volume=volume,surfaces=reports)

    def safe_step_bound(self,q,direction,*,safety=.9):
        if not 0<safety<1:raise ValueError('safety must be in (0,1)')
        injectivity=None if self.injectivity is None else self.injectivity.direction_path(q,direction)
        if injectivity is not None and injectivity['certified']:
            bound=1.
        else:
            bound=self.nonshared.safe_step_bound(q,direction,safety=safety,max_step=1.)
            bound=self.shared.safe_step_bound(q,direction,safety=safety,max_step=bound)
        for term in self.rigid_terms:
            bound=term.safe_step_bound(q,direction,safety=safety,max_step=bound)
        for _ in range(32):
            if certify_positive_volume(self.solid,q,q+bound*direction,floor=self.jacobian_floor)['certified']:
                return float(bound)
            bound*=.5
        return 0.


class UnavailableGuard:
    fixture_only=False
    def __init__(self,reason):
        self.certification={k:False for k in REQUIRED_GUARDS}
        self.report=dict(status='blocked_before_motion',reason=reason,certification=self.certification)


def create_guard(solid,constraints):
    """Mechanics entry point. No implicit adoption of voxel preview geometry.

    Add provenance.contact_manifest_path, or place <domain-stem>-contact.json
    beside the domain. Manifest and payload hashes bind all external evidence.
    A missing/unqualified manifest yields an explicitly unavailable guard.
    """
    domain=Path(solid.provenance.get('domain_path',''))
    path=Path(solid.provenance.get('contact_manifest_path',domain.with_name(domain.stem+'-contact.json') if domain.name else HERE/'absent-contact.json'))
    if not path.is_file():
        return UnavailableGuard('No qualified repaired-cut boundary/contact manifest. Literal voxel baseline is construction-only and cannot certify motion.')
    try:
        manifest=json.loads(path.read_text())
        if manifest.get('kind')!='conforming_cut_boundary_contact_v1':raise ValueError('repaired conforming-cut contact manifest required')
        if manifest.get('domain_sha256')!=solid.provenance.get('domain_sha256'):raise ValueError('contact manifest domain hash mismatch')
        required_initial=('source_geometry_qualified','native_rigid_inventory_complete','cavity_inventory_complete',
            'boundary_is_physical_material_boundary','rigid_source_self_intersections_checked','reference_boundary_embedding_certified')
        missing=[key for key in required_initial if manifest.get('initial_evidence',{}).get(key) is not True]
        if missing:raise ValueError('Initial geometry evidence incomplete: '+', '.join(missing))
        boundary=(path.parent/manifest['boundary_path']).resolve()
        if sha(boundary)!=manifest['boundary_sha256']:raise ValueError('boundary packet hash mismatch')
        with np.load(boundary,allow_pickle=False) as z:
            C,F,mapping=boundary_carrier(solid,z['vertices'],z['triangles'],z['exterior_tet'],z['exterior_barycentric'])
            classes=z['boundary_class'].copy()
        owners=[]
        for row in manifest['rigid_owners']:
            source=(path.parent/row['path']).resolve()
            if sha(source)!=row['sha256']:raise ValueError('native rigid packet hash mismatch')
            with np.load(source,allow_pickle=False) as z:
                V=z[row.get('vertices_key','vertices')]*solid.mm_per_L0
                F=z[row.get('faces_key','triangles')]
                if 'part_id' in row:F=F[z[row.get('part_ids_key','part_ids')]==row['part_id']]
                if not len(F):raise ValueError('native rigid source selection is empty')
                used,inverse=np.unique(F,return_inverse=True);V=V[used];F=inverse.reshape(-1,3)
                mode=row.get('mode','closed_signed')
                if mode=='closed_signed':owner=ClosedTriangleOwner(V,F,name=row['name'],owner_id=row['owner_id'])
                elif mode=='unsigned_sheet':
                    if row.get('initial_side_qualified') is not True or row.get('finite_sheet_model_declared') is not True:
                        raise ValueError('unsigned source sheet requires explicit finite-sheet model and initial-side evidence')
                    owner=UnsignedTriangleSheetOwner(V,F,name=row['name'],owner_id=row['owner_id'])
                    owner.initial_side_qualified=True
                else:raise ValueError('unknown native rigid contact mode')
                owners.append(owner)
        attachment=manifest.get('attachment_evidence',{})
        # The mechanics support map and its finite material ownership audit
        # must be explicitly tied together; current point-only candidates fail.
        if constraints is None or attachment.get('constraint_source')!=constraints.report.get('source'):
            attachment={**attachment,'finite_material_attachment_qualified':False}
        guard=ContinuumContactGuard(solid,C,F,boundary_class=classes,rigid_owners=owners,
            clearance_mm=manifest['clearance_mm'],barrier_stiffness_N_per_mm=manifest['barrier_stiffness_N_per_mm'],
            initial_evidence=manifest['initial_evidence'],attachment_evidence=attachment,
            grid_spacing_mm=None if 'proof_grid_spacing_L0' not in manifest else float(manifest['proof_grid_spacing_L0'])*solid.mm_per_L0)
        guard.report.update(mapping=mapping,manifest_path=str(path),manifest_sha256=sha(path))
        return guard
    except (ValueError,KeyError,OSError,TypeError) as exc:
        return UnavailableGuard(str(exc))
