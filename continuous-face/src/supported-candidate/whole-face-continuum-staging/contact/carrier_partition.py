"""Exact local P1-injectivity ownership, never a proximity exclusion.

A cut facet is eligible only when every nonzero scalar carrier coefficient
lies in one parent tetrahedron, all coefficients are nonnegative, and their
binary-rational sum is EXACTLY one. This deliberately fails closed for rows
whose floating sums only approximately equal one; no coordinate snapping or
weight normalization is applied. Reference embedding must be certified first.
"""
import numpy as np
from itertools import combinations
from unified_moving_surface_contact import NonsharedSelfContact
from moving_triangle_contact import _map_interval, _iadd, _imul
from conservative_broadphase import candidates as spatial_candidates
from ragged_group_pairs import candidates as ragged_candidates


def exact_unit_sum(values):
    ratios=[float(x).as_integer_ratio() for x in values]
    den=max((b for a,b in ratios),default=1)
    return sum(a*(den//b) for a,b in ratios)==den


class CarrierLocalPartition:
    def __init__(self,solid,C,faces,*,reference_embedding_certified=False):
        self.parents=np.full(len(faces),-1,int)
        self.adjacent_keys=np.empty(0,np.int64)
        self.count=len(solid.T)
        self.enabled=reference_embedding_certified is True
        self.report=dict(enabled=self.enabled,certified_faces=0,unresolved_faces=len(faces),
            reason='reference finite embedding certificate absent' if not self.enabled else '',
            geometry='whole finite facet convexly contained in one parent by exact binary-rational carrier partition',
            assumption='all involved parent determinants remain positive over every accepted segment')
        if not self.enabled:return
        C=C.copy();C.sum_duplicates();C.eliminate_zeros();C.sort_indices()
        incidence=[set() for _ in range(len(solid.X))]
        for parent,tet in enumerate(solid.T):
            for node in tet:incidence[node].add(parent)
        supports=[];valid=[]
        for lo,hi in zip(C.indptr[:-1],C.indptr[1:]):
            values=C.data[lo:hi]
            supports.append(set(C.indices[lo:hi]))
            valid.append(bool(len(values) and np.all(values>=0) and exact_unit_sum(values)))
        for face_id,face in enumerate(faces):
            if not all(valid[v] for v in face):continue
            nodes=set().union(*(supports[v] for v in face))
            if not nodes or len(nodes)>4:continue
            candidates=set.intersection(*(incidence[v] for v in nodes))
            if candidates:self.parents[face_id]=min(candidates)
        # Exact shared node IDs define a complete common face. Verify opposite
        # reference sides independently; same-side/nonmanifold carrier fans
        # never receive this certificate.
        face_owners={}
        for parent,tet in enumerate(solid.T):
            for opposite in range(4):
                face=tuple(sorted(np.delete(tet,opposite)))
                face_owners.setdefault(face,[]).append((parent,int(tet[opposite])))
        keys=[]
        for face,owners in face_owners.items():
            if len(owners)!=2:continue
            (a,va),(b,vb)=owners
            p=solid.X[list(face)];normal=np.cross(p[1]-p[0],p[2]-p[0])
            da=np.dot(normal,solid.X[va]-p[0]);db=np.dot(normal,solid.X[vb]-p[0])
            scale=np.linalg.norm(normal)*max(np.linalg.norm(solid.X[va]-p[0]),np.linalg.norm(solid.X[vb]-p[0]))
            if da*db<0 and min(abs(da),abs(db))>1e-12*scale:
                keys.append(min(a,b)*self.count+max(a,b))
        self.adjacent_keys=np.asarray(sorted(keys),np.int64)
        self.report.update(certified_faces=int(np.sum(self.parents>=0)),
            unresolved_faces=int(np.sum(self.parents<0)),certified_face_neighbor_parent_pairs=len(keys),
            excludes_edge_vertex_only_parent_neighbors=False,
            coarse_limitation='separate fold/slot deformation within one affine carrier cell is unresolved')

    def certified_pairs(self,pairs):
        pairs=np.asarray(pairs,int)
        if not self.enabled or not len(pairs):return np.zeros(len(pairs),bool)
        a,b=self.parents[pairs[:,0]],self.parents[pairs[:,1]]
        valid=(a>=0)&(b>=0)
        key=np.minimum(a,b)*self.count+np.maximum(a,b)
        slots=np.searchsorted(self.adjacent_keys,key)
        adjacent=np.zeros(len(key),bool);within=slots<len(self.adjacent_keys)
        adjacent[within]=self.adjacent_keys[slots[within]]==key[within]
        return valid&((a==b)|adjacent)


def unresolved_adjacency_pairs(faces,material_ids,partition,*,chunk_size=65536):
    """Filter local affine pairs BEFORE storing global shared-feature pairs.

    No multi-million-entry Python pair set is constructed. Only unresolved
    batches survive into the final sorted unique integer pair array.
    """
    ids=np.asarray(material_ids)[faces]
    if len(np.unique(np.sort(ids,axis=1),axis=0))!=len(faces):
        raise ValueError('duplicate material facets must be removed by domain ownership')
    incident=[[] for _ in range(int(ids.max(initial=-1))+1)]
    for face,row in enumerate(ids):
        for vertex in set(row):incident[int(vertex)].append(face)
    pending=[];kept=[];examined=0
    def flush():
        nonlocal examined
        if not pending:return
        pairs=np.asarray(pending,int);examined+=len(pairs)
        pairs=pairs[~partition.certified_pairs(pairs)]
        if len(pairs):kept.append(pairs)
        pending.clear()
    for entries in incident:
        for pair in combinations(entries,2):
            pending.append(pair)
            if len(pending)>=chunk_size:flush()
    flush()
    result=np.unique(np.concatenate(kept),axis=0) if kept else np.empty((0,2),int)
    return result,dict(incident_pair_visits=examined,stored_unresolved_shared_pairs=len(result),
        construction='chunked sparse material incidence; parent-certified pairs removed before global storage')


class CarrierPartitionedSelfContact(NonsharedSelfContact):
    def __init__(self,C,faces,ids,partition,*,leaf_strategy='ragged',**kwargs):
        if leaf_strategy not in ('ragged','spatial'):raise ValueError('unknown exact leaf strategy')
        self.leaf_strategy=leaf_strategy
        self.local_partition=partition
        super().__init__(C,faces,ids,**kwargs)
        parents=partition.parents
        # Unresolved faces have their own conservative leaf group. They never
        # acquire a fictitious parent just to improve the candidate count.
        raw=np.where(parents>=0,parents,partition.count+np.arange(len(faces)))
        _,self.face_group=np.unique(raw,return_inverse=True)
        self.group_count=int(self.face_group.max(initial=-1))+1
        self.group_parent=np.full(self.group_count,-1,int)
        self.group_parent[self.face_group[parents>=0]]=parents[parents>=0]
        self.last_candidate_report={}

    def candidates(self,q,direction=None,max_step=1.,clearance=None):
        if not np.isfinite(max_step) or max_step<0:
            raise ValueError('nonnegative finite step required')
        h=self.clearance if clearance is None else clearance
        V=_map_interval(self.C,q)
        if direction is not None:
            end=_iadd(V,_imul((max_step,max_step),_map_interval(self.C,direction)))
            V=(np.minimum(V[0],end[0]),np.maximum(V[1],end[1]))
        lo,hi=V[0][self.faces_a].min(1),V[1][self.faces_a].max(1)
        glo=np.full((self.group_count,3),np.inf);ghi=np.full_like(glo,-np.inf)
        np.minimum.at(glo,self.face_group,lo);np.maximum.at(ghi,self.face_group,hi)
        groups=spatial_candidates(glo,ghi,glo,ghi,h)
        groups=groups[groups[:,0]<groups[:,1]]
        coarse_candidates=len(groups)
        if len(groups):
            a,b=self.group_parent[groups[:,0]],self.group_parent[groups[:,1]]
            keys=np.minimum(a,b)*self.local_partition.count+np.maximum(a,b)
            local=(a>=0)&(b>=0)&((a==b)|np.isin(keys,self.local_partition.adjacent_keys))
            groups=groups[~local]
        self.last_candidate_report=dict(parent_or_unresolved_leaf_groups=self.group_count,
            coarse_candidate_pairs=coarse_candidates,unresolved_group_pairs=len(groups),
            physical_triangles=len(lo),swept=direction is not None)
        if not len(groups):
            self.last_candidate_report.update(leaf_candidate_pairs=0)
            return np.empty((0,2),int)
        if self.leaf_strategy=='spatial':
            # Retained exact baseline for qualification, not the default.
            active=np.isin(self.face_group,np.unique(groups));ids=np.flatnonzero(active)
            pairs=spatial_candidates(lo[ids],hi[ids],lo[ids],hi[ids],h)
            pairs=ids[pairs];pairs=pairs[pairs[:,0]<pairs[:,1]]
            ga,gb=self.face_group[pairs[:,0]],self.face_group[pairs[:,1]]
            group_keys=groups[:,0]*self.group_count+groups[:,1]
            leaf_keys=np.minimum(ga,gb)*self.group_count+np.maximum(ga,gb)
            pairs=pairs[np.isin(leaf_keys,group_keys)]
            if len(pairs):
                f=self._face_material_ids
                shared=(f[pairs[:,0],:,None]==f[pairs[:,1],None,:]).any((1,2));pairs=pairs[~shared]
            self.last_candidate_report.update(leaf_candidate_pairs=len(pairs),active_leaf_faces=len(ids),leaf_strategy='spatial_baseline')
            return pairs
        pairs,leaf_report=ragged_candidates(lo,hi,self.face_group,groups,self._face_material_ids,h,return_report=True)
        self.last_candidate_report.update(leaf_report)
        return pairs
