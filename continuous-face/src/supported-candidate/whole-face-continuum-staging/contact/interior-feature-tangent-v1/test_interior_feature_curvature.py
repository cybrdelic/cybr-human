"""Focused numerical qualification only; never runs a mechanics solve."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE): sys.path.insert(0,str(p))
from coarse_rigid_query import NativeRigidQuery
from coarse_physical_site_contact import CoarsePhysicalSiteContact
from coarse_physical_site_contact_epigraph import ExteriorBranchPhysicalContact
from coarse_physical_site_contact_curvature import InteriorCurvaturePhysicalContact
from interior_feature_curvature import assemble_interior_curvature, _feature


def fixture(point, *, cavity=True, volume=2.7, cube=False):
    faces = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    inner = np.array([[0.,0,0],[2,0,0],[0,2,0],[0,0,2]])
    if cube:
        vertices=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0],[-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2]],float)
        faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    elif cavity:
        outer=np.array([[-5.,-5,-5],[20,-5,-5],[-5,20,-5],[-5,-5,20]])
        vertices=np.r_[outer,inner];faces=np.r_[faces,faces[:,::-1]+4]
    else:
        vertices=inner
    native=NativeRigidQuery(vertices, faces, np.ones(len(faces),int))
    X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float);T=np.array([[0,1,2,3]])
    C=csr_matrix([[.1,.2,.3,.4]])
    X += np.asarray(point)-(C@X)[0]
    solid=SimpleNamespace(X=X,T=T,inv=np.eye(3)[None],mm_per_L0=1.)
    solid.deformation=lambda q: np.stack([q[T[:,k]]-q[T[:,0]] for k in (1,2,3)],axis=2)
    base=CoarsePhysicalSiteContact(solid,C,[volume],native,initial_occupancy_verified=True,attachment_qualified=True)
    old=ExteriorBranchPhysicalContact.from_base(base)
    new=InteriorCurvaturePhysicalContact.from_epigraph(old)
    return solid,old,new


class InteriorCurvatureTests(unittest.TestCase):
    def test_rounded_singular_triangle_only_omits_addition(self):
        native=SimpleNamespace(resolved_faces=np.array([True]),
            triangles=np.array([[[0.,0,0],[1,0,0],[1,1e-12,0]]]), faces=np.array([[0,1,2]]))
        feature,reason=_feature(native,np.array([.5,0,.03]),0,np.array([.5,0,0]),np.array([0.,0.,-1.]),1e-12)
        self.assertIsNone(feature);self.assertEqual(reason,'unresolved_barycentric')

    def test_stable_reentrant_edge_analytic_and_finite_difference(self):
        solid,old,new=fixture([-.03,-.03,.713]);q=solid.X
        data=new.interior_curvature_data(q);self.assertEqual(len(data['rows']),1)
        row=data['rows'][0];self.assertEqual(row['kind'],'edge');self.assertEqual(len(row['tied_faces']),2)
        P=np.eye(3)-np.outer(row['edge_unit'],row['edge_unit'])-np.outer(row['normal'],row['normal'])
        expected=np.kron((new.C.T@new.C).toarray(),row['stiffness_N_per_mm']*P)
        np.testing.assert_allclose(assemble_interior_curvature(new,data).toarray(),expected,atol=1e-15)
        direction=np.random.default_rng(9).normal(size=q.shape);eps=1e-7
        fd=(new.energy(q+eps*direction)[1]-new.energy(q-eps*direction)[1])/(2*eps)
        np.testing.assert_allclose(new.tangent(q)@direction.ravel(),fd.ravel(),rtol=2e-7,atol=1e-9)

    def test_stable_reentrant_vertex_analytic_and_finite_difference(self):
        solid,old,new=fixture([-.025,-.025,-.025]);q=solid.X
        data=new.interior_curvature_data(q);self.assertEqual(len(data['rows']),1)
        row=data['rows'][0];self.assertEqual(row['kind'],'vertex');self.assertEqual(len(row['tied_faces']),3)
        P=np.eye(3)-np.outer(row['normal'],row['normal'])
        expected=np.kron((new.C.T@new.C).toarray(),row['stiffness_N_per_mm']*P)
        np.testing.assert_allclose(assemble_interior_curvature(new,data).toarray(),expected,atol=1e-15)
        direction=np.random.default_rng(19).normal(size=q.shape);eps=1e-7
        fd=(new.energy(q+eps*direction)[1]-new.energy(q-eps*direction)[1])/(2*eps)
        np.testing.assert_allclose(new.tangent(q)@direction.ravel(),fd.ravel(),rtol=2e-7,atol=1e-9)

    def test_planar_face_zero_addition(self):
        solid,old,new=fixture([.312,.417,-.045])
        self.assertEqual(new.interior_curvature_data(solid.X)['report']['stable_face_terms_zero_addition'],1)
        np.testing.assert_array_equal(new.tangent(solid.X).toarray(),old.tangent(solid.X).toarray())

    def test_planar_seam_is_not_curved_edge(self):
        solid,old,new=fixture([0.,0.,.03],cube=True)
        data=new.interior_curvature_data(solid.X)
        self.assertEqual(len(data['rows']),0)
        self.assertEqual(data['report']['skipped_ambiguous_or_unresolved']['edge_face_transition'],1)
        np.testing.assert_array_equal(new.tangent(solid.X).toarray(),old.tangent(solid.X).toarray())

    def test_exterior_edge_no_wrong_sign_addition(self):
        solid,old,new=fixture([-.03,-.03,.713],cavity=False)
        self.assertGreater(new._state(solid.X)['force_gap'][0],0)
        self.assertEqual(len(new.interior_curvature_data(solid.X)['rows']),0)
        np.testing.assert_array_equal(new.tangent(solid.X).toarray(),old.tangent(solid.X).toarray())

    def test_distinct_closest_witness_ambiguity_omitted(self):
        solid,old,new=fixture([.02,.02,.02],cavity=False)
        data=new.interior_curvature_data(solid.X)
        self.assertEqual(len(data['rows']),0)
        self.assertEqual(data['report']['skipped_ambiguous_or_unresolved']['ambiguous_different_closest_points'],1)
        self.assertTrue(new.endpoint(solid.X)['feasible'])

    def test_energy_gradient_guards_inherited_exactly(self):
        solid,old,new=fixture([-.03,-.03,.713]);q=solid.X
        for name in ('energy','_state','endpoint','path','safe_step_bound','contact_branches','contact_accounting','set_barrier_scale'):
            self.assertIs(getattr(type(new),name),getattr(type(old),name))
        e,g=old.energy(q);ne,ng=new.energy(q);self.assertEqual(e,ne);np.testing.assert_array_equal(g,ng)
        for name in ('feasible','minimum_J','maximum_site_penetration_mm','minimum_penetration_slack_mm'):
            self.assertEqual(new.endpoint(q)[name],old.endpoint(q)[name])
        for key,value in old.provenance.items():self.assertEqual(value,new.provenance[key])
        d=np.tile([.001,.0001,0],(4,1))
        self.assertEqual(new.safe_step_bound(q,d),old.safe_step_bound(q,d))
        self.assertEqual(new.path(q,q+d)['feasible'],old.path(q,q+d)['feasible'])
        over=q+[-.01,-.01,0]
        self.assertFalse(old.endpoint(over)['feasible']);self.assertFalse(new.endpoint(over)['feasible'])

    def test_scale_cache_invalidation_and_single_volume_factor(self):
        solid,old,new=fixture([-.03,-.03,.713],volume=2.7);q=solid.X
        before=new.interior_curvature_data(q);H=assemble_interior_curvature(new,before)
        new.set_barrier_scale(.2);after=new.interior_curvature_data(q)
        self.assertIsNot(before,after)
        row=after['rows'][0];s=new._state(q);R=new.k*2.7*(new.h-s['force_gap'][0])+s['barrier']['reaction'][0]
        self.assertEqual(row['reaction_N'],R)
        t=row['basis'][0];direction=np.tile(t,(4,1))
        actual=float(direction.ravel()@(assemble_interior_curvature(new,after)@direction.ravel()))
        self.assertAlmostEqual(actual,R/abs(s['force_gap'][0]),places=12)
        self.assertLess(row['reaction_N'],before['rows'][0]['reaction_N'])
        self.assertGreaterEqual(np.linalg.eigvalsh(new.tangent(q).toarray()).min(),-1e-12)

if __name__=='__main__': unittest.main(verbosity=2)
