"""Small chart/topology/transfer tests; no full head, binding, or solve."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('oral24_binding',ROOT/'src/prototypes/build_bilateral_native_binding_oral24.py')
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class Oral24BindingTests(unittest.TestCase):
    def test_external_material_rows_exact_once(self):
        self.check_rows(b.EXTERNAL_ROWS,17)

    def test_mucosal_material_rows_exact_once(self):
        self.check_rows(b.MUCOSAL_ROWS,25)

    def check_rows(self,rows,count):
        seen=set()
        for angle in range(120):
            for radial in range(4):
                for row,angular,uv in b.material_samples(radial,rows,angle):
                    self.assertNotIn((row,angular),seen)
                    seen.add((row,angular))
                    self.assertEqual(uv[0],(angular-4*angle)/4)
                    self.assertEqual(uv[1],(row-rows[radial])/(rows[radial+1]-rows[radial]))
                    self.assertTrue(np.all((uv>=0)&(uv<=1)))
        self.assertEqual(seen,{(r,a) for r in range(count) for a in range(480)})

    def test_reversed_quad_actual_diagonal(self):
        uv=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        # Both orientations split through their actual first/third corners.
        for order in [np.arange(4),np.arange(4)[::-1]]:
            q=uv[order]
            triangles=[q[:3],q[[0,2,3]]]
            for _,_,point in b.material_samples(0,b.MUCOSAL_ROWS,0):
                choices=[(tri,b.barycentric(point,tri)) for tri in triangles]
                valid=[(tri,w) for tri,w in choices if w is not None]
                self.assertTrue(valid)
                for tri,w in valid:
                    np.testing.assert_allclose(w@tri,point,rtol=0,atol=1e-15)
                    self.assertGreaterEqual(w.min(),0)

    def test_shared_edge_support_ignores_zero_weights(self):
        self.assertEqual(b.support_key([1,2,3],[.25,.75,0]),
                         b.support_key([4,2,1],[0,.75,.25]))

    def test_explicit_mucosal_sidewall_periodic_seam(self):
        native=np.arange(25*480).reshape(25,480)
        triangles=np.array([[0,1,2],[0,2,3]])
        coords=np.array([[[476,16],[480,16],[480,24]],
                         [[476,16],[480,24],[476,24]]])
        records={}
        def put(nid,nodes,weights,region,parent,uv):
            key=b.support_key(nodes,weights)
            if int(nid) in records:
                self.assertEqual(records[int(nid)],key)
            records[int(nid)]=key
            self.assertEqual(region,2)
            self.assertTrue(np.all((uv>=0)&(uv<=1)))
        b.bind_explicit_mucosal_facets(native,triangles,coords,np.array([1590,1590]),put)
        self.assertEqual(set(records),{int(native[r,a%480]) for r in range(16,25) for a in range(476,481)})

    def test_explicit_mucosal_sidewall_rejects_wrapped_long_triangle(self):
        with self.assertRaises(ValueError):
            b.bind_explicit_mucosal_facets(np.arange(25*480).reshape(25,480),
                np.array([[0,1,2]]),np.array([[[476,16],[0,16],[0,24]]]),np.array([1590]),lambda *x:None)

    def test_missing_cut_is_rejected(self):
        with self.assertRaises(ValueError):
            b.binding_rows(np.array([2,5,8]),np.array([2,9]))
        with self.assertRaises(ValueError):
            b.binding_rows(np.array([2,5,8]),np.array([2,7]))

    def test_actual_incident_star_uses_positive_support_only(self):
        t=np.array([[0,1,2,3],[1,2,3,4],[2,3,4,5]])
        nodes,incident,star=b.recovery_star(t,np.array([[0,1,2],[4,3,2]]),
                                          np.array([[1.,0,0],[.5,.25,.25]]),np.array([0]))
        np.testing.assert_array_equal(nodes,[0])
        np.testing.assert_array_equal(incident,[0])
        np.testing.assert_array_equal(star,[0,1,2,3])

    def test_actual_boundary_owners_exclude_shared_face(self):
        owners=b.boundary_owners(np.array([[0,1,2,3],[1,2,3,4]]))
        self.assertNotIn((1,2,3),owners)
        self.assertEqual(owners[(0,1,2)],0)
        self.assertEqual(owners[(2,3,4)],1)
        self.assertEqual(len(owners),6)

    def test_native_cap_boundary_edge_identity(self):
        triangles=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        np.testing.assert_array_equal(b.boundary_edges(triangles),
                                      b.sorted_edges(b.ring_edges(np.arange(4))))

    def test_rejects_nonmanifold_native_edge(self):
        with self.assertRaises(ValueError):
            b.boundary_edges(np.array([[0,1,2],[1,0,3],[0,1,4]]))

    def test_original_transfer_and_nontrivial_frozen_collar(self):
        v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
                    [1.,1.,1.],[2.,1.,0.]])
        t=np.array([[0,1,2,3],[1,2,3,4],[2,3,4,5]])
        for tet in t:
            if np.linalg.det(np.stack([v[tet[k]]-v[tet[0]] for k in [1,2,3]],axis=-1))<0:
                tet[1],tet[2]=tet[2],tet[1]
        payload={'points':np.array([[.0001,.0002,-.0003],[.0004,.0003,.0002]]),
                 'indices':np.array([[0,1,2],[3,4,5]]),'weights':np.array([[1.,0,0],[.2,.4,.4]]),
                 'unmatched_cut_binding_rows':np.array([0])}
        nodes,incident,star=b.recovery_star(t,payload['indices'],payload['weights'],np.array([0]))
        payload.update(unmatched_cut_support_node_ids=nodes,cut_recovery_incident_node_ids=star)
        report=b.verify_original_transfer(payload,v,t)
        probe=report['syntheticCollarProbe']
        self.assertEqual(probe['perturbedNoncollarNodes'],2)
        self.assertLessEqual(probe['maximumNativeCutCoordinateErrorDesignUnits'],1e-10)
        self.assertLessEqual(report['originalFloat32RecoveredMaxCoordinateErrorDesignUnits'],1e-5)


if __name__=='__main__':
    unittest.main(verbosity=2)
