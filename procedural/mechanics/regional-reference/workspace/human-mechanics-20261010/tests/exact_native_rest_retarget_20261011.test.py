"""Tiny independent mesh-authoring tests; no candidate-head solve or geometry edit."""
from pathlib import Path
import importlib.util,json,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('retarget',ROOT/'src/prototypes/exact_native_rest_retarget.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RESULTS={}

def fixture():
    x=np.array([[i/2,j/2,k/2] for k in range(3) for j in range(3) for i in range(3)],float)
    t=[]
    for k in range(2):
        for j in range(2):
            for i in range(2):
                ids=np.array([((k+(v>>2))*3+(j+((v>>1)&1)))*3+i+(v&1) for v in range(8)])
                for cell in [[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]]:
                    q=ids[cell].copy()
                    if np.linalg.det((x[q[1:]]-x[q[0]]).T)<0:q[1],q[2]=q[2],q[1]
                    t.append(q)
    return x,np.array(t),np.flatnonzero(((x==0)|(x==1)).any(1))

def binding(x,support):
    return dict(vertex_ids=np.array(support),points=x[support].copy(),
        indices=np.c_[support,(support+1)%len(x),(support+2)%len(x)],weights=np.tile([1.,0,0],(len(support),1)),
        solver_units_per_design_unit=np.array(1.),source_head_sha256=np.array('0'*64),
        source_geometry_sha256=np.array('1'*64))

class Tests(unittest.TestCase):
    def test_identity_is_byte_exact(self):
        x,t,support=fixture();b=binding(x,support);a=m.exact_native_anchors(x,b,x,x,[],scale=1)
        y,q=m.extend_rest(x,t,a)
        self.assertTrue(q['identity_exact']);self.assertTrue(np.array_equal(x,y));RESULTS['identity']=q
    def test_translation_and_affine_patch(self):
        x,t,support=fixture();b=binding(x,support)
        for name,A,v in [('translation',np.eye(3),np.array([.2,-.1,.3])),
                         ('affine',np.array([[1.02,.01,0],[0,.99,.005],[0,0,1.01]]),np.array([.02,.01,-.03]))]:
            target=x@A.T+v;a=m.exact_native_anchors(x,b,x,target,[],scale=1)
            y,q=m.extend_rest(x,t,a,tolerance=1e-10)
            error=float(abs(y-target).max());self.assertTrue(q['converged']);self.assertLess(error,1e-10)
            RESULTS[name]={'converged':True,'maximum_coordinate_error':error,
                'minimum_relative_jacobian':float(np.linalg.det(m.cell_matrices(y,t)@np.linalg.inv(m.cell_matrices(x,t))).min())}
    def test_rooted_nonaffine_extension(self):
        x,t,boundary=fixture();roots=np.flatnonzero(x[:,2]==0);support=np.setdiff1d(boundary,roots)
        b=binding(x,support);target=x.copy();target[22,0]+=.02
        a=m.exact_native_anchors(x,b,x,target,roots,scale=1)
        y,q=m.extend_rest(x,t,a,tolerance=1e-9,iterations_per_stage=160)
        self.assertTrue(q['converged'],str(q));self.assertTrue(np.array_equal(y[roots],x[roots]))
        self.assertTrue(np.array_equal(y[support],target[support]));self.assertGreater(np.linalg.norm(y[13]-x[13]),0)
        J=np.linalg.det(m.cell_matrices(y,t)@np.linalg.inv(m.cell_matrices(x,t)))
        RESULTS['rooted_nonaffine']={'converged':True,'root_exact':True,'surface_exact':True,
            'interior_displacement':(y[13]-x[13]).tolist(),'J_range':[float(J.min()),float(J.max())],
            'accepted_stages':sum(z['converged'] for z in q['stages']),
            'total_inner_iterations':sum(z.get('iterations',0) for z in q['stages'])}
    def test_objective_gradient(self):
        x,t,_=fixture();objective=m.RestMeshObjective(x,t);y=x+.001*np.sin(np.arange(x.size).reshape(x.shape)*.7)
        E,g,F,J=objective.evaluate(y);error=0
        for k in range(y.size):
            old=y.flat[k];h=1e-6;y.flat[k]=old+h;plus=objective.evaluate(y,False)[0]
            y.flat[k]=old-h;minus=objective.evaluate(y,False)[0];y.flat[k]=old
            error=max(error,abs((plus-minus)/(2*h)-g.flat[k]))
        self.assertLess(error,2e-8);RESULTS['objective_gradient_maximum_absolute_error']=float(error)
    def test_segment_rejects_hidden_inversion(self):
        A=np.eye(3)[None];B=np.diag([-2.,-.5,1.])[None]
        self.assertGreater(np.linalg.det(B)[0],0)
        minimum=m.segment_minimum_jacobian(A,B);self.assertLess(minimum,0)
        RESULTS['positive_endpoint_hidden_inversion_minimum']=minimum
    def test_arrays_and_fibres_are_recomputed(self):
        x,t,_=fixture();dm=m.cell_matrices(x,t);V=np.linalg.det(dm)/6;g=np.zeros((len(t),4,3));g[:,1:]=np.linalg.inv(dm);g[:,0]=-g[:,1:].sum(1)
        old=dict(rest=x,nodes=t,gradients=g,material=np.c_[V,np.ones((len(t),2))],mass=np.ones(len(x)))
        A=np.array([[1.03,.02,0],[0,.98,.01],[0,0,1.01]]);target=x@A.T
        regions=[dict(id='test',tetrahedra=list(range(len(t))),fibreDirections=np.tile([1.,0,0],(len(t),1)).tolist(),volumeFraction=.4,activation=0)]
        arrays,updated,quality=m.regenerate_reference(old,target,regions)
        self.assertFalse(quality['accepted_for_mechanics']);self.assertEqual(quality['overlap_audit'],'NOT RUN')
        np.testing.assert_allclose(arrays['material'][:,0],V*np.linalg.det(A),rtol=1e-13)
        np.testing.assert_allclose(updated[0]['fibreDirections'],np.tile(A[:,0]/np.linalg.norm(A[:,0]),(len(t),1)),atol=1e-13)
        self.assertAlmostEqual(arrays['mass'].sum(),1000*arrays['material'][:,0].sum(),places=10)
        self.assertEqual(updated[0]['volumeFraction'],.4);RESULTS['recomputed_reference']=quality
    def test_missing_or_conflicting_identity_is_rejected(self):
        x,t,support=fixture();b=binding(x,support)
        with self.assertRaises(ValueError):m.exact_native_anchors(x,b,x,x,[int(support[0])],scale=1)
        b['weights'][0]=[.5,.5,0]
        with self.assertRaises(ValueError):m.exact_native_anchors(x,b,x,x,[],scale=1)
    def test_binding_rebased_without_changing_correspondence(self):
        x,t,support=fixture();b=binding(x,support);target=x+[.1,.2,.3]
        out=m.retarget_binding(b,target,target,'2'*64,'3'*64)
        for key in ['vertex_ids','indices','weights']:self.assertTrue(np.array_equal(out[key],b[key]))
        self.assertTrue(np.array_equal(out['points'],target[support]));self.assertEqual(str(out['source_head_sha256']),'3'*64)
        self.assertLess(abs(out['rest_offsets']).max(),1e-14)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if result.wasSuccessful():
        path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/exact-native-rest-retarget-20261011-tiny-tests.json'
        with path.open('x') as f:json.dump(dict(passed=True,tests=result.testsRun,scope='27 nodes / 48 tets; no real head remap or global contact/overlap certification',results=RESULTS),f,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
