"""Re-evaluate original semantic vertex loops on a same-order shape checkpoint."""
from pathlib import Path
import argparse,hashlib,json,numpy as np

def refresh(mesh,source_semantics,output):
    mesh=Path(mesh);output=Path(output);output.mkdir(exist_ok=True)
    with np.load(mesh) as q:v=q['vertices'];f=q['triangles'];oid=q['object_ids']
    with np.load(source_semantics) as q:data={k:q[k] for k in q.files}
    for key in list(data):
        if key.endswith('_points'):
            ids=data[key[:-7]+'_vertex_ids'];data[key]=v[ids]
        if key.endswith('_normals'):
            ids=data[key[:-8]+'_vertex_ids'];flag=np.zeros(len(v),bool);flag[ids]=True
            ff=f[np.any(flag[f],axis=1)];p=v[ff];nn=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);normals=np.zeros_like(v)
            for j in range(3):np.add.at(normals,ff[:,j],nn)
            out=normals[ids];out/=np.maximum(np.linalg.norm(out,axis=-1,keepdims=True),1e-15);data[key]=out
    flag=np.zeros(len(v),bool);flag[data['vermilion_vertex_ids']]=True;data['vermilion_face_ids']=np.flatnonzero(np.all(flag[f],axis=1))
    np.savez_compressed(output/'authored_lip_lid_semantics.npz',**data)
    sha=hashlib.sha256(mesh.read_bytes()).hexdigest()
    report={'target_mesh':str(mesh.resolve()),'target_sha256':sha,'source_semantics':str(Path(source_semantics).resolve()),'vertices':len(v),'triangles':len(f),'vermilion_face_count':len(data['vermilion_face_ids']),'vertex_loops_retained':True,'points_and_normals_refreshed':True,'scope':'Same-order authored tissue mapping; not an anatomical, collision or optical transport certificate.'}
    (output/'semantics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mesh');a.add_argument('source_semantics');a.add_argument('output');q=a.parse_args();refresh(q.mesh,q.source_semantics,q.output)
