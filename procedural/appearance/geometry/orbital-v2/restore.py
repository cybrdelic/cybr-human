"""Restore the frozen orbit-only diagnostic from its exact preferred-head input."""
from pathlib import Path
import argparse,hashlib,io,json
import numpy as np

def reconstruct(preferred,checkpoint):
    checkpoint=Path(checkpoint);meta=json.loads((checkpoint/'CHECKPOINT.json').read_text());raw=Path(preferred).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['preferred_head_sha256']:raise ValueError('Exact preferred-v53 head dependency does not match')
    old=np.load(io.BytesIO(raw));m=np.load(checkpoint/'retained_source_maps.npz');d=np.load(checkpoint/'numerical_delta.npz');mapping=m['old_to_new_vertex'];valid=mapping>=0
    v=np.empty((meta['vertices'],3),dtype=np.dtype(meta['vertex_dtype']));v[mapping[valid]]=old['vertices'][valid];v[d['new_vertex_ids']]=d['new_vertices']
    keep=m['retained_source_face_ids'];f=np.empty((meta['triangles'],3),dtype=np.dtype(meta['triangle_dtype']));f[:len(keep)]=mapping[old['triangles'][keep]];f[d['modified_face_ids']]=d['modified_triangles']
    oid=np.empty(meta['triangles'],dtype=np.dtype(meta['object_id_dtype']));oid[:len(keep)]=old['object_ids'][keep];oid[d['modified_face_ids']]=d['modified_object_ids']
    stream=io.BytesIO();np.savez_compressed(stream,vertices=v,triangles=f,object_ids=oid);result=stream.getvalue();sha=hashlib.sha256(result).hexdigest()
    if sha!=meta['candidate_head_sha256']:raise ValueError(f'Restored output SHA mismatch: {sha}')
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preferred',required=True);ap.add_argument('--output',required=True);ap.add_argument('--checkpoint',default=str(Path(__file__).resolve().parent));args=ap.parse_args();data=reconstruct(args.preferred,args.checkpoint);Path(args.output).write_bytes(data);print(json.dumps({'output':args.output,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}))
