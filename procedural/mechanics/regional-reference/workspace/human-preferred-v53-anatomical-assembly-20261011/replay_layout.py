"""Create a new, non-overwriting cloud workspace and replay the numerical layout.
Run this setup outside the HUMAN guard; run the generated shell command inside it.
Only read dependencies are linked. No selected rest surface or source is modified.
"""
from pathlib import Path
import argparse,json,hashlib,shutil
p=argparse.ArgumentParser();p.add_argument('--new-root',required=True);p.add_argument('--dependencies-root',required=True);a=p.parse_args();N=Path(a.new_root).resolve();D=Path(a.dependencies_root).resolve();S=Path(__file__).resolve().parent;assert not N.exists();N.mkdir();out=N/S.name;shutil.copytree(S/'src',out/'src',ignore=shutil.ignore_patterns('__pycache__'))
deps=[('user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz','545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e'),('human-preferred-v53-mechanics-20261011/cheek-remesh-v9/geometry.npz','5fe664216b7670dfecb37500828becbec29c37ea86e6aa59f1e229606fed976c'),('human-preferred-v53-mechanics-20261011/cheek-perioral-insertion-v3/geometry.npz','30a8e5ca8e80f5432190f343d6a636acb87b34a9f2d6173935a67008ed381a1f')]
records=[]
for rel,wanted in deps:
 src=D/rel;got=hashlib.sha256(src.read_bytes()).hexdigest();assert got==wanted,(rel,got);dest=N/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(src);records.append({'path':rel,'sha256':got,'read_only_role':'Input file is read only by all replay scripts; target unchanged.'})
helper='human-preferred-v53-mechanics-20261011/src/triangle_sat.py';dest=N/helper;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(D/helper,dest)
steps=['build_mandible_v3.py','check_mandible_v3.py','resolve_mandible_parity_v3.py','build_routes_v6.py','repair_dao_root_interface_v7.py','snap_origin_bone_edges_v8.py','check_routes_v8.py','release_route_layout_v8.py','build_shared_union_v1.py','check_shared_union_v1.py']
cmd='set -eu\n'+''.join('/usr/bin/blender -b --threads 2 --python-exit-code 1 --python '+str(out/'src'/s)+'\n' for s in steps);(N/'run.sh').write_text(cmd);(N/'dependency_inputs.json').write_text(json.dumps(records,indent=2));print(N/'run.sh')
