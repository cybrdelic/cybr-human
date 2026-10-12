"""Prepare or render a hash-pinned procedural HUMAN appearance profile."""
from pathlib import Path
import argparse,hashlib,json,shutil,re,subprocess,os,fcntl,sys
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--profile',choices=['preferred','orbital-v2','margin-v3'],default='preferred');p.add_argument('--geometry-root',type=Path);p.add_argument('--head',type=Path);p.add_argument('--semantics',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--view',choices=['head_front','head'],default='head_front');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
if a.geometry_root:
 g=a.geometry_root.resolve();candidates=[g,g/'project/output/coordinated-orbital-scale-v53']
 found=[x for x in candidates if (x/'head.npz').is_file()]
 if len(found)!=1:p.error('--geometry-root must identify one exact head.npz or a preferred build output root')
 if a.head or a.semantics:p.error('Use either --geometry-root or --head/--semantics')
 a.head=found[0]/'head.npz';a.semantics=found[0]/'anatomical_semantics/authored_lip_lid_semantics.npz'
if a.head is None:p.error('Provide --geometry-root or --head')
if a.semantics is None:
 if a.profile=='preferred':p.error('Preferred profile requires exact --semantics when --head is explicit')
 a.semantics=R/'geometry'/a.profile/'anatomical_semantics/authored_lip_lid_semantics.npz'
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();profiles=json.loads((R/'profiles.json').read_text());cfg=profiles[a.profile]
for path,key in [(a.head,'head_sha256'),(a.semantics,'semantics_sha256')]:
 if sha(path)!=cfg[key]:p.error(f'Input hash mismatch: {key}')
out=a.output.resolve()
if out.exists():p.error('Choose a fresh --output directory')
out.mkdir(parents=True);scene=out/'scene';shutil.copytree(R/'runtime/scene',scene)
src=scene/'scene_mapped.py';s=src.read_text();template_sha=sha(src)
def one(pattern,replacement):
 global s
 s,n=re.subn(pattern,lambda _:replacement,s,flags=re.M)
 if n!=1:raise RuntimeError((pattern,n))
one(r'^REFERENCE_SOURCE=.*$',f'REFERENCE_SOURCE=Path({str(a.head.resolve())!r})')
one(r'^REFERENCE_CAPTURE=.*$',f'REFERENCE_CAPTURE=Path({str((R/"inputs/reference_eye_frame.json").resolve())!r})')
one(r'^SOURCE=.*$',f'SOURCE=REFERENCE_SOURCE if a.reference_control else Path({str(a.head.resolve())!r})')
one(r'^SEMANTICS=.*$',f'SEMANTICS=Path({str(a.semantics.resolve())!r})')
one(r'^PROFILE_GEOMETRY_UNCHANGED=.*$',f'PROFILE_GEOMETRY_UNCHANGED={a.profile=="preferred"}')
s=s.replace(profiles['preferred']['head_sha256'],cfg['head_sha256']).replace(profiles['preferred']['semantics_sha256'],cfg['semantics_sha256']);compile(s,str(src),'exec');src.write_text(s)
for n in ['render.py','uniform_numerical_scale.py']:shutil.copy2(R/'runtime'/n,out/n);compile((out/n).read_text(),n,'exec')
config={**cfg,'profile':a.profile,'input_head_sha256':sha(a.head),'input_semantics_sha256':sha(a.semantics),'template_scene_sha256':template_sha,'prepared_scene_sha256':sha(src),'path_mapping_only':True,'prepare_only':a.prepare_only};(out/'RUN_CONFIGURATION.json').write_text(json.dumps(config,indent=2)+'\n')
if a.prepare_only:print(json.dumps(config,indent=2));sys.exit(0)
cpus=sorted(os.sched_getaffinity(0));chosen=[2,3] if {2,3}<=set(cpus) else cpus[:2];assert len(chosen)==2
a.lock.parent.mkdir(parents=True,exist_ok=True);cmd=['taskset','-c',','.join(map(str,chosen)),sys.executable,str(R/'tools/memory_guard.py'),'--max-rss-mib','900','--nominal-maximum-mib','1024','--min-available-mib','2048','--timeout-seconds','360','--','blender','-b','-t','2','--factory-startup','--python-exit-code','1','--python',str(out/'render.py'),'--','--view',a.view]
with a.lock.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (out/'capture.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'})
sys.exit(r.returncode)
