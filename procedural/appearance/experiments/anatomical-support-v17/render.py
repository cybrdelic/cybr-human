"""Reuse the published appearance runtime for an explicit opt-in candidate."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,os,fcntl,shutil,re
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--appearance-root',type=Path,required=True);p.add_argument('--geometry-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--view',choices=['head_front','head'],default='head_front');p.add_argument('--fit-only',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();A=a.appearance_root.resolve();G=a.geometry_root.resolve();O=a.output.resolve();cfg=json.loads((R/'PROFILE.json').read_text());preferred=json.loads((A/'profiles.json').read_text())['preferred'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();head=G/'head.npz';sem=G/'anatomical_semantics/authored_lip_lid_semantics.npz'
for path,key in [(head,'head_sha256'),(sem,'semantics_sha256')]:assert sha(path)==cfg[key],key
if not O.exists():
 O.mkdir(parents=True);shutil.copytree(A/'runtime/scene',O/'scene');src=O/'scene/scene_mapped.py';s=src.read_text();template_sha=sha(src)
 def one(pattern,replacement):
  global s
  s,n=re.subn(pattern,lambda _:replacement,s,flags=re.M);assert n==1,(pattern,n)
 one(r'^REFERENCE_SOURCE=.*$',f'REFERENCE_SOURCE=Path({str(head)!r})');one(r'^REFERENCE_CAPTURE=.*$',f'REFERENCE_CAPTURE=Path({str(A/"inputs/reference_eye_frame.json")!r})');one(r'^SOURCE=.*$',f'SOURCE=REFERENCE_SOURCE if a.reference_control else Path({str(head)!r})');one(r'^SEMANTICS=.*$',f'SEMANTICS=Path({str(sem)!r})');one(r'^PROFILE_GEOMETRY_UNCHANGED=.*$','PROFILE_GEOMETRY_UNCHANGED=False')
 s=s.replace(preferred['head_sha256'],cfg['head_sha256']).replace(preferred['semantics_sha256'],cfg['semantics_sha256']).replace('preferred-v53-integrated-tissue-eye-groom-candidate-v1',cfg['profile']).replace('Unselected coordinated tissue/eye/groom candidate on exact preferred-v53 rest geometry',cfg['geometry_scope']);compile(s,str(src),'exec');src.write_text(s)
 for name in ['render.py','uniform_numerical_scale.py']:shutil.copy2(A/'runtime'/name,O/name)
 q=O/'render.py';s=q.read_text().replace("p=argparse.ArgumentParser();","p=argparse.ArgumentParser();p.add_argument('--fit-only',action='store_true');");marker="path=O/f'{view}_conditioned_transport_raw.png'";assert marker in s;s=s.replace(marker,"if a.fit_only:\n (O/'pre_render_fit_manifest.json').write_text(json.dumps(m,indent=2)+'\\n');print('CANDIDATE_FIT_VERIFIED',flush=True);raise SystemExit(0)\n"+marker);compile(s,str(q),'exec');q.write_text(s)
 cfg.update(input_head_sha256=sha(head),input_semantics_sha256=sha(sem),template_scene_sha256=template_sha,prepared_scene_sha256=sha(src),head_path=str(head),semantics_path=str(sem),path_mapping_only=True);(O/'RUN_CONFIGURATION.json').write_text(json.dumps(cfg,indent=2)+'\n')
else:
 old=json.loads((O/'RUN_CONFIGURATION.json').read_text());assert old['profile']==cfg['profile'] and old['input_head_sha256']==cfg['head_sha256'] and old['input_semantics_sha256']==cfg['semantics_sha256'];assert sha(O/'scene/scene_mapped.py')==old['prepared_scene_sha256']
if a.prepare_only:print(O);sys.exit(0)
cpus=sorted(os.sched_getaffinity(0));chosen=[2,3] if {2,3}<=set(cpus) else cpus[:2];assert len(chosen)==2;a.lock.parent.mkdir(parents=True,exist_ok=True);command=['taskset','-c',','.join(map(str,chosen)),sys.executable,str(A/'tools/memory_guard.py'),'--max-rss-mib','900','--min-available-mib','2048','--timeout-seconds','90' if a.fit_only else '360','--','blender','-b','-t','2','--factory-startup','--python-exit-code','1','--python',str(O/'render.py'),'--','--view',a.view];command+=['--fit-only'] if a.fit_only else []
with a.lock.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (O/('fit.log' if a.fit_only else a.view+'.log')).open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'})
