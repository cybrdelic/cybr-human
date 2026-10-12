"""Explicit opt-in replay from the exact preferred head. Defaults are untouched."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,os,fcntl,shutil
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--preferred',type=Path,required=True);p.add_argument('--appearance-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);a=p.parse_args();A=a.appearance_root.resolve();O=a.output.resolve();cfg=json.loads((R/'PROFILE.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(a.preferred)==cfg['preferred_head_sha256'];assert not O.exists(),'Choose a fresh output directory';O.mkdir(parents=True)
cpus=sorted(os.sched_getaffinity(0));chosen=[2,3] if {2,3}<=set(cpus) else cpus[:2];assert len(chosen)==2;env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'};a.lock.parent.mkdir(parents=True,exist_ok=True)
def stage(name,command,already_guarded=False):
 if already_guarded:
  with (O/(name+'.log')).open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
  return
 cmd=['taskset','-c',','.join(map(str,chosen)),sys.executable,str(A/'tools/memory_guard.py'),'--max-rss-mib','900','--min-available-mib','2048','--timeout-seconds','90','--',*command]
 with a.lock.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with (O/(name+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
stage('margin',[sys.executable,str(A/'geometry/replay.py'),'--preferred',str(a.preferred.resolve()),'--variant','margin-v3','--output',str(O/'margin'),'--lock',str(a.lock.resolve())],True)
stage('nasal',[sys.executable,str(R/'nasal/build.py'),'--baseline',str(O/'margin/head.npz'),'--output',str(O/'nasal')])
stage('surface',[sys.executable,str(R/'surface/build.py'),'--baseline',str(O/'nasal/head.npz'),'--output',str(O/'surface')])
for script,name in [('audit_surface.py','surface_audit'),('check_topology.py','topology')]:stage(name,[sys.executable,str(R/'surface/source'/script),'--candidate',str(O/'surface')])
shutil.copy2(O/'surface/candidate_head.npz',O/'head.npz');shutil.copytree(O/'surface/anatomical_semantics',O/'anatomical_semantics');assert sha(O/'head.npz')==cfg['head_sha256'];assert sha(O/'anatomical_semantics/authored_lip_lid_semantics.npz')==cfg['semantics_sha256'];cfg.update(fresh_full_chain_replay=True,geometry_root=str(O),default_changed=False);(O/'REPLAY_RECEIPT.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(cfg,indent=2))
