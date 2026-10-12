"""Restore optional visual geometry from a hash-pinned preferred rest input."""
from pathlib import Path
import argparse,subprocess,sys,shutil,json,hashlib,os,fcntl
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--preferred',type=Path,required=True);p.add_argument('--variant',choices=['orbital-v2','margin-v3'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);O=a.output.resolve();commands=[]
orb=O/'orbital-v2-head.npz' if a.variant=='margin-v3' else O/'head.npz';commands.append([sys.executable,str(R/'orbital-v2/restore.py'),'--preferred',str(a.preferred.resolve()),'--output',str(orb)])
if a.variant=='margin-v3':commands.append([sys.executable,str(R/'margin-v3/restore.py'),'--baseline',str(orb),'--output',str(O/'head.npz')])
runner=O/'run_owned.py';runner.write_text('import subprocess\ncommands='+repr(commands)+'\nfor command in commands: subprocess.run(command,check=True)\n');cpus=sorted(os.sched_getaffinity(0));chosen=[2,3] if {2,3}<=set(cpus) else cpus[:2];assert len(chosen)==2
a.lock.parent.mkdir(parents=True,exist_ok=True);cmd=['taskset','-c',','.join(map(str,chosen)),sys.executable,str(R.parent/'tools/memory_guard.py'),'--max-rss-mib','900','--nominal-maximum-mib','1024','--min-available-mib','2048','--timeout-seconds','90','--',sys.executable,str(runner)]
with a.lock.open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (O/'replay.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'})
if result.returncode:sys.exit(result.returncode)
shutil.copytree(R/a.variant/'anatomical_semantics',O/'anatomical_semantics');s=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(O/'REPLAY_RECEIPT.json').write_text(json.dumps({'variant':a.variant,'head_sha256':s(O/'head.npz'),'semantics_sha256':s(O/'anatomical_semantics/authored_lip_lid_semantics.npz'),'selected':False,'mechanical_rest_replacement':False},indent=2)+'\n');print(O)
