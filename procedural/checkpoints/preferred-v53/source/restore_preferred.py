#!/usr/bin/env python3
"""Restore the exact preferred original v53, never into an existing directory."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,signal,subprocess,sys,time
from replay_runtime import sha,rss,stop
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--positions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
 pos=a.positions.resolve();out=a.output.resolve();project=out/'project';d=json.loads((ROOT/'REPLAY_INPUTS.json').read_text());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 allowed=sorted(os.sched_getaffinity(0));affinity=[2,3] if {2,3}.issubset(allowed) else allowed[:2];os.sched_setaffinity(0,affinity)
 if a.child:
  import numpy as np
  v=np.load(pos/'vertices.npy');f=np.load(ROOT/'input/triangles.npy');ids=np.load(ROOT/'input/object_ids.npy');input_path=project/'output/coordinated-gestalt-v5/head.npz';input_path.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(input_path,vertices=v,triangles=f,object_ids=ids);del v,f,ids
  assert sha(input_path)==d['expected']['v5'],'Original input assembly mismatch'
  commands=[['reproduce.py','--rebuild-controls','--output','output/coordinated-projection-v6c/head.npz'],['src/freeze_lid_body_v8.py'],['src/freeze_nasal_cervical_v46.py'],['src/freeze_orbital_scale_v53.py']]
  for cmd in commands:subprocess.run([sys.executable]+cmd,cwd=project,env=env,check=True)
  paths={'v5':'coordinated-gestalt-v5/head.npz','v6c':'coordinated-projection-v6c/head.npz','v8':'coordinated-lid-body-v8/head.npz','v46':'coordinated-nasal-cervical-v46/head.npz','v53':'coordinated-orbital-scale-v53/head.npz','v53_semantics':'coordinated-orbital-scale-v53/anatomical_semantics/authored_lip_lid_semantics.npz'}
  checks={k:{'expected':d['expected'][k],'actual':sha(project/'output'/rel)} for k,rel in paths.items()}
  for c in checks.values():c['exact']=c['actual']==c['expected']
  receipt={'preferred_namespace':'original-coordinated-20261010-v53','outputs':checks,'all_exact':all(c['exact'] for c in checks.values()),'v6c_curve_controls_reauthored':True,'starting_input':'Pinned original-generated v5, no claim of all prior stages or r20 recreation'};(out/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
  if not receipt['all_exact']:raise SystemExit(2)
  return
 if out.exists():raise SystemExit('Use a new output directory; no existing selected checkpoint is overwritten')
 for rel,want in d['project_files'].items():
  if sha(ROOT/'project'/rel)!=want:raise SystemExit('Source hash mismatch: '+rel)
 for name,want in d['input_arrays'].items():
  if sha((pos if name=='vertices.npy' else ROOT/'input')/name)!=want:raise SystemExit('Original array hash mismatch: '+name)
 out.mkdir(parents=True);shutil.copy2(ROOT/'PREFERRED_BASELINE.json',out/'PREFERRED_BASELINE.json')
 for rel in d['project_files']:
  dst=project/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'project'/rel,dst)
 started=time.monotonic();proc=None;rc=None;reason=None;peak=0.
 with a.lock.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);admitted=time.monotonic();available=next(float(t.split()[1])/1024 for t in Path('/proc/meminfo').read_text().splitlines() if t.startswith('MemAvailable:'))
  if available<2048:raise SystemExit('Memory admission below 2048 MiB')
  print('ADMITTED preferred original v53 restoration',flush=True)
  try:
   with (out/'restore.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--positions',str(pos),'--output',str(out),'--lock',str(a.lock.resolve()),'--child'],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while proc.poll() is None:
     peak=max(peak,rss(proc.pid))
     if peak>900:reason='900 MiB sampled owned-tree stop';os.killpg(proc.pid,signal.SIGKILL);proc.wait();break
     if time.monotonic()-admitted>87.5:reason='90-second bounded phase';stop(proc);break
     time.sleep(.05)
    rc=proc.wait()
  except BaseException:
   if proc is not None:stop(proc)
   raise
  finally:
   receipt={'exit_code':rc,'stop_reason':reason,'seconds_after_admission':round(time.monotonic()-admitted,3),'seconds_including_wait':round(time.monotonic()-started,3),'peak_sampled_rss_mib':round(peak,2),'sampled_stop_margin_mib':900,'admission_available_mib':available,'admission_required_mib':2048,'cpu_affinity':affinity,'guard_kind':'Sampled user-space owned-tree guard, not a kernel-hard limit','complete':rc==0 and reason is None};(out/'RESOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2));fcntl.flock(lock,fcntl.LOCK_UN);print('RELEASED '+json.dumps(receipt),flush=True)
 if rc or reason:raise SystemExit(rc or 124)
if __name__=='__main__':main()
