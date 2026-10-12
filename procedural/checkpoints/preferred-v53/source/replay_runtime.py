#!/usr/bin/env python3
"""Rebuild the pinned original orbital-body and terminal-eye checkpoint from portable source inputs."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rss(pid):
 rows={}
 for p in Path('/proc').iterdir():
  if not p.name.isdecimal():continue
  try:
   t=p.joinpath('status').read_text().splitlines();rows[int(p.name)]=(next((int(s.split()[1]) for s in t if s.startswith('PPid:')),0),next((float(s.split()[1])/1024 for s in t if s.startswith('VmRSS:')),0.))
  except (FileNotFoundError,ProcessLookupError):continue
 owned={pid}
 while True:
  n={k for k,(parent,_) in rows.items() if parent in owned}-owned
  if not n:break
  owned.update(n)
 return sum(rows.get(k,(0,0))[1] for k in owned)
def stop(p):
 if p.poll() is not None:return
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(timeout=1)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=.5)
def main():
 p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');p.add_argument('--workdir',type=Path,required=True);p.add_argument('--lock',type=Path,default=ROOT/'rebuild.lock');p.add_argument('--child',action='store_true');a=p.parse_args()
 W=a.workdir.resolve();D=json.loads((ROOT/'REPLAY_INPUTS.json').read_text());E=D['expected'];R=W/'human-wholehead-reference-rebuild-20261011';O=R/'volume-v29';F=W/'human-eye-interface-fit-20261011/candidate-v5-terminal-flow'
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 if a.child:
  for cmd in [[sys.executable,str(R/'src/volume_head_v29.py')],[sys.executable,str(R/'src/join_original_tissue_eye_faired_v4.py'),'--source',str(O/'skin.npz'),'--output',str(O/'integrated'),'--eye-patch-root',str(F)]]:subprocess.run(cmd,cwd=W,env=env,check=True)
  checks={name:{'expected_sha256':E[key],'actual_sha256':sha(O/'integrated'/name)} for name,key in [('head.npz','head_sha256'),('anatomical_semantics.npz','semantics_sha256')]}
  for c in checks.values():c['exact_byte_match']=c['actual_sha256']==c['expected_sha256']
  assert sha(O/'skin.npz')==E['body_sha256'],'Body field replay differs'
  result={'checkpoint':'orbital body29 / terminal eye5','outputs':checks,'all_exact':all(c['exact_byte_match'] for c in checks.values()),'scope':'Numerical source replay only; not anatomy, collision, mechanics or realism acceptance'}
  (W/'REPLAY_RECEIPT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
  if not result['all_exact']:raise SystemExit(2)
  if a.render:
   protocol=W/'human-appearance-recovery-20261011/work/free-topology-clay-protocol/run_three.py';controls=W/'human-appearance-recovery-20261011/work/whole-face-clay-gate'
   subprocess.run([sys.executable,str(protocol),'--source',str(O/'integrated/head.npz'),'--sha256',E['head_sha256'],'--controls-root',str(controls),'--output',str(W/'replayed-clay'),'--label','Original orbital body29 / terminal eye5 | unselected','--lock',str(a.lock.resolve()),'--inside-lock','--inherit-process-group'],env=env,cwd=W,check=True)
  return
 if W.exists():raise SystemExit('Use a new workdir; existing files are never overwritten')
 for rel,want in D['files'].items():
  if sha(ROOT/rel)!=want:raise SystemExit(f'Input hash mismatch: {rel}')
 W.mkdir(parents=True)
 for rel in D['files']:
  dst=W/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
 started=time.monotonic();proc=None;rc=None;reason=None;peak=0
 with a.lock.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);admitted=time.monotonic();available=next(float(t.split()[1])/1024 for t in Path('/proc/meminfo').read_text().splitlines() if t.startswith('MemAvailable:'))
  if available<2048:raise SystemExit(f'Memory admission refused: {available:.1f} MiB')
  print('ADMITTED original orbital-body and terminal-eye replay',flush=True)
  try:
   with (W/'rebuild.log').open('w') as f:
    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--workdir',str(W),'--lock',str(a.lock.resolve()),'--child']+(['--render'] if a.render else []),env=env,cwd=W,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    while proc.poll() is None:
     current=rss(proc.pid);peak=max(peak,current)
     if current>900:
      reason='900 MiB sampled owned-tree stop';os.killpg(proc.pid,signal.SIGKILL);proc.wait();break
     if time.monotonic()-admitted>87.5:reason='90-second bounded phase';stop(proc);break
     time.sleep(.05)
    rc=proc.wait()
  except BaseException:
   if proc is not None:stop(proc)
   raise
  finally:
   receipt={'exit_code':rc,'stopped_reason':reason,'seconds_after_admission':round(time.monotonic()-admitted,3),'seconds_including_lock_wait':round(time.monotonic()-started,3),'peak_sampled_rss_mib':round(peak,2),'sample_interval_seconds':.05,'sampled_stop_margin_mib':900,'admission_available_mib':available,'admission_required_mib':2048,'cpu_affinity':sorted(os.sched_getaffinity(0)),'guard_kind':'User-space owned-tree sample, not kernel-hard cap','complete':rc==0 and reason is None}
   (W/'RESOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2));fcntl.flock(lock,fcntl.LOCK_UN);print('RELEASED '+json.dumps(receipt),flush=True)
 if rc or reason:raise SystemExit(rc or 124)
if __name__=='__main__':main()
