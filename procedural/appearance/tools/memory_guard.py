#!/usr/bin/env python3
"""Sampled aggregate-RSS guard for an owned subprocess tree. Invoke inside flock.
This is a user-space 0.25-second guard, not a kernel hard limit. Shared resident
pages may be counted twice across processes. Raw render logs remain untouched.
"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--max-rss-mib',type=float,default=1800);p.add_argument('--min-available-mib',type=float,default=2048);p.add_argument('--nominal-maximum-mib',type=float,default=1024);p.add_argument('--timeout-seconds',type=float,default=360);p.add_argument('--poll-seconds',type=float,default=.05);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
assert cmd and a.max_rss_mib>0
available_mib=next(float(x.split()[1])/1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
if available_mib<a.min_available_mib:
    print('CYBR_MEMORY_ADMISSION_REFUSED '+json.dumps({'available_mib':available_mib,'required_mib':a.min_available_mib}),flush=True);sys.exit(75)
print('CYBR_CONCURRENT_ADMISSION '+json.dumps({'available_mib':available_mib,'required_mib':a.min_available_mib,'sampled_stop_margin_mib':a.max_rss_mib,'nominal_maximum_mib':a.nominal_maximum_mib,'kernel_hard_cap':False,'concurrent_load':'Other authorized CPU jobs may be active; no speed benchmark claim'}),flush=True)
proc=subprocess.Popen(cmd,start_new_session=True);peak=0.;stopped=False;timed_out=False;start=time.monotonic()
def tree_rss(pid):
    # This sandbox omits /proc/PID/task/PID/children. Read PPid from ordinary
    # status records instead, retaining only descendants of our owned child.
    rows={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        try:
            data=entry.joinpath('status').read_text().splitlines()
            ppid=next((int(x.split()[1]) for x in data if x.startswith('PPid:')),0)
            rss=next((float(x.split()[1])/1024 for x in data if x.startswith('VmRSS:')),0.)
            rows[int(entry.name)]=(ppid,rss)
        except (FileNotFoundError,ProcessLookupError):continue
    owned={pid};changed=True
    while changed:
        found={k for k,(parent,_) in rows.items() if parent in owned}-owned
        changed=bool(found);owned.update(found)
    return sum(rows.get(k,(0,0.))[1] for k in owned)

def stop():
    if proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
try:
    while proc.poll() is None:
        if time.monotonic()-start>=a.timeout_seconds:
            timed_out=True;print('CYBR_TIME_GUARD_STOP',flush=True);stop();break
        rss=tree_rss(proc.pid);peak=max(peak,rss)
        if rss>a.max_rss_mib:
            stopped=True;print('CYBR_RSS_GUARD_STOP '+json.dumps({'observed_aggregate_rss_mib':rss,'stop_margin_mib':a.max_rss_mib,'nominal_maximum_mib':a.nominal_maximum_mib}),flush=True)
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait();break
        time.sleep(a.poll_seconds)
    rc=proc.wait()
except BaseException:
    stop();raise
finally:
    print('CYBR_RSS_GUARD '+json.dumps({'peak_sampled_aggregate_rss_mib':round(peak,2),'sampled_stop_margin_mib':a.max_rss_mib,'nominal_maximum_mib':a.nominal_maximum_mib,'kernel_hard_cap':False,'admission_available_mib':available_mib,'concurrent_load_possible':True,'interval_seconds':a.poll_seconds,'stopped_for_rss':stopped,'stopped_for_time':timed_out,'timeout_seconds':a.timeout_seconds,'guard_measurement_valid':peak>0,'elapsed_seconds':round(time.monotonic()-start,2),'scope':'Owned Blender tree, including optional OIDN child; user-space sampled guard, not kernel hard limit'}),flush=True)
sys.exit(137 if stopped else (124 if timed_out else rc))
