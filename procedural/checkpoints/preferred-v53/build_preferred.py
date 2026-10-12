#!/usr/bin/env python3
"""Rebuild the exact preferred rest head from bundled original source/arrays."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();out=a.output.resolve();prep=out.parent/(out.name+'-inputs')
 if out.exists() or prep.exists():raise SystemExit('Both output and companion -inputs paths must be new')
 pins=json.loads((ROOT/'INPUTS.json').read_text());parts=pins['parts']
 if any(Path(q['name']).name!=q['name'] or not q['name'].startswith('input-arrays.zip.part') for q in parts):raise SystemExit('Unsafe input part name')
 for q in parts:
  p=ROOT/q['name']
  if not p.is_file() or p.stat().st_size!=q['bytes'] or sha(p)!=q['sha256']:raise SystemExit('Input part hash mismatch: '+q['name'])
 prep.mkdir(parents=True);archive=prep/'input-arrays.zip'
 with archive.open('wb') as output:
  for q in parts:
   with (ROOT/q['name']).open('rb') as f:shutil.copyfileobj(f,output)
 if archive.stat().st_size!=pins['archive_bytes'] or sha(archive)!=pins['archive_sha256']:raise SystemExit('Reassembled input archive hash mismatch')
 with zipfile.ZipFile(archive) as z:
  if sorted(z.namelist())!=sorted(pins['members']):raise SystemExit('Unexpected input member')
  if sum(q.file_size for q in z.infolist())!=pins['decompressed_bytes']:raise SystemExit('Input size mismatch')
  shutil.copytree(ROOT/'source',prep/'source');(prep/'positions').mkdir();(prep/'source/input').mkdir()
  for name,want in pins['members'].items():
   dest=(prep/'positions' if name=='vertices.npy' else prep/'source/input')/name
   with z.open(name) as f,dest.open('wb') as g:shutil.copyfileobj(f,g)
   if sha(dest)!=want:raise SystemExit('Input member hash mismatch: '+name)
 command=[sys.executable,str(prep/'source/restore_preferred.py'),'--positions',str(prep/'positions'),'--output',str(out),'--lock',str(a.lock.resolve())]
 (prep/'COMMAND.json').write_text(json.dumps(command,indent=2)+'\n')
 if a.prepare_only:print(json.dumps({'prepared':str(prep),'numerical_replay_executed':False,'command':command}));return
 subprocess.run(command,check=True)
 receipt=json.loads((out/'REPLAY_RECEIPT.json').read_text())
 if not receipt['all_exact']:raise SystemExit('Exact selected output check failed')
 print(json.dumps({'head':str(out/'project/output/coordinated-orbital-scale-v53/head.npz'),'semantics':str(out/'project/output/coordinated-orbital-scale-v53/anatomical_semantics/authored_lip_lid_semantics.npz'),'all_six_stage_hashes_exact':True}))
if __name__=='__main__':main()
