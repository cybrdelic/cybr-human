#!/usr/bin/env python3
"""Copy source and pinned inputs for the limited original right-cheek replay."""
from pathlib import Path
import argparse,json,hashlib,shutil
ROOT=Path(__file__).resolve().parent;P='human-preferred-v53-mechanics-20261011';M='human-mechanics-20261010';HEAD='user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';SHA='545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--head',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();head=a.head.resolve()
if out.exists() or out.is_relative_to(ROOT):raise SystemExit('Use a new output outside the checkpoint')
if sha(head)!=SHA:raise SystemExit('Wrong preferred head')
files=json.loads((ROOT/'PAYLOAD.json').read_text())['files'];prefixes=[P+'/src/',P+'/reference-anatomy/',P+'/rest-targets-v1/',P+'/rest-fit-v4/',M+'/src/',M+'/reference-source/',M+'/output/',M+'/inputs/','user-human-preferred/']
chosen={rel:r for rel,r in files.items() if any(rel.startswith(x) for x in prefixes) or rel==P+'/audit-attachment-reactions-v1/extract_reactions.mjs'}
for rel,r in chosen.items():
 if sha(ROOT/'workspace'/rel)!=r['sha256']:raise SystemExit('Input changed: '+rel)
out.mkdir(parents=True)
for rel in chosen:
 dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'workspace'/rel,dst)
(out/'checkpoint_tools').mkdir();shutil.copy2(ROOT/'checkpoint_tools/run_selected_transfer.py',out/'checkpoint_tools/run_selected_transfer.py');dst=out/HEAD;dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(head)
(out/'REPLAY_COMMANDS.txt').write_text('''# From this fresh root, run each phase under the documented CPU/RSS guard.\nB=human-preferred-v53-mechanics-20261011\nblender -b -t 2 --python "$B/src/build_preferred_cheek_volume_v9.py"\npython "$B/src/create_cheek_case_v3.py"\nnode "$B/src/solve_regional_cheek_pcg500.mjs" "$B/cheek-case-v3" "$B/cheek-equilibrium-v3" neutral-0 0\nnode "$B/src/solve_regional_cheek_pcg500.mjs" "$B/cheek-case-v3" "$B/cheek-equilibrium-v3" right-02 .02 "$B/cheek-equilibrium-v3/neutral-0-state.json"\nblender -b -t 2 --python "$B/src/bind_preferred_cheek_skin_v5.py"\npython checkpoint_tools/run_selected_transfer.py --states neutral-0 right-02\n''')
print(json.dumps({'prepared':str(out),'files':len(chosen),'head_exact':True,'numerical_execution':False}))
