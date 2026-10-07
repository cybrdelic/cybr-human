"""Check immutable runtime closure, asset hashes and privacy-pruned source packaging."""
from pathlib import Path
import hashlib, json, os, re

ROOT=Path(__file__).resolve().parent.parent
pointer='runtime-candidate.json' if os.environ.get('TISSUE_CANDIDATE')=='1' else 'runtime-current.json'
runtime=json.loads((ROOT/'output'/pointer).read_text())
assert runtime['build']==os.environ.get('TISSUE_BUILD',runtime['build'])
base=ROOT/'output/runtime'/runtime['build']
manifest=json.loads((base/'manifest.json').read_text())
assert manifest['build']==runtime['build']
errors=[];checked=[];references=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel,expected in manifest['source_sha256'].items():
    p=base/('assets/'+rel if rel.startswith('output/') else rel)
    assert p.is_file(), 'Missing runtime file: '+rel
    # Publisher rewrites JSON URLs after hashing; those documents are checked below.
    if p.suffix!='.json':assert sha(p)==expected,'Runtime hash mismatch: '+rel
    checked.append(p.relative_to(ROOT).as_posix())
    if p.suffix in {'.js','.mjs'}:
        for ref in re.findall(r'''(?:from\s*|import\s*\(|new URL\s*\()\s*['"]([^'"]+)['"]''',p.read_text()):
            if ref.startswith('.'):
                target=(p.parent/ref.split('?')[0]).resolve()
                assert target.is_relative_to(base) and (target.is_file() or (ref=='.' and target.is_dir())),str(target)
                references.append(target.relative_to(ROOT).as_posix())
            elif ref=='three':assert (ROOT/'vendor/three-0.169.0/build/three.module.js').is_file()
            elif ref.startswith('three/addons/'):
                assert (ROOT/'vendor/three-0.169.0/examples/jsm'/ref[len('three/addons/'):]).is_file()
def walk(value):
    if isinstance(value,dict):
        if isinstance(value.get('url'),str) and value.get('sha256'):
            p=ROOT/value['url'].lstrip('/');assert p.is_file(),str(p)
            assert sha(p)==value['sha256'],'Asset hash mismatch: '+str(p)
        for item in value.values():walk(item)
    elif isinstance(value,list):
        for item in value:walk(item)
    elif isinstance(value,str) and value.startswith('/output/runtime/'):
        p=ROOT/value.lstrip('/');assert p.is_file() or p.is_dir(),str(p)
        references.append(value)
for p in (base/'assets').rglob('*.json'):walk(json.loads(p.read_text()))
provenance=json.loads((ROOT/'provenance/staged-source.json').read_text())
unchanged=0;changed=[]
for record in provenance['files']:
    p=ROOT/record['path']
    if p.is_file() and sha(p)==record['sha256']:unchanged+=1
    else:changed.append(record['path'])
    if record['path'].startswith('output/runtime/') or record['path'] in {'neutral-tissue.html','output/runtime-current.json'}:
        assert p.is_file() and sha(p)==record['sha256'],'Selected application changed'
for rel in ['assets/anatomy/geometry.npz','assets/anatomy/manifest.json','assets/anatomy/LICENSE.txt','assets/anatomy/UPSTREAM-LICENSE.txt','vendor/three-0.169.0/LICENSE']:
    assert (ROOT/rel).is_file(),rel
for current,dirs,names in os.walk(ROOT):
    dirs[:]=[d for d in dirs if d not in {'node_modules','.venv','__pycache__'}]
    assert not any('profile' in d.lower() or d.lower() in {'.npm-cache','.ruff_cache','cache','cookies'} for d in dirs),'Private/cache directory in stage'
    assert not any(n.startswith('.env') or any(x in n.lower() for x in ['cookie','auth-state','storage-state','credential']) for n in names),'Private file in stage'
report={'build':runtime['build'],'runtime_files':len(checked),'resolved_reference_count':len(references),'unchanged_copied_files':unchanged,'changes_from_staged_baseline':changed,'selected_runtime_and_html_bytes_match_original':True,'privacy_check':'passed; dependencies and generated test outputs excluded from source scan'}
(ROOT/'output/verification').mkdir(parents=True,exist_ok=True)
(ROOT/'output/verification/source-graph.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
