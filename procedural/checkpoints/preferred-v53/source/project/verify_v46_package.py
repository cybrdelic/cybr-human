from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
m=json.loads((R/'MANIFEST-v46.json').read_text())
for item in m['files']:
 p=R/item['path']
 if not p.is_file() or p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise SystemExit('Mismatch: '+item['path'])
print('Verified',len(m['files']),'packaged files. External v8 input is checked by the freezer.')
