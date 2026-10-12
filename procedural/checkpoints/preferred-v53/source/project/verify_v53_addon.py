from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'v53_addon_manifest.json').read_text())
for item in manifest['files']:
    rel=Path(item['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
    p=root/rel
    if not p.is_file() or p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Package mismatch: '+str(rel))
print('Verified original v53 source add-on; geometry reproduction is a separate command.')
