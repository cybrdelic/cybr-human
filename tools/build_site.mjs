// Package only the reviewed, currently selected browser runtime. No solver build.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {gzipSync, gunzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const source=JSON.parse(fs.readFileSync(path.join(root,'provenance/public-source.json')));
const runtime=JSON.parse(fs.readFileSync(path.join(root,'output/runtime-current.json')));
const rollback=JSON.parse(fs.readFileSync(path.join(root,'output/runtime-rollback.json')));
const candidateMode=process.argv.includes('--candidate');
const candidate=candidateMode?JSON.parse(fs.readFileSync(path.join(root,'output/runtime-candidate.json'))):null;
const selected=candidate??runtime;
assert.equal(rollback.build,'4bc8b53a8f15b8b62cf6a86e','Deployment must preserve the exact rollback runtime');
let files=source.files.filter(f=>f.path.startsWith(`output/runtime/${rollback.build}/`)||f.path.startsWith('vendor/')||['neutral-tissue.html','src/boot.js','src/runtime_transport.js','output/runtime-current.json'].includes(f.path));
assert.equal(files.length,70,'Reviewed baseline closure changed');
const contentOverrides=new Map();
if(selected.build!==rollback.build){
 const walk=(dir)=>fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(dir,e.name)):[path.join(dir,e.name)]);
 const additional=walk(path.join(root,'output/runtime',selected.build)).map(p=>path.relative(root,p).replaceAll('\\','/'));
 const bootstrap=['neutral-tissue.html','rollback.html','src/boot.js','output/runtime-rollback.json'];
 const names=[...new Set([...files.map(f=>f.path),...additional,...bootstrap])].sort();
 contentOverrides.set('output/runtime-current.json',Buffer.from(JSON.stringify(selected,null,2)+'\n'));
 files=names.map(rel=>({path:rel,sha256:createHash('sha256').update(contentOverrides.get(rel)??fs.readFileSync(path.join(root,rel))).digest('hex')}));
}
const extra=['index.html','ASSET-LICENSES.md','LICENSING.md','LICENSES/GPL-2.0.txt','assets/anatomy/LICENSE.txt','assets/anatomy/UPSTREAM-LICENSE.txt','anatomy-source/License.txt','anatomy-source/Readme.md'];
const dist=path.join(root,candidateMode?'dist-candidate':'dist');
assert.equal(path.dirname(dist),root);
if(fs.existsSync(dist)){
 assert(fs.statSync(path.join(dist,'deployment.json'),{throwIfNoEntry:false})?.isFile(),'Preserve an unrecognized existing dist directory');
 fs.rmSync(dist,{recursive:true});
}
fs.mkdirSync(dist);
const maxAssetBytes=25*1024*1024,records=[],compressed=[];
const hash=b=>createHash('sha256').update(b).digest('hex');
for(const f of files){
 const original=contentOverrides.get(f.path)??fs.readFileSync(path.join(root,f.path));
 assert.equal(hash(original),f.sha256,`Reviewed source hash changed: ${f.path}`);
 const transport=original.length>maxAssetBytes?gzipSync(original,{level:9}):original;
 assert(transport.length<=maxAssetBytes,`Asset still exceeds host limit: ${f.path}`);
 if(transport!==original){assert.deepEqual(gunzipSync(transport),original);compressed.push(f.path);}
 const destination=path.join(dist,f.path);fs.mkdirSync(path.dirname(destination),{recursive:true});fs.writeFileSync(destination,transport);
 records.push({path:f.path,bytes:original.length,sha256:f.sha256,transport_bytes:transport.length,transport_sha256:hash(transport),content_encoding:transport===original?'identity':'gzip'});
}
for(const rel of extra){const to=path.join(dist,rel);fs.mkdirSync(path.dirname(to),{recursive:true});fs.copyFileSync(path.join(root,rel),to);}
const headers=[
 '/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: no-referrer',
 '/*.js\n  Content-Type: text/javascript; charset=utf-8',
 '/*.json\n  Content-Type: application/json',
 '/*.bin\n  Content-Type: application/octet-stream',
 '/*.f32\n  Content-Type: application/octet-stream',
 '/*.glb\n  Content-Type: model/gltf-binary',
 '/output/runtime/*\n  Cache-Control: public, max-age=31536000, immutable',
 '/output/runtime-current.json\n  Cache-Control: no-store',
 '/output/runtime-rollback.json\n  Cache-Control: no-store',
 '/src/boot.js\n  Cache-Control: no-store',
 '/neutral-tissue.html\n  Cache-Control: no-cache',
 '/index.html\n  Cache-Control: no-cache',
 ...compressed.map(rel=>`/${rel}\n  Content-Encoding: gzip`),
];
fs.writeFileSync(path.join(dist,'_headers'),headers.join('\n\n')+'\n');
fs.writeFileSync(path.join(dist,'deployment.json'),JSON.stringify({selected_runtime:selected.build,rollback_runtime:rollback.build,candidate_only:candidateMode,application_files:files.length,application_hash_contract:'Exact decoded file hashes; preserved baseline plus selected immutable closure',application_tree_sha256:hash(files.map(f=>f.path+'\0'+f.sha256+'\n').join('')),lossless_transport_compression:compressed,files:records},null,2)+'\n');
console.log(JSON.stringify({output:path.basename(dist),application_files:files.length,selected_runtime:selected.build,rollback_runtime:rollback.build,candidate_only:candidateMode,losslessly_compressed:compressed,max_transport_bytes:Math.max(...records.map(f=>f.transport_bytes))}));
