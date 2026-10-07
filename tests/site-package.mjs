// Exercise actual HTTP decoding, module MIME and worker asset paths without GPU use.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createRuntimeFetch} from '../src/runtime_transport.js';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),dist=path.join(root,'dist');
const manifest=JSON.parse(fs.readFileSync(path.join(dist,'deployment.json')));
const blocks=fs.readFileSync(path.join(dist,'_headers'),'utf8').trim().split(/\n\s*\n/).map(block=>{const [pattern,...lines]=block.split('\n');return {pattern,headers:Object.fromEntries(lines.map(line=>{const at=line.indexOf(':');return [line.slice(0,at).trim(),line.slice(at+1).trim()];}))};});
const matches=(pattern,url)=>new RegExp('^'+pattern.split('*').map(p=>p.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('.*')+'$').test(url);
let serveEncodingHeader=true;
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost').pathname,rel=url==='/'?'index.html':url.slice(1);
 if(!['index.html',...manifest.files.map(f=>f.path)].includes(rel)){res.writeHead(404);return res.end();}
 const headers={};for(const rule of blocks)if(matches(rule.pattern,url))Object.assign(headers,rule.headers);
 if(!serveEncodingHeader)delete headers['Content-Encoding'];
 if(rel.endsWith('.html'))headers['Content-Type']='text/html; charset=utf-8';
 res.writeHead(200,headers);res.end(fs.readFileSync(path.join(dist,rel)));
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
try{
 const origin=`http://127.0.0.1:${server.address().port}`;
 for(const file of manifest.files){
  const response=await fetch(origin+'/'+file.path);assert.equal(response.status,200);
  const body=Buffer.from(await response.arrayBuffer());
  assert.equal(body.length,file.bytes);assert.equal(createHash('sha256').update(body).digest('hex'),file.sha256,file.path);
  if(file.path.endsWith('.js'))assert.match(response.headers.get('content-type'),/^text\/javascript/);
  assert.equal(response.headers.get('content-encoding')??'identity',file.content_encoding);
 }
 const pointer=await (await fetch(origin+'/output/runtime-current.json')).json();assert.equal(pointer.build,manifest.selected_runtime);
 assert(manifest.files.some(f=>f.path.endsWith('/src/tissue_cpu_worker.js')));
 assert(manifest.files.some(f=>f.path.endsWith('/src/wrinkle_worker.js')));
 assert.equal((await fetch(origin+'/output/runtime-candidate.json')).status,404);
 assert.equal((await fetch(origin+'/.env')).status,404);
 assert.equal((await fetch(origin+'/output/verification/report.json')).status,404);
 assert.equal((await fetch(origin+'/')).status,200);
 // Reproduce the production gateway dropping Content-Encoding from gzip bytes.
 const asset=manifest.files.find(f=>f.content_encoding==='gzip');assert(asset);
 for(const encodingHeader of [true,false]){
  serveEncodingHeader=encodingHeader;
  const normalized=createRuntimeFetch(fetch,origin+'/'+asset.path);
  const response=await normalized(origin+'/'+asset.path);
  const bytes=Buffer.from(await response.arrayBuffer());
  assert.equal(bytes.length,asset.bytes);
  assert.equal(createHash('sha256').update(bytes).digest('hex'),asset.sha256);
 }
 const untouched=await createRuntimeFetch(fetch,origin+'/'+asset.path)(origin+'/output/runtime-current.json');assert.equal(untouched.headers.get('content-type'),'application/json');
 console.log(JSON.stringify({passed:true,http_decoded_application_hashes:manifest.files.length,module_mime:'passed',same_origin_worker:'present',inactive_and_private_paths:'404',selected_runtime:pointer.build}));
}finally{await new Promise(resolve=>server.close(resolve));}
