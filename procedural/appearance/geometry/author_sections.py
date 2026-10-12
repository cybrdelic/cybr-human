"""Evaluate the editable optional finite-section generators on fixed host interfaces.
These local surfaces alone are not a newly admitted complete head.
"""
from pathlib import Path
import argparse,importlib.util,numpy as np,json,hashlib
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--variant',choices=['orbital-v2','margin-v3'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
source=R/a.variant/'authoring/orbital_surface.py';spec=importlib.util.spec_from_file_location('finite_sections',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=R/'orbital-v2'
records={}
for side,sign in [('right',1),('left',-1)]:
 data=m.build(side,sign);out=a.output/(side+'_orbital.npz');np.savez_compressed(out,**data);records[side]={'vertices':len(data['vertices']),'triangles':len(data['triangles']),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(a.output/'SECTION_REPORT.json').write_text(json.dumps({'variant':a.variant,'scope':'Local procedural section evaluation only; edited sections require fresh assembly, topology, collision and optical checks','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'surfaces':records},indent=2)+'\n');print(json.dumps(records))
