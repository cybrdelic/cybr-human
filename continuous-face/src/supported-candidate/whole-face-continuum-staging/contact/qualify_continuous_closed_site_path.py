"""Bounded saved 10-ms chord failure and safe prefix, no dynamics."""
from pathlib import Path
import json,time
import numpy as np
from coarse_physical_site_contact_epigraph import create_guard
from coarse_physical_site_contact import sha
from source_transfer import load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle
from continuous_closed_site_path import certify_closed_owner_site_chord

HERE=Path(__file__).resolve().parent
started=time.perf_counter()
solid=load_geometry_with_source_fibres(HERE.parent/'geometry/whole-face-continuum-final.npz',HERE.parent/'geometry/whole-face-muscle-phases-final.npz')
supports,_=load_support_bundle(solid,HERE.parent/'mechanics/coarse-physical-site-v1/supports')
guard=create_guard(solid,supports)
source=HERE.parent/'mechanics/coarse-barrier-v1/first-step-v5/passive-step0001-NONPHYSICAL-checkpoint.npz'
packet=np.load(source);q0=packet['q_previous_mm'];q1=packet['nonphysical_iterate_mm']
failed=certify_closed_owner_site_chord(guard,q0,q1)
print(json.dumps(dict(case='actual_failed_10ms_chord',result=failed)),flush=True)
(HERE/'continuous-closed-site-path-first-result.json').write_text(json.dumps(failed,indent=2)+'\n')
assert not failed['feasible'] and failed.get('offenders')
prefix=certify_closed_owner_site_chord(guard,q0,q0+.25*(q1-q0))
assert prefix['feasible']
print(json.dumps(dict(case='diagnostic_quarter_prefix_not_a_physical_step',result=prefix)),flush=True)
report=dict(qualified=True,scope='One actual failed chord and one geometric prefix only; no dynamics or accepted frames',
    source_checkpoint_sha256=sha(source),helper_sha256=sha(HERE/'continuous_closed_site_path.py'),
    failed_actual_chord=failed,safe_diagnostic_prefix=prefix,total_seconds=time.perf_counter()-started)
(HERE/'continuous-closed-site-path-qualification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
