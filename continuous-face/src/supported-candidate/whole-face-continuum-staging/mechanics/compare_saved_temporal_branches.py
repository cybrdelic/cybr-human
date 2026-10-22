"""Bounded actual single-step/two-half-step sensitivity, with no new solve."""
from pathlib import Path
import importlib.util,json,time
import numpy as np
from source_transfer import sha,load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle
here=Path(__file__).resolve().parent
base=here/'coarse-barrier-v1/automatic-reconstruction-v1'
full=base/'interval-0006-attempt-004/passive-step0006-NONPHYSICAL-checkpoint.npz'
full_meta=json.loads(full.with_suffix('.json').read_text())
assert sha(full)==full_meta['npz_sha256']
full_report=base/'interval-0006-attempt-004/report.json'
frow=json.loads(full_report.read_text())['runs']['passive']['reports'][-1]
half_report=base/'interval-0007-attempt-006/report.json';hr=json.loads(half_report.read_text())
hrun=hr['runs']['passive'];half=Path(hrun['accepted_npz_path']);assert sha(half)==hrun['accepted_npz_sha256']
with np.load(full,allow_pickle=False) as z:
 qfull=z['nonphysical_iterate_mm'].copy();previous=z['q_previous_mm'].copy();vprevious=z['velocity_previous_mm_s'].copy()
with np.load(half,allow_pickle=False) as z:
 assert np.array_equal(z['positions_mm'][-3],previous) and np.array_equal(z['velocities_mm_s'][-3],vprevious)
 assert abs(z['physical_times_s'][-1]-full_meta['physical_time_s'])<1e-14
 qhalf=z['positions_mm'][-1].copy();vhalf=z['velocities_mm_s'][-1].copy()
vfull=(qfull-previous)/full_meta['dt_s']
started=time.perf_counter()
solid=load_geometry_with_source_fibres(here.parent/'geometry/whole-face-continuum-final.npz',here.parent/'geometry/whole-face-muscle-phases-final.npz')
support,_=load_support_bundle(solid,here/'coarse-physical-site-v1/supports')
gpath=here.parent/'contact/coarse_physical_site_contact_epigraph.py';spec=importlib.util.spec_from_file_location('temporal_guard',gpath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);guard=module.create_guard(solid,support)
dq=guard.C@(qhalf-qfull);dv=guard.C@(vhalf-vfull)
def norms(a):
 n=np.linalg.norm(a,axis=1);i=int(n.argmax());return dict(maximum=float(n[i]),rms=float(np.sqrt(np.mean(n*n))),site_index=i,reference_site_mm=guard.physical_site_positions(solid.X)[i].tolist(),vector=a[i].tolist())
report=dict(scope='Local combined timestep/barrier sensitivity at this corner; not a global temporal/state error bound and not a new solve',
 source_history_q_v_exactly_equal=True,physical_interval_s=[.01,.010625],single_step_dt_s=.000625,two_half_step_dt_s=[.0003125,.0003125],
 single_step_stationary_checkpoint_path=str(full),single_step_stationary_checkpoint_sha256=sha(full),
 single_step_report_path=str(full_report),single_step_report_sha256=sha(full_report),single_step_reconstruction_requalification_pending=True,
 two_half_step_npz_path=str(half),two_half_step_npz_sha256=sha(half),two_half_step_report_path=str(half_report),two_half_step_report_sha256=sha(half_report),
 single_step_barrier_scale=frow['barrier_continuation']['accepted_barrier_scale'],single_step_mu_sensitivity_mm=frow['lower_mu_site_difference_mm'],
 half_step_barrier_scales=[r['barrier_continuation']['accepted_barrier_scale'] for r in hrun['reports'][-2:]],
 half_step_mu_sensitivities_mm=[r['lower_mu_site_difference_mm'] for r in hrun['reports'][-2:]],
 occupied_site_position_difference_mm=norms(dq),occupied_site_velocity_difference_mm_s=norms(dv),
 total_dt_times_velocity_difference_mm=norms(.000625*dv),
 sensitivities_not_subtracted_as_error_bounds=True,wall_seconds=time.perf_counter()-started)
out=here/'coarse-barrier-v1/reconstruction-020-sequence-v1/local-quarter-comparison.json'
if out.exists():raise ValueError('Preserve existing comparison')
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
