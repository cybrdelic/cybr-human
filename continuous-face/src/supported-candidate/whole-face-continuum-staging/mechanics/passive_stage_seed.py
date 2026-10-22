"""Optional first-mu passive coordinates, never physical state or history.

The final independent passive audit must bind this artifact through its frozen
report before an active solve may use it as a q-only initial iterate. A rejected
seed is not an acceptance failure: callers retain their independently valid
existing numerical guess and record the rejection.
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from accepted_history import validate_resume_interval_audits
from report_storage import read_report

SCHEMA = 'nonphysical_first_passive_stage_q_v1'
MODEL_FIELDS = (
    'source_domain_sha256', 'source_phase_sha256', 'source_support_sha256',
    'source_support_bundle_sha256', 'guard_module_sha256', 'activation_ramp_s',
    'physical_mass_kg', 'damping_per_second', 'gravity_mm_s2', 'contact_mode',
    'contact_enforcement', 'stationarity_accuracy_policy', 'barrier_scales',
    'lower_mu_site_tolerance_mm', 'reconstruction_contract',
    'physical_endpoint_penetration_cap_mm', 'interframe_reconstruction_allowance_mm',
    'certified_interframe_penetration_cap_mm', 'stage_q_seed_source_sha256',
)
CONTACT_FIELDS = (
    'clearance_mm', 'penalty_stiffness_N_per_mm4',
    'maximum_allowed_site_penetration_mm', 'guard_scope',
    'maximum_trial_site_increment_mm', 'maximum_path_sample_spacing_mm',
    'numerical_barrier',
)
FLAGS = dict(nonphysical_internal_iterate=True, usable_as_accepted_frame=False,
             final_physical_path_certified=False)


class StageSeedRejected(ValueError):
    """The optional artifact has an integrity or interval-mapping mismatch."""


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha_bytes(Path(path).read_bytes())


def array_sha(value):
    """Identical convention to the independent interval auditor."""
    value = np.ascontiguousarray(value)
    return sha_bytes(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def model_sha(report):
    # Bind stable physical parameters and local implementation hashes, but not
    # process timings, paths, active/passive histories or live-report contents.
    model = {key: report[key] for key in MODEL_FIELDS}
    model['implementation'] = {key: value for key, value in report.items()
        if key.endswith('_source_sha256')}
    contact = report['coarse_contact_provenance']
    model['contact'] = {key: contact[key] for key in CONTACT_FIELDS}
    model['contact'].update({key: value for key, value in contact.items()
        if key.endswith('_sha256')})
    return sha_bytes(canonical(model).encode())


class FirstPassiveStageCapture:
    """Callable that writes the first converged scale once, without overwriting.

    Its descriptor must be put into the passive run before the final external
    audit snapshots that report. History coordinates are hashed, never saved.
    """
    def __init__(self, path, *, report, source_initial, previous, velocity,
                 previous_time, target_time, dt, first_scale, prior_audit_sha256):
        self.path = Path(path)
        self.descriptor = None
        self.shape = np.asarray(previous).shape
        self.binding = dict(schema=SCHEMA, case='passive', barrier_stage=0,
            barrier_scale=float(first_scale), dt_s=float(dt),
            previous_time_s=float(previous_time), physical_time_s=float(target_time),
            source_initial_positions_sha256=array_sha(source_initial),
            passive_q_previous_sha256=array_sha(previous),
            passive_velocity_previous_sha256=array_sha(velocity),
            passive_prior_audit_sha256=prior_audit_sha256,
            model_sha256=model_sha(report), **FLAGS)
        if (not np.isfinite([previous_time, target_time, dt, first_scale]).all()
                or not 0 < dt or not 0 < first_scale
                or target_time != previous_time + dt
                or np.asarray(velocity).shape != self.shape):
            raise StageSeedRejected('Invalid capture interval or history shape')

    def __call__(self, coordinates, stage):
        if self.descriptor is not None or stage.get('barrier_stage') != 0:
            return
        if (stage.get('inner_stationarity_converged') is not True
                or stage.get('final_accounting_qualified') is not True
                or any(stage.get(key) != value for key, value in FLAGS.items())
                or stage.get('barrier_scale') != self.binding['barrier_scale']
                or stage.get('dt_s') != self.binding['dt_s']):
            raise StageSeedRejected('First stage is not qualified at the bound dt/mu')
        q = np.array(coordinates, dtype=float, copy=True)
        if q.shape != self.shape or not np.isfinite(q).all():
            raise StageSeedRejected('Malformed first-stage coordinates')
        binding = dict(self.binding, q_sha256=array_sha(q))
        payload = io.BytesIO()
        np.savez_compressed(payload, nonphysical_initial_iterate_mm=q,
                            binding_json=np.array(canonical(binding)))
        data = payload.getvalue()
        # Exclusive creation: never turn a later attempt into the earlier seed.
        with self.path.open('xb') as stream:
            stream.write(data)
        self.descriptor = dict(path=str(self.path.resolve()), npz_sha256=sha_bytes(data),
            binding_sha256=sha_bytes(canonical(binding).encode()), schema=SCHEMA, **FLAGS)


def _require(condition, reason):
    if not condition:
        raise StageSeedRejected(reason)


def audited_first_passive_stage_guess(audit_path, current_report, source_initial,
        *, case, physical_time, dt, first_scale, own_previous, guard, constraints):
    """Return copied q only, or a normal not-used reason; reject corrupt inputs.

    No velocity or previous-state argument is returned or modified. The caller
    still passes its exact own accepted q/v into the unchanged dynamics core.
    """
    evidence = dict(used=False, numerical_guess_only=True, **FLAGS)
    if case != 'zygomaticus_frontalis':
        return None, dict(evidence, reason='not_active_case')
    if audit_path is None:
        return None, dict(evidence, reason='no_final_passive_audit')
    try:
        audit_path = Path(audit_path)
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
        if audit.get('qualified') is not True:
            return None, dict(evidence, reason='final_passive_interval_not_admitted')
        _require(audit.get('case') == 'passive', 'Final audit belongs to a different case')
        _require(audit.get('audited_snapshots_unchanged_during_audit') is True
            and audit.get('contact_sources_unchanged_during_audit') is True,
            'Final audit did not bind unchanged snapshots and sources')
        stored = read_report(audit['mechanics_report_path'],
                             expected_file_sha256=audit['mechanics_report_sha256'])
        frozen = stored.document
        run = frozen['runs']['passive']
        descriptor = run.get('first_stage_q_seed')
        if descriptor is None:
            return None, dict(evidence, reason='first_stage_seed_not_captured')
        validate_resume_interval_audits(frozen, [audit_path], ['passive'])
        _require(file_sha(audit['accepted_npz_snapshot_path']) == audit['accepted_npz_sha256'],
                 'Frozen final passive arrays hash mismatch')
        _require(audit['source_initial_positions_sha256'] == array_sha(source_initial),
                 'Passive source reference differs')
        _require(model_sha(frozen) == model_sha(current_report), 'Passive seed model/source mismatch')
        _require(descriptor.get('schema') == SCHEMA
            and all(descriptor.get(k) == v for k, v in FLAGS.items()),
            'Seed descriptor is not explicitly NONPHYSICAL')
        seed_bytes = Path(descriptor['path']).read_bytes()
        _require(sha_bytes(seed_bytes) == descriptor['npz_sha256'], 'Stage seed NPZ hash mismatch')
        with np.load(io.BytesIO(seed_bytes), allow_pickle=False) as archive:
            _require(set(archive.files) == {'nonphysical_initial_iterate_mm', 'binding_json'},
                     'Stage seed must contain only q and its binding, never velocity/QP/reactions')
            binding = json.loads(str(archive['binding_json'].item()))
            candidate = np.array(archive['nonphysical_initial_iterate_mm'], dtype=float, copy=True)
        _require(sha_bytes(canonical(binding).encode()) == descriptor['binding_sha256'],
                 'Stage seed binding hash mismatch')
        _require(candidate.shape == np.asarray(own_previous).shape
            and np.isfinite(candidate).all() and array_sha(candidate) == binding['q_sha256'],
            'Stage seed coordinates malformed or changed')
        expected = dict(schema=SCHEMA, case='passive', barrier_stage=0,
            barrier_scale=float(first_scale), dt_s=float(dt), physical_time_s=float(physical_time),
            model_sha256=model_sha(current_report),
            source_initial_positions_sha256=array_sha(source_initial), **FLAGS)
        _require(all(binding.get(k) == v for k, v in expected.items()),
                 'Stage seed target/dt/mu/model mapping mismatch')
        intervals = audit['intervals']
        interval = intervals[-1]
        rows = [row for row in run['reports'] if row.get('accepted') is True]
        final_row = rows[-1]
        # Only the LAST final admitted interval may supply a seed, not a stale
        # prefix with identical geometry or a rejected larger-dt attempt.
        _require(run['actual_times_s'][-1] == physical_time
            and interval['t1_s'] == physical_time
            and binding['previous_time_s'] == interval['t0_s']
            and binding['previous_time_s'] + dt == physical_time
            and final_row['dt_s'] == dt and final_row['physical_time_s'] == physical_time
            and frozen['barrier_scales'][0] == first_scale,
            'Final passive audit is for a different target interval/dt/mu')
        _require(binding['passive_q_previous_sha256'] == interval['q0_sha256']
            and binding['passive_velocity_previous_sha256'] == interval['previous_velocity_sha256']
            and binding['passive_prior_audit_sha256'] == audit['reused_prefix_audit_sha256'],
            'Stage seed passive source/history provenance mismatch')
        stages = final_row['barrier_continuation']['stages']
        _require(stages[0]['barrier_scale'] == first_scale
            and stages[0]['inner_stationarity_converged'] is True,
            'Final passive report does not bind the converged first scale')
        _require(file_sha(audit_path) == sha_bytes(audit_bytes),
                 'Final passive audit changed during seed validation')
        # Validation uses current guards/supports. It does not import passive
        # history, make a physical interval claim, or bypass active acceptance.
        if constraints is not None:
            error = constraints.error(candidate)
            if not np.isfinite(error) or error > 1e-8:
                return None, dict(evidence, reason='current_support_infeasible')
        if guard.endpoint(candidate).get('feasible') is not True:
            return None, dict(evidence, reason='current_endpoint_infeasible')
        return candidate, dict(evidence, used=True, reason='matched_first_passive_scale_q_only',
            audit_path=str(audit_path.resolve()), audit_sha256=sha_bytes(audit_bytes),
            seed_npz_sha256=descriptor['npz_sha256'], barrier_scale=float(first_scale),
            physical_time_s=float(physical_time), dt_s=float(dt))
    except StageSeedRejected:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, zipfile.BadZipFile) as exc:
        raise StageSeedRejected('Invalid optional passive stage seed: '+str(exc)) from exc
