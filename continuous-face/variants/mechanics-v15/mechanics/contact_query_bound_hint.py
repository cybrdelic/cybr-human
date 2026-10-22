"""Optional audited, common-anchor distance hints. Never physical/checkpoint state.

Only already-eligible rolling exterior certificates are translated. No query,
force, witness, multiplier, KKT state, or substitute q/v is serialized. Callers
must first perform their ordinary accepted-history and external-audit checks.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from report_storage import load_report

SCHEMA = 'accepted_exterior_distance_hint_v1'
FIELD = 'contact_query_bound_hint'
MAX_SITES = 10_000_000
DESCRIPTOR_KEYS = {'schema', 'helper_sha256', 'path', 'sha256', 'bytes', 'identity', 'retained_sites'}
OPTIONAL_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, zipfile.BadZipFile)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_identity(value):
    a = np.ascontiguousarray(value)
    return dict(dtype=a.dtype.str, shape=list(a.shape),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def _source(obj):
    # The driver's factory module is loaded with exec_module, without a
    # sys.modules entry. Methods still identify that actual source file.
    if inspect.isclass(obj):
        for value in obj.__dict__.values():
            if isinstance(value, (classmethod, staticmethod)):
                value = value.__func__
            if inspect.isfunction(value):
                obj = value
                break
    path = inspect.getsourcefile(obj)
    if path is None:
        raise ValueError('Query hint source identity unavailable')
    return file_sha(path)


def runtime_identity(guard):
    """Bind actual loaded implementations and exact mapping/reference bytes."""
    rolling = guard.rolling
    native = guard.rigid.native
    query_module = inspect.getmodule(type(native))
    sources = {'guard_mro:'+cls.__name__: _source(cls)
               for cls in type(guard).__mro__ if cls is not object}
    sources.update(native_query=_source(type(native)), rolling=_source(type(rolling)),
                   finite_pairs=_source(query_module._finite_pairs),
                   closest_surface=_source(query_module.closest_surface),
                   surface_index=_source(query_module.SurfaceIndex),
                   recorded_query=_source(type(guard.rigid)))
    # These are computed by the ordinary guard factory from its actual inputs.
    provenance = guard.provenance
    provenance_keys = ('source_domain_sha256', 'source_cells_sha256',
                       'reference_band_cache_sha256', 'native_source_sha256')
    native_keys = ('source_sha256', 'metadata_sha256', 'coordinate_scale', 'inventory_rule')
    C = guard.C
    if C.format != 'csr':
        raise ValueError('Exact CSR mapping required')
    return dict(native_geometry_and_closed_owners=rolling.token,
        native_input={key: native.report[key] for key in native_keys},
        inputs={key: provenance[key] for key in provenance_keys},
        sources=sources, C=dict(shape=list(C.shape), indptr=array_identity(C.indptr),
            indices=array_identity(C.indices), data=array_identity(C.data)),
        P0=array_identity(guard.P0), reference=array_identity(rolling.reference),
        reference_lower=array_identity(rolling.reference_lower),
        adapter_reference_lower=array_identity(guard.lower),
        h=float(rolling.h), refresh_radius=float(rolling.refresh_radius),
        units='mm', mm_per_L0=float(guard.solid.mm_per_L0))


def _authoritative_points(guard, q, v, run):
    """Not optional failures: never hide live geometry or accepted-state errors."""
    guard.rolling._check_geometry()
    q, v = np.asarray(q), np.asarray(v)
    if (q.shape != guard.solid.X.shape or v.shape != q.shape
            or q.dtype != np.dtype('float64') or v.dtype != np.dtype('float64')
            or not np.isfinite(q).all() or not np.isfinite(v).all()):
        raise ValueError('Authoritative accepted q/v is malformed')
    points = guard.C@q
    guard.rolling._points(points)
    if (run['accepted_frames'] != len(run['actual_times_s']) or not run['actual_times_s']
            or not np.isfinite(run['actual_times_s']).all()):
        raise ValueError('Authoritative accepted history is malformed')
    if file_sha(run['accepted_npz_path']) != run['accepted_npz_sha256']:
        raise ValueError('Authoritative accepted NPZ changed')
    return points


def _identity(guard, case, run, q, v, points):
    return dict(runtime=runtime_identity(guard), case=case,
        accepted_npz_sha256=run['accepted_npz_sha256'],
        accepted_frames=run['accepted_frames'], actual_times_s=run['actual_times_s'],
        q=array_identity(q), v=array_identity(v), accepted_sites=array_identity(points))


def _audited_descriptor(evidence, case, run):
    """Read only the required passing own-case audit's immutable report.

    Receipt/history failures propagate, like the ordinary required validator.
    Missing legacy snapshot metadata means no optional hint, never a sidecar scan.
    """
    if evidence is None:
        return None
    if file_sha(evidence['path']) != evidence['sha256']:
        raise ValueError('Required own-case audit changed after history validation')
    receipt = json.loads(Path(evidence['path']).read_text())
    if (receipt.get('schema') != 'continuous_closed_site_external_audit_v1'
            or receipt.get('qualified') is not True or receipt.get('case') != case
            or receipt.get('accepted_npz_sha256') != run['accepted_npz_sha256']
            or receipt.get('accepted_frames_audited') != run['accepted_frames']
            or receipt.get('physical_times_s') != run['actual_times_s']
            or receipt.get('saved_velocity_history_consistent') is not True):
        raise ValueError('Required own-case audit no longer binds accepted history')
    if 'mechanics_report_path' not in receipt or 'mechanics_report_sha256' not in receipt:
        return None
    report = load_report(receipt['mechanics_report_path'],
                         expected_file_sha256=receipt['mechanics_report_sha256'])
    own = report['runs'][case]
    for key in ('accepted_npz_sha256', 'accepted_frames', 'actual_times_s'):
        if own[key] != run[key]:
            raise ValueError('Immutable own-case report differs from accepted history')
    return own.get(FIELD)


def translated_payload(rolling, points):
    """Same displacement and numerical padding as the unchanged rolling screen."""
    pad = 1024*np.finfo(float).eps*(np.linalg.norm(points, axis=1)
          + np.linalg.norm(rolling.anchors, axis=1) + rolling.refresh_radius + 1)
    translated = rolling.anchor_lower-np.linalg.norm(points-rolling.anchors, axis=1)-pad
    keep = (rolling.eligible & np.isfinite(translated) & (translated > rolling.h)
            & (translated <= rolling.refresh_radius))
    lower = np.zeros(len(points), dtype=np.float64)
    lower[keep] = translated[keep]
    return lower, np.asarray(keep, dtype=bool)


def _payload_bytes(lower, eligible):
    buffer = io.BytesIO()
    np.savez(buffer, lower=lower, eligible=eligible)
    return buffer.getvalue()


def _read_payload(descriptor, identity, count, h, radius):
    if not 0 < count <= MAX_SITES:
        raise ValueError('Unsupported query hint site count')
    if (not isinstance(descriptor, dict) or set(descriptor) != DESCRIPTOR_KEYS
            or descriptor['schema'] != SCHEMA
            or descriptor['helper_sha256'] != file_sha(__file__)
            or descriptor['identity'] != identity):
        raise ValueError('Query hint schema/source/geometry/case/history mismatch')
    limit = 9*count+4096
    size = descriptor['bytes']
    if type(size) is not int or not 0 < size <= limit:
        raise ValueError('Query hint size exceeds bounded payload')
    with open(descriptor['path'], 'rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
        raise ValueError('Query hint payload length/checksum mismatch')
    # Validate ZIP entries AND NPY headers before NumPy allocates any arrays.
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) != 2 or {x.filename for x in entries} != {'lower.npy', 'eligible.npy'}:
            raise ValueError('Unexpected query hint payload keys')
        for entry in entries:
            expected = np.dtype('float64') if entry.filename == 'lower.npy' else np.dtype(bool)
            if (entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits & 1
                    or entry.file_size > count*expected.itemsize+512):
                raise ValueError('Unexpected query hint storage/entry length')
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError('Unsupported query hint NPY version')
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=512)
                if shape != (count,) or fortran or dtype != expected:
                    raise ValueError('Query hint array shape/dtype mismatch')
    with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
        lower, eligible = payload['lower'].copy(), payload['eligible'].copy()
    if (not np.isfinite(lower).all() or np.any(lower[~eligible] != 0)
            or np.any(lower[eligible] <= h) or np.any(lower[eligible] > radius)
            or type(descriptor['retained_sites']) is not int
            or descriptor['retained_sites'] != int(eligible.sum())):
        raise ValueError('Query hint retained bounds malformed')
    return lower, eligible


def _publish(path, raw):
    """Atomic complete publication without overwriting any earlier artifact."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.contact-query-hint-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


class ContactQueryHintSession:
    """At most one import; the original shared live guard is never cleared."""
    def __init__(self, guard, enabled=False):
        self.guard = guard
        self.enabled = bool(enabled)
        self.first_case_seen = False

    def import_once(self, *, case, run, q, v, audit_evidence):
        if not self.enabled:
            return None
        if self.first_case_seen:
            return dict(imported=False, reason='preserve_shared_in_process_guard')
        self.first_case_seen = True
        if not run.get('accepted_frames'):
            return dict(imported=False, reason='no_own_accepted_anchor')
        points = _authoritative_points(self.guard, q, v, run)
        # The pre-loop diagnostic preflight may already have learned useful
        # bounds. Do not replace these or expand scope to accelerate preflight.
        if np.any(self.guard.rolling.eligible):
            return dict(imported=False, reason='preserve_existing_live_anchors')
        descriptor = _audited_descriptor(audit_evidence, case, run)
        if descriptor is None:
            return dict(imported=False, reason='no_audited_hint_cold_fallback')
        try:
            identity = _identity(self.guard, case, run, q, v, points)
            lower, eligible = _read_payload(descriptor, identity, len(points),
                self.guard.rolling.h, self.guard.rolling.refresh_radius)
            anchors = points.copy()
        except OPTIONAL_ERRORS as exc:
            return dict(imported=False, reason='optional_hint_rejected_cold_fallback', detail=str(exc))
        # All checks and allocations complete before any live assignment.
        self.guard.rolling.anchors = anchors
        self.guard.rolling.anchor_lower = lower
        self.guard.rolling.eligible = eligible
        return dict(imported=True, retained_sites=int(eligible.sum()),
                    payload_sha256=descriptor['sha256'])

    def export(self, *, path, case, run, q, v):
        if not self.enabled:
            return None, None
        points = _authoritative_points(self.guard, q, v, run)
        try:
            if not 0 < len(points) <= MAX_SITES:
                raise ValueError('Unsupported query hint site count')
            identity = _identity(self.guard, case, run, q, v, points)
            lower, eligible = translated_payload(self.guard.rolling, points)
            raw = _payload_bytes(lower, eligible)
            descriptor = dict(schema=SCHEMA, helper_sha256=file_sha(__file__),
                path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw), identity=identity, retained_sites=int(eligible.sum()))
            _publish(path, raw)
            return descriptor, dict(exported=True, retained_sites=int(eligible.sum()), bytes=len(raw))
        except OPTIONAL_ERRORS as exc:
            return None, dict(exported=False, reason='optional_hint_unavailable', detail=str(exc))
