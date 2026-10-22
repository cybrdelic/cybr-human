"""Lossless, immutable report snapshots; live report writers stay unchanged.

A .json wrapper authenticates a sibling .json.gz payload and its exact decoded
bytes. Plain historical reports retain their existing identity and behavior.
No physics, report-field normalization, history elision, or q/v I/O occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zlib

FORMAT_KEY = 'report_storage_format'
FORMAT = 'lossless_json_gzip_v1'
WRAPPER_KEYS = {FORMAT_KEY, 'encoding', 'payload', 'encoded_sha256',
                'encoded_bytes', 'raw_sha256', 'raw_bytes'}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _document(raw: bytes):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('A mechanics report must be a JSON object')
    return value


@dataclass(frozen=True)
class Report:
    path: Path
    file_bytes: bytes
    raw: bytes
    document: dict
    payload_path: Path | None = None
    payload_bytes: bytes | None = None

    @property
    def file_sha256(self):
        """Existing file-hash semantics: wrapper hash, or legacy plain-file hash."""
        return _sha(self.file_bytes)

    @property
    def raw_sha256(self):
        """Exact original JSON bytes, not a reserialization of the document."""
        return _sha(self.raw)

    def identity(self):
        files = [{'path': str(self.path.resolve()), 'sha256': self.file_sha256,
                  'bytes': len(self.file_bytes), 'role': 'report'}]
        if self.payload_path is not None:
            files.append({'path': str(self.payload_path.resolve()),
                          'sha256': _sha(self.payload_bytes),
                          'bytes': len(self.payload_bytes), 'role': 'encoded_payload'})
        return {'report_storage_format': FORMAT if self.payload_path else 'plain_json',
                'file_sha256': self.file_sha256, 'raw_sha256': self.raw_sha256,
                'raw_bytes': len(self.raw), 'required_files': files}


def read_report(path, *, expected_file_sha256=None) -> Report:
    """Authenticate every required byte before returning any logical report data.

    A trusted expected_file_sha256 binds the wrapper, then its encoded/raw hashes
    bind the payload transitively. No fallback to plain JSON after wrapper error.
    """
    path = Path(path)
    file_bytes = path.read_bytes()
    if expected_file_sha256 is not None and _sha(file_bytes) != expected_file_sha256:
        raise ValueError('Mechanics report file SHA256 differs from its bound identity')
    envelope = _document(file_bytes)
    if FORMAT_KEY not in envelope:
        return Report(path, file_bytes, file_bytes, envelope)
    if set(envelope) != WRAPPER_KEYS or envelope[FORMAT_KEY] != FORMAT or envelope['encoding'] != 'gzip':
        raise ValueError('Unknown or malformed report storage wrapper')
    for key in ('encoded_sha256', 'raw_sha256'):
        if not _digest(envelope[key]):
            raise ValueError('Malformed report SHA256: ' + key)
    for key in ('encoded_bytes', 'raw_bytes'):
        if type(envelope[key]) is not int or envelope[key] < 1:
            raise ValueError('Malformed report length: ' + key)
    name = envelope['payload']
    expected_name = 'report-payload-' + envelope['encoded_sha256'] + '.json.gz'
    if not isinstance(name, str) or name != expected_name or Path(name).name != name:
        raise ValueError('Report payload must be a local content-addressed .json.gz sibling')
    payload_path = path.parent/name
    if payload_path.is_symlink() or not payload_path.is_file():
        raise ValueError('Required report payload is missing or is a symbolic link')
    encoded = payload_path.read_bytes()
    if len(encoded) != envelope['encoded_bytes'] or _sha(encoded) != envelope['encoded_sha256']:
        raise ValueError('Report encoded payload length/SHA256 differs')
    try:
        raw = gzip.decompress(encoded)
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError('Invalid gzip report payload') from exc
    if len(raw) != envelope['raw_bytes'] or _sha(raw) != envelope['raw_sha256']:
        raise ValueError('Report decoded bytes length/SHA256 differs')
    document = _document(raw)
    if FORMAT_KEY in document:
        raise ValueError('Nested report wrappers are not supported')
    return Report(path, file_bytes, raw, document, payload_path, encoded)


def load_report(path, *, expected_file_sha256=None):
    return read_report(path, expected_file_sha256=expected_file_sha256).document


def read_report_bytes(path, *, expected_file_sha256=None):
    return read_report(path, expected_file_sha256=expected_file_sha256).raw


def report_sha256(path):
    """Drop-in report-only file hash that also verifies its complete payload."""
    return read_report(path).file_sha256


def report_raw_sha256(path):
    return read_report(path).raw_sha256


def _publish_new(path: Path, data: bytes, *, allow_identical=False):
    """Publish fully-written bytes without replacing any existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.report-storage-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not allow_identical or path.is_symlink() or path.read_bytes() != data:
                raise FileExistsError('Preserve existing report artifact: ' + str(path))
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def write_report_snapshot(path, raw: bytes, *, compressed=True) -> Report:
    """Create only a NEW immutable snapshot from exact already-read JSON bytes.

    Never use for a repeatedly rewritten live report. Never normalizes the JSON,
    including existing diagnostic Infinity sentinels. Payload is published first;
    wrapper is published last. Identical payloads in one directory are shared.
    """
    path = Path(path)
    if path.suffix != '.json':
        raise ValueError('An explicit .json report/wrapper destination is required')
    if path.exists() or path.is_symlink():
        raise FileExistsError('Use a new immutable report snapshot path')
    if not isinstance(raw, bytes) or FORMAT_KEY in _document(raw):
        raise ValueError('Exact original report JSON bytes are required')
    if not compressed:
        _publish_new(path, raw)
    else:
        encoded = gzip.compress(raw, compresslevel=6, mtime=0)
        encoded_sha = _sha(encoded)
        name = 'report-payload-' + encoded_sha + '.json.gz'
        envelope = {FORMAT_KEY: FORMAT, 'encoding': 'gzip', 'payload': name,
                    'encoded_sha256': encoded_sha, 'encoded_bytes': len(encoded),
                    'raw_sha256': _sha(raw), 'raw_bytes': len(raw)}
        file_bytes = (json.dumps(envelope, indent=2, sort_keys=True)+'\n').encode('utf-8')
        _publish_new(path.parent/name, encoded, allow_identical=True)
        _publish_new(path, file_bytes)
    saved = read_report(path)
    if saved.raw != raw:
        raise ValueError('Immutable report snapshot did not preserve exact bytes')
    return saved


def copy_report_snapshot(source, destination, *, expected_file_sha256=None,
                         compress_plain=False) -> Report:
    """Retain every byte needed by an independent snapshot or restored archive.

    A compressed source is copied byte-exact (wrapper AND encoded payload), so
    existing proofs keep their file hashes. A plain source stays plain unless explicitly
    requested with compress_plain=True; callers must record the returned new file identity.
    """
    saved = read_report(source, expected_file_sha256=expected_file_sha256)
    destination = Path(destination)
    if destination.suffix != '.json':
        raise ValueError('An explicit .json report/wrapper destination is required')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Use a new immutable snapshot path')
    if saved.payload_path is None:
        return write_report_snapshot(destination, saved.raw, compressed=compress_plain)
    _publish_new(destination.parent/saved.payload_path.name, saved.payload_bytes, allow_identical=True)
    _publish_new(destination, saved.file_bytes)
    return read_report(destination, expected_file_sha256=saved.file_sha256)


def restore_plain_report(source, destination, *, expected_file_sha256):
    """Explicit byte-exact restoration; caller retains the returned provenance."""
    before = read_report(source, expected_file_sha256=expected_file_sha256)
    after = write_report_snapshot(destination, before.raw, compressed=False)
    return {'source': before.identity(), 'restored': after.identity(),
            'exact_original_json_bytes_restored': before.raw == after.raw}
