"""Freeze a consistent, hash-bound accepted case while its actual solve continues.

Copies only already-persisted bytes. Does not invent, interpolate, truncate or
alter physical states, their times, or the mechanics report.
"""
import argparse
import hashlib
import json
from pathlib import Path
from report_storage import read_report, copy_report_snapshot


def snapshot(run_directory, case, destination):
    run_directory, destination = Path(run_directory), Path(destination)
    if destination.exists():
        raise ValueError('Use a new snapshot directory')
    report_path = run_directory/'report.json'
    stored_report = read_report(report_path)
    report = stored_report.document
    run = report.get('runs', {}).get(case, {})
    if run.get('accepted_frames', 0) < 1:
        raise ValueError('No actual accepted physical states yet')
    source = Path(run['accepted_npz_path'])
    if not source.is_absolute():
        source = run_directory/source
    frame_bytes = source.read_bytes()
    frame_sha = hashlib.sha256(frame_bytes).hexdigest()
    if frame_sha != run.get('accepted_npz_sha256'):
        raise RuntimeError('Frame/report pair changed during read; retry at a stable checkpoint')
    read_report(report_path, expected_file_sha256=stored_report.file_sha256)
    destination.mkdir(parents=True)
    saved_report = copy_report_snapshot(report_path, destination/'report.json',
        expected_file_sha256=stored_report.file_sha256, compress_plain=False)
    read_report(report_path, expected_file_sha256=stored_report.file_sha256)
    (destination/'accepted.npz').write_bytes(frame_bytes)
    provenance = dict(source_report=str(report_path.resolve()), source_frames=str(source.resolve()),
        report_sha256=stored_report.file_sha256, report_raw_sha256=stored_report.raw_sha256,
        source_report_storage=stored_report.identity(), snapshot_report_storage=saved_report.identity(), frames_sha256=frame_sha,
        case=case, accepted_frames=run['accepted_frames'], actual_times_s=run['actual_times_s'],
        byte_exact_snapshot=True, physical_states_modified=False)
    (destination/'snapshot-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-directory', required=True, type=Path)
    p.add_argument('--case', required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(snapshot(a.run_directory, a.case, a.output), indent=2))
