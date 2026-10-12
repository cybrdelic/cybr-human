"""Offline regional material/fixture contracts. No runtime model is modified.

All quantities use SI. Published protocols are not fitted material parameters.
The CLI deliberately cannot claim biological validation from synthetic traces.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

PAPER = "https://doi.org/10.1016/j.jmbbm.2017.10.021"
REGIONS = ("forehead", "parotid", "jaw", "unassigned")
BASELINE = {"skin": {"young_pa": 120000.0, "poisson": 0.45},
            "fat": {"young_pa": 18000.0, "poisson": 0.48}}
# Table 1. Pressure is a positive suction magnitude, not signed surface traction.
PROTOCOLS = {
    "2S300": (2, "step", 60, 300), "2S500": (2, "step", 60, 500),
    "8S66": (8, "step", 30, 66), "8S133": (8, "step", 30, 133),
    "8S200": (8, "step", 30, 200), "2R10": (2, "ramp", 17.5, 175),
    "2R15": (2, "ramp", 17.5, 263), "8R10": (8, "ramp", 10, 100),
    "8R15": (8, "ramp", 10, 150), "8R20": (8, "ramp", 10, 200),
}


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def lame(young_pa, poisson):
    e, nu = number(young_pa, "young_pa"), number(poisson, "poisson")
    if e <= 0 or not -1 < nu < 0.5:
        raise ValueError("Require E > 0 and -1 < nu < 0.5")
    coefficients = {"mu_pa": e / (2 * (1 + nu)),
                    "lambda_pa": e * nu / ((1 + nu) * (1 - 2 * nu))}
    if not all(math.isfinite(value) for value in coefficients.values()):
        raise ValueError("Material coefficients overflow")
    return coefficients


def material_manifest(overrides=None):
    """Explicit regional values; unknown anatomy never silently maps to a region.

    Overrides require their own evidence label; they are candidates, never fitted
    merely because this manifest was generated. Anisotropy/relaxation stay absent.
    """
    overrides = overrides or {}
    if set(overrides) - set(REGIONS):
        raise ValueError("Unknown regional override")
    regions = {}
    for region in REGIONS:
        layers = {}
        edits = overrides.get(region, {})
        if set(edits) - set(BASELINE):
            raise ValueError("Unknown layer override")
        for layer, baseline in BASELINE.items():
            edit = edits.get(layer, {})
            if set(edit) - {"young_pa", "poisson", "evidence", "source"}:
                raise ValueError("Unknown material field")
            if edit and (edit.get("evidence") not in ("authored_candidate", "published_parameter")
                         or not isinstance(edit.get("source"), str) or not edit["source"].strip()):
                raise ValueError("Overrides require evidence and source")
            values = {**baseline, **edit}
            values.update(lame(values["young_pa"], values["poisson"]))
            values.setdefault("evidence", "authored_baseline")
            values.setdefault("source", "tools/build_volume.py:361-362")
            layers[layer] = values
        regions[region] = layers
    return {"schema": "cybr-human.material-candidate.v1", "units": "SI",
            "regions": regions, "runtime_applied": False, "fitted": False,
            "published_modulus_ranges": None, "regional_evidence_source": PAPER,
            "region_assignment": "External explicit per-element labels required; no inferred landmarks",
            "limitations": ["Baseline is authored, not measured calibration",
                            "Suction response depends on geometry and boundary conditions",
                            "No viscoelasticity, anisotropy or muscle activation is enabled"]}


def fixtures():
    suction = []
    for name, (diameter, mode, on, mbar) in PROTOCOLS.items():
        suction.append({"id": name, "kind": "suction", "evidence": "published_protocol",
                        "source": PAPER, "regions": list(REGIONS[:-1]),
                        "opening_diameter_m": diameter * 0.001,
                        "mode": mode, "on_s": float(on), "off_s": 0.1,
                        "peak_pressure_pa": mbar * 100.0,
                        "preconditioning_rest_s": [30.0, 45.0],
                        "boundary_contract": "Seal probe ring; specify contact preload, substrate and layer geometry",
                        "observe": ["displacement_m"], "instantaneous_sample_s": 0.1})
    # A displacement-controlled hold observes FORCE relaxation, not creep. This
    # protocol is authored for implementation acceptance, not from the paper.
    hold = {"id": "passive-hold-release", "kind": "displacement_hold",
            "evidence": "authored_benchmark", "regions": list(REGIONS[:-1]),
            "ramp_s": 0.25, "hold_s": 5.0, "release_s": 0.25, "recovery_s": 5.0,
            "displacement_m": 0.001, "observe": ["force_n", "displacement_m"],
            "boundary_contract": "Explicit patch node IDs, pull direction, fixed substrate IDs; no muscle drive",
            "acceptance_contract": {"positive_jacobians": True, "finite_energy": True,
                                    "force_balance_required": True, "reference_cpu_required": True},
            "biological_target": None}
    return {"schema": "cybr-human.passive-fixtures.v1", "units": "SI",
            "fixtures": suction + [hold], "measured_data_included": False}


def stimulus(fixture, time_s):
    """Scalar command magnitude; solver must supply its own ring/contact model.

    Suction ramps load AND unload at the published pressure rate: on_s is each
    branch duration (peak pressure / pressure rate). Negative time is a zero-load
    prehistory. Observation acquisition must resolve load discontinuities.
    """
    t = number(time_s, "time_s")
    if t < 0:
        return 0.0
    if fixture["kind"] == "suction":
        on, peak = fixture["on_s"], fixture["peak_pressure_pa"]
        if fixture["mode"] == "step":
            return peak if t < on else 0.0
        return peak * max(0.0, min(t / on, 2.0 - t / on))
    ramp, hold, release = fixture["ramp_s"], fixture["hold_s"], fixture["release_s"]
    peak = fixture["displacement_m"]
    if t < ramp:
        return peak * t / ramp
    if t < ramp + hold:
        return peak
    return peak * max(0.0, 1.0 - (t - ramp - hold) / release)


def validate_dataset(data):
    """Validate a public/authorized SI trace import and prevent subject leakage.

    No age, name, image or subject profile fields are part of this schema. Trace
    subject IDs must be opaque study IDs. Metadata is required, never fabricated.
    """
    if not isinstance(data, dict) or set(data) != {"schema", "units", "provenance", "traces"}:
        raise ValueError("Require exactly schema, units, provenance and traces")
    if data.get("schema") != "cybr-human.passive-observations.v1" or data.get("units") != "SI":
        raise ValueError("Unsupported observation schema or units (SI required)")
    provenance = data.get("provenance", {})
    if (not isinstance(provenance, dict) or set(provenance) != {"kind", "access", "source", "license"}
            or provenance.get("kind") not in ("measured", "synthetic_test")
            or provenance.get("access") not in ("public", "authorized")
            or not all(isinstance(provenance.get(k), str) and provenance[k].strip()
                       for k in ("source", "license"))):
        raise ValueError("Require observation kind, permitted access, source and license")
    traces = data.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("Require nonempty traces")
    known = {f["id"]: f for f in fixtures()["fixtures"]}
    splits, ids = {}, set()
    for trace in traces:
        if not isinstance(trace, dict):
            raise ValueError("Trace must be an object")
        if set(trace) != {"id", "subject_id", "split", "fixture_id", "region", "boundary_id", "samples"}:
            raise ValueError("Unsupported trace fields; keep subject profiles outside this contract")
        tid, subject = trace.get("id"), trace.get("subject_id")
        if (not isinstance(tid, str) or not tid.strip() or tid in ids
                or not isinstance(subject, str) or not subject.strip()):
            raise ValueError("Require unique trace ID and opaque subject ID")
        ids.add(tid)
        split = trace.get("split")
        if split not in ("train", "holdout") or subject in splits and splits[subject] != split:
            raise ValueError("Subjects must belong exclusively to train or holdout")
        splits[subject] = split
        protocol = known.get(trace.get("fixture_id"))
        if not protocol or trace.get("region") not in REGIONS[:-1]:
            raise ValueError("Unknown fixture or unsupported observation region")
        # Required separately from protocol: geometry/preload change fitted results.
        if not isinstance(trace.get("boundary_id"), str) or not trace["boundary_id"].strip():
            raise ValueError("Require explicit boundary/geometry record ID")
        samples = trace.get("samples")
        if not isinstance(samples, list) or len(samples) < 2:
            raise ValueError("Require at least two time samples")
        if not all(isinstance(sample, dict) for sample in samples):
            raise ValueError("Samples must be objects")
        previous = -math.inf
        channels = set(samples[0]) - {"time_s"}
        if not channels or channels - set(protocol["observe"]):
            raise ValueError("Unknown or missing observable")
        if protocol["kind"] == "displacement_hold" and "force_n" not in channels:
            raise ValueError("Displacement hold must record reaction force")
        for sample in samples:
            if set(sample) != channels | {"time_s"}:
                raise ValueError("Channels must remain consistent")
            t = number(sample["time_s"], "time_s")
            if t < 0 or t <= previous:
                raise ValueError("Times must be nonnegative and strictly increasing")
            previous = t
            for channel in channels:
                number(sample[channel], channel)
    if set(splits.values()) != {"train", "holdout"}:
        raise ValueError("Require subject-separated train AND holdout traces")
    return {"traces": len(traces), "subjects": len(splits),
            "measured": provenance["kind"] == "measured"}


def interpolate(samples, channel, t):
    if t < samples[0]["time_s"] or t > samples[-1]["time_s"]:
        raise ValueError("Prediction must cover observations; extrapolation prohibited")
    for left, right in zip(samples, samples[1:]):
        if left["time_s"] <= t <= right["time_s"]:
            weight = (t - left["time_s"]) / (right["time_s"] - left["time_s"])
            return left[channel] * (1 - weight) + right[channel] * weight
    return samples[-1][channel]


def evaluate(observations, predictions):
    """Compare full curves, independently report held-out subjects and regions.

    Predictions use the same contract/IDs/metadata. This scores a supplied solver
    candidate, and neither fits parameters nor certifies biological accuracy.
    """
    info = validate_dataset(observations)
    validate_dataset(predictions)
    if predictions["provenance"]["kind"] != "synthetic_test":
        raise ValueError("Predictions must be labeled synthetic_test")
    indexed = {trace["id"]: trace for trace in predictions["traces"]}
    if set(indexed) != {trace["id"] for trace in observations["traces"]}:
        raise ValueError("Prediction trace IDs must match observations exactly")
    report = []
    for observed in observations["traces"]:
        predicted = indexed[observed["id"]]
        for key in ("subject_id", "fixture_id", "region", "split", "boundary_id"):
            if predicted[key] != observed[key]:
                raise ValueError(f"Prediction metadata mismatch: {key}")
        channels = set(observed["samples"][0]) - {"time_s"}
        if channels != set(predicted["samples"][0]) - {"time_s"}:
            raise ValueError("Prediction channels must match observations")
        for channel in sorted(channels):
            errors = [interpolate(predicted["samples"], channel, sample["time_s"]) - sample[channel]
                      for sample in observed["samples"]]
            scale = max(abs(sample[channel]) for sample in observed["samples"])
            rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
            report.append({"id": observed["id"], "split": observed["split"],
                           "region": observed["region"], "channel": channel,
                           "samples": len(errors), "rmse_si": rmse,
                           "max_abs_error_si": max(map(abs, errors)),
                           "normalized_rmse": rmse / scale if scale > 0 else None})
    return {"schema": "cybr-human.calibration-score.v1", **info,
            "scores": report, "biological_accuracy_accepted": False,
            "limitations": "Curve comparison only; no fitted parameters, uncertainty or independent biological acceptance"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--overrides", type=Path)
    commands.add_parser("fixtures")
    validate = commands.add_parser("validate")
    validate.add_argument("observations", type=Path)
    score = commands.add_parser("evaluate")
    score.add_argument("observations", type=Path)
    score.add_argument("predictions", type=Path)
    args = parser.parse_args()
    if args.command == "manifest":
        result = material_manifest(load(args.overrides) if args.overrides else None)
    elif args.command == "fixtures":
        result = fixtures()
    elif args.command == "validate":
        result = validate_dataset(load(args.observations))
    else:
        result = evaluate(load(args.observations), load(args.predictions))
        result["input_sha256"] = {label: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for label, path in (("observations", args.observations),
                                                      ("predictions", args.predictions))}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
