"""Bounded, synthetic contract regressions; not physiological validation."""
import copy
import json
import math
from pathlib import Path
import sys
import subprocess
import contextlib
import io
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import tissue_calibration as calibration


def dataset():
    return {"schema": "cybr-human.passive-observations.v1", "units": "SI",
            "provenance": {"kind": "synthetic_test", "access": "public",
                           "source": "analytical contract test, not human observations", "license": "test-only"},
            "traces": [{"id": f"trace-{i}", "subject_id": f"synthetic-{i}",
                        "split": split, "fixture_id": "8S133", "region": "forehead",
                        "boundary_id": "synthetic-fixed-ring",
                        "samples": [{"time_s": t, "displacement_m": 0.001 * t}
                                    for t in (0.0, 0.5, 1.0)]}
                       for i, split in enumerate(("train", "holdout"))]}


class Contracts(unittest.TestCase):
    def test_baseline_and_regional_override(self):
        base = calibration.material_manifest()
        self.assertFalse(base["runtime_applied"])
        self.assertFalse(base["fitted"])
        self.assertIsNone(base["published_modulus_ranges"])
        for region in calibration.REGIONS:
            self.assertEqual(base["regions"][region]["skin"]["young_pa"], 120000)
            self.assertEqual(base["regions"][region]["fat"]["poisson"], 0.48)
        modified = calibration.material_manifest({"jaw": {"skin": {
            "young_pa": 60000, "evidence": "authored_candidate", "source": "sensitivity experiment"}}})
        self.assertEqual(modified["regions"]["jaw"]["skin"]["mu_pa"], 60000 / 2.9)
        self.assertEqual(modified["regions"]["forehead"], base["regions"]["forehead"])
        for e, nu in ((0, 0.45), (120000, 0.5), (1, -1), (math.nan, 0.4), (True, 0.4)):
            with self.assertRaises(ValueError):
                calibration.lame(e, nu)
        for override in ({"cheek": {}}, {"jaw": {"skin": {"young_pa": 1}}},
                         {"jaw": {"muscle": {}}}):
            with self.assertRaises(ValueError):
                calibration.material_manifest(override)

    def test_published_and_authored_waveforms(self):
        fixtures = {f["id"]: f for f in calibration.fixtures()["fixtures"]}
        self.assertEqual(len(fixtures), 11)
        step = fixtures["8S133"]
        self.assertEqual(step["opening_diameter_m"], 0.008)
        self.assertEqual(calibration.stimulus(step, 0.1), 13300)
        self.assertEqual(calibration.stimulus(step, 30), 0)
        ramp = fixtures["2R10"]
        for t, p in ((0, 0), (8.75, 8750), (17.5, 17500), (26.25, 8750), (35, 0)):
            self.assertEqual(calibration.stimulus(ramp, t), p)
        hold = fixtures["passive-hold-release"]
        self.assertIsNone(hold["biological_target"])
        self.assertEqual(calibration.stimulus(hold, 0.125), 0.0005)
        self.assertEqual(calibration.stimulus(hold, 5), 0.001)
        self.assertEqual(calibration.stimulus(hold, 5.5), 0)
        self.assertEqual(calibration.stimulus(hold, -1), 0)

    def test_reject_import_corruption_and_leakage(self):
        self.assertEqual(calibration.validate_dataset(dataset())["subjects"], 2)
        mutations = [lambda d: d.update(units="mm"),
                     lambda d: d["provenance"].update(source=""),
                     lambda d: d["provenance"].update(kind="fitted"),
                     lambda d: d["traces"][1].update(subject_id="synthetic-0"),
                     lambda d: d["traces"][1].update(id="trace-0"),
                     lambda d: d["traces"][1].update(split="train"),
                     lambda d: d["traces"][0].update(boundary_id=""),
                     lambda d: d["traces"][0].update(name="not allowed"),
                     lambda d: d["traces"][0]["samples"][1].update(time_s=0),
                     lambda d: d["traces"][0]["samples"][1].update(displacement_m=math.inf),
                     lambda d: d["traces"][0]["samples"][1].update(force_n=1),
                     lambda d: d["traces"][0].update(fixture_id="passive-hold-release")]
        for mutate in mutations:
            data = dataset()
            mutate(data)
            with self.assertRaises(ValueError):
                calibration.validate_dataset(data)

    def test_curve_scores_and_held_out_metadata(self):
        observed, predicted = dataset(), dataset()
        # A coarser exact linear prediction must score zero by interpolation.
        for trace in predicted["traces"]:
            del trace["samples"][1]
        report = calibration.evaluate(observed, predicted)
        self.assertFalse(report["measured"])
        self.assertFalse(report["biological_accuracy_accepted"])
        self.assertEqual({r["split"] for r in report["scores"]}, {"train", "holdout"})
        self.assertTrue(all(r["rmse_si"] == 0 for r in report["scores"]))
        for trace in predicted["traces"]:
            for sample in trace["samples"]:
                sample["displacement_m"] += 0.0002
        self.assertAlmostEqual(calibration.evaluate(observed, predicted)["scores"][0]["rmse_si"], 0.0002)
        bad = copy.deepcopy(predicted)
        bad["traces"][0]["samples"][0]["time_s"] = 0.1
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            calibration.evaluate(observed, bad)
        bad = copy.deepcopy(predicted)
        bad["traces"][0]["boundary_id"] = "different-ring"
        with self.assertRaisesRegex(ValueError, "metadata"):
            calibration.evaluate(observed, bad)

    def test_zero_signal_is_not_false_normalized_accuracy(self):
        data = dataset()
        for trace in data["traces"]:
            for sample in trace["samples"]:
                sample["displacement_m"] = 0
        scores = calibration.evaluate(data, data)["scores"]
        self.assertTrue(all(score["normalized_rmse"] is None for score in scores))
        json.dumps(scores, allow_nan=False)

    def test_cli_roundtrip_and_exact_input_hashes(self):
        tool = Path(calibration.__file__)
        def run(*args):
            return json.loads(subprocess.check_output([sys.executable, str(tool), *map(str, args)], text=True))
        self.assertFalse(run("manifest")["runtime_applied"])
        self.assertEqual(len(run("fixtures")["fixtures"]), 11)
        data = dataset()
        data["provenance"]["kind"] = "measured"
        data["provenance"]["source"] = "Synthetic test of measured label; no human data"
        observation_bytes = json.dumps(data).encode()
        prediction_bytes = json.dumps(dataset()).encode()
        output = io.StringIO()
        # Mock file transport to keep the test usable in restricted read-only
        # environments; manifest/fixtures above exercise actual CLI subprocesses.
        with mock.patch.object(sys, "argv", [str(tool), "evaluate", "observed.json", "predicted.json"]), \
                mock.patch.object(calibration, "load", side_effect=[data, dataset()]), \
                mock.patch.object(Path, "read_bytes", side_effect=[observation_bytes, prediction_bytes]), \
                contextlib.redirect_stdout(output):
            calibration.main()
        report = json.loads(output.getvalue())
        self.assertTrue(report["measured"])
        self.assertFalse(report["biological_accuracy_accepted"])
        self.assertEqual(report["input_sha256"]["observations"],
                         calibration.hashlib.sha256(observation_bytes).hexdigest())


if __name__ == "__main__":
    unittest.main()
