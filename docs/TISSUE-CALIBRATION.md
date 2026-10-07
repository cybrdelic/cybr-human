# Regional tissue calibration contracts

This milestone provides offline material candidates, passive fixture definitions,
strict observation imports and held-out curve comparisons. It does not change
the viewer, generated assets, selected runtime or FEM solver. No human dataset
or fitted calibration is included. Passing its tests establishes contract
behavior, not physiological accuracy.

## Evidence and baseline

The existing `tools/build_volume.py` assigns skin E = 120,000 Pa, nu = 0.45 and
fat E = 18,000 Pa, nu = 0.48. These are authored baseline values. The new manifest
keeps them unchanged in forehead, parotid, jaw and explicitly unassigned regions.
It exports Lamé coefficients from E/nu and validates material admissibility.
Region assignment needs explicit element labels and a reviewed geometry map;
this tool does not infer locations from bounding boxes or pretend to have
segmented tissue compartments.

[Pensalfini et al., 2018](https://doi.org/10.1016/j.jmbbm.2017.10.021)
([public paper](https://weickenmeierlab.com/wp-content/uploads/paper/Pensalfini_JMBBM18.pdf))
characterized forehead, parotid and jaw using suction and ultrasound in nine
volunteers. Table 1 defines ten step/ramp protocols with 2/8 mm probe openings.
The study supports region-dependent behavior and comparison across rates and
opening sizes. It does **not** supply interchangeable Young's-modulus ranges
for this authored mesh. Thickness, probe contact and attachment conditions must
be modeled with the response. Its supplementary measurements have not been
imported or licensed for redistribution here. The PDF's DOI is
`10.1016/j.jmbbm.2017.10.021`; `2017.02.022` is not its DOI.

Overrides require `evidence: authored_candidate` or `published_parameter` and a
nonempty source. A published parameter still needs checking for constitutive
model, units and experimental context. No override becomes a fitted parameter
through manifest generation. Published E/nu ranges remain `null`. Anisotropy,
viscoelasticity, sliding and active muscles remain future solver changes requiring
their own calibration and numerical verification.

## Commands

Run from the repository root with Python 3.10 or newer. No extra dependencies,
browser, GPU or network are required.

```powershell
python tools/tissue_calibration.py manifest
python tools/tissue_calibration.py fixtures
python tools/tissue_calibration.py validate observations.json
python tools/tissue_calibration.py evaluate observations.json predictions.json
python tests/tissue-calibration.py
```

Commands print JSON to standard output. Saving it locally with shell redirection
is optional; no private inputs or dataset are uploaded. Evaluation reports SHA-256
of the exact input files. Material overrides can be supplied with
`manifest --overrides regional-overrides.json`:

```json
{
  "jaw": {
    "skin": {
      "young_pa": 60000,
      "evidence": "authored_candidate",
      "source": "Sensitivity experiment only; not measured tissue"
    }
  }
}
```

## Passive fixture adapter contract

`fixtures()` emits ten published suction commands and one authored displacement
hold/release benchmark. `stimulus(fixture, time_s)` gives the scalar positive
suction-pressure or displacement magnitude. A step switches off at `on_s`;
a ramp increases to the peak over `on_s`, then decreases at the same rate.
Pressure magnitude is converted from mbar to Pa; it is not a signed force.
The solver adapter must apply traction along the correct surface normal and
model probe-ring sealing/contact. Preserve the paper's acquisition and preload
conditions before comparing any experimental records.

The authored benchmark ramps a patch to 1 mm in 0.25 s, holds 5 s, releases over
0.25 s and observes 5 s of recovery. Record reaction force during the hold and
displacement after release. A displacement hold tests **force relaxation**;
a constant pressure hold tests **creep displacement**. Solver damping or
quasi-static settling alone is not evidence of a tissue relaxation law.
The current elastic solver has no calibrated viscoelastic response to accept.

Every adapter must specify patch/ring node IDs, force direction, fixed substrate
IDs, tissue geometry, preload and simulation clock. It must preserve positive
Jacobians, finite energy, force balance and comparison with the CPU reference.
These requirements are exported as a fixture contract; this tool does not run
the FEM or claim those checks passed. Avoid broadening the 1 mm benchmark into
a large deformation before checking mesh/contact adequacy.

## Observation and prediction format

JSON uses strict SI units and opaque study IDs. Unknown fields are rejected,
including personal profile fields. Each trace requires a geometry/boundary record
ID. A complete import needs at least one train subject and a different holdout
subject. All records from a subject must remain in one split; never tune using
holdout subjects. Obtain genuinely independent data and preserve license/access
information before replacing the synthetic test provenance.

```json
{
  "schema": "cybr-human.passive-observations.v1",
  "units": "SI",
  "provenance": {
    "kind": "synthetic_test",
    "access": "public",
    "source": "Contract demonstration; not human measurements",
    "license": "Test only"
  },
  "traces": [
    {
      "id": "example-train", "subject_id": "synthetic-A", "split": "train",
      "fixture_id": "8S133", "region": "forehead", "boundary_id": "example-ring",
      "samples": [
        {"time_s": 0, "displacement_m": 0},
        {"time_s": 0.1, "displacement_m": 0.0001}
      ]
    },
    {
      "id": "example-holdout", "subject_id": "synthetic-B", "split": "holdout",
      "fixture_id": "8S133", "region": "jaw", "boundary_id": "example-ring",
      "samples": [
        {"time_s": 0, "displacement_m": 0},
        {"time_s": 0.1, "displacement_m": 0.0001}
      ]
    }
  ]
}
```

Predictions use the same format and must be labeled `synthetic_test`. IDs,
subjects, splits, regions, protocols, boundaries and observable channels must
match observations exactly. Suction uses `displacement_m`; hold/release requires
`force_n` and may additionally include `displacement_m`. Times must strictly
increase; all values must be finite. The format can represent signed reactions
and displacement along an explicitly documented direction.

The scorer linearly interpolates predictions onto observation times, prohibits
extrapolation and reports RMSE, maximum error and peak-normalized RMSE separately
for each trace/channel/split/region. A zero observation magnitude has no normalized
score. Resolve transient/discontinuous behavior with actual samples on both sides;
coarse interpolation can otherwise understate error. Short traces remain short
trace comparisons, not completed creep/recovery experiments. The demonstration
above covers only 0.1 s and cannot assess the 30 s pressure hold.

## Release gate for future mechanical changes

Obtain permitted raw time series and geometry/preload records; freeze subject
splits and experimental uncertainty before fitting. Use training traces to fit
regional constitutive candidates, then compare withheld subjects, loads and rates.
Choose acceptance bounds from measurement uncertainty and the intended use;
this milestone deliberately invents no biological tolerance or fitted dataset.
Test parameter identifiability because several layer/attachment combinations
can yield similar suction response. Repeat numerical positivity, force/energy,
CPU/GPU agreement and device-loss recovery checks for each integrated candidate.
Publish baseline/candidate curves and input provenance with actual results.
Numerical stability and low residual error alone do not establish biological
validation or subject-specific accuracy.
