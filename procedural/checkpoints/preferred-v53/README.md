# Preferred original v53 rest checkpoint

This is the selected rough whole-face rest shape. It is an original procedural geometry checkpoint, not a complete physiological model. Optional orbital variants do not replace this mechanical/identity baseline.

## Reproduce

Requirements: Python 3.12, NumPy 2.3.5, SciPy 1.17.0. No network, installation or Library access is needed. The source begins from the exact original-generated v5 arrays and reconstructs v6c, v8, v46 and v53. It does not claim reconstruction of earlier unpublished stages.

From the repository root, using a new absolute output directory:

```
python checkpoints/preferred-v53/build_preferred.py --output /tmp/human-preferred-replay --lock /tmp/cybr-human-cpu.lock
```

The launcher uses two allowed CPUs, requires 2 GiB available memory, and stops its owned process tree at a sampled 900 MiB or 90-second bound. The guard is sampled user-space supervision, not a kernel cap. It never overwrites an existing output. A companion `-inputs` directory contains the exact extracted replay source. `--prepare-only` prepares inputs without numerical execution.

Outputs:
- `project/output/coordinated-orbital-scale-v53/head.npz`
- `project/output/coordinated-orbital-scale-v53/anatomical_semantics/authored_lip_lid_semantics.npz`
- `REPLAY_RECEIPT.json` and `RESOURCE_RECEIPT.json`

Expected head SHA-256: `545b8ea2b292107dc2da9f1e90379989eb620c6ee5c2c50b3f9b5316cc77420e`.
Expected semantic SHA-256: `d2587be48722543d557a329870f78b29125d1bfa0f0d0aa54d0fc8adc2dce811`.
All six stage identities are tested by the replay. The exact original source was freshly replayed on 2026-10-12; the relocated public entrypoint has its own validation receipt at integration level.

Five byte parts, each at most 6 MiB, reassemble the exact 27.4 MB compressed array asset. The launcher verifies every part and the original whole-archive SHA-256 before extraction in the fresh replay workspace. This asset supplies the original rest positions, connectivity and object IDs. Source code and authored control arrays are readable under `source/project`. Public relocation changes only historical metadata paths; numerical code/arrays remain exact. `PUBLIC_RELOCATION.json` records those changes. Existing repository GPL-2.0 and component attribution/share-alike notices remain applicable and must be retained.
