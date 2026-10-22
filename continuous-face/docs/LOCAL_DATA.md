# Local data and portability contract

All data remain outside Git. `local-data-manifest.json` gives exact project-relative paths, sizes and SHA-256 values for 15 observed input/support/cache files (37,825,027 bytes total). That inventory is not a complete accepted-history bundle.

The unchanged sources currently expect these paths beneath `src/supported-candidate/`:

- `full-face-coverage-plan/restored/exact-rest/anatomy-registered-rest.npz` and matching anatomy metadata under restored/anatomy-reference/prior-body-handoff/.
- Five restored/source-body/body-{frontalis,corrugator,depressor,zygomaticus,orbicularis}.npz rest/fibre packets.
- W/geometry/whole-face-continuum-final.npz, whole-face-muscle-phases-final.npz, whole-face-integration-cells.npz and source-membership.npz.
- W/mechanics/coarse-physical-site-v1/supports/clipped-origin-support.json, clipped-origin-support.npz and constraint-matrix.npz.
- W/contact/coarse-site-reference-band-v1.npz. The code can generate this reference cache, but historical reports bind its exact bytes. Do not casually replace it for an existing history.

The exact anatomy SHA is `2555740759e8dd2f50a504914d02f582ba13725c8c77da1fbbb67c55f4c9978d`; finalized geometry SHA is `52be768af2970a48404aaec97a6588a84b370b2c944f336ca3da5b2fa36040d8`.

For a new machine, use a locally authorized data location. `data/`, `runs/`, `recovery/` and `outputs/` are ignored placeholders, not already-supported environment variables. The runtime has **not** been refactored to read `data/`. Preserve its expected path layout or implement a separately reviewed configuration adapter.

## Minimum portability work still required

1. Make data/project roots explicit instead of deriving them through `__file__.parents[...]`; preserve each source byte until deliberately creating a new version.
2. Make contact/reconstruction import paths and tool paths explicit. The supplied smoke establishes the necessary PYTHONPATH without rewriting imports. Renderer code expects `/usr/bin/c++`, `/usr/bin/ffmpeg` and `/usr/bin/ffprobe` and writes native build products next to its module.
3. Provide a verified external-path mapping for accepted-history proofs and immutable report snapshots. Reports contain original absolute provenance paths. Do not edit historical proofs to force them to pass, remove hash checks, or assume relative source relocation also relocates historical evidence.
4. Capture a tested, platform-specific environment/lockfile and run actual data-dependent validation on a clean authorized copy. Observed package versions alone are not that proof.

Geometry helpers retain optional imports of auricular_support, nasal_support and neck_support. These are absent from this bounded export and occur in optional helper methods/branches. The supplied construction path disables ear/neck additions, but those optional branches and a fresh geometry rebuild are unverified here.

## Scientific scope

The coarse carrier has 10,716 nodes and 38,703 tetrahedra over 661,671 selected 1-L0 voxel sites. Treating L0 as mm is an engineering assumption. The existing geometry report documents sampled rigid leakage and nonmanifold boundary edges. Coarse-site endpoint/chord audits are not global rendered-surface collision certification, an open-sheet non-crossing proof, a total trajectory-error bound, or biological validation.
