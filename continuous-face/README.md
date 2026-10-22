# CYBR HUMAN

Source-only handoff of a coarse anatomy-driven whole-face mechanics engineering prototype. This is not a finished character, validated realistic facial animation, medical model, or biologically calibrated simulation. It uses no target-face fitting or image generation.

## What is here

- A byte-preserving v14 mechanics/contact source milestone, including synthetic tests.
- The exact parent constitutive/tangent and inherited contact modules needed by those imports.
- Geometry construction sources and archived helpers, with their provenance limitations disclosed.
- Separate frozen renderer 0005 and 0006 source readers. **Use 0006 for v14 receipts**; 0005 recognizes older controller identities.
- A separate three-file v15 source delta. It is not applied to the v14 tree.
- Hash manifests, local-data requirements, and small repository smoke tests.

No anatomy meshes, simulation histories, large arrays, rendered media, compiled binaries, recovery archives, credentials, private conversation content, or private recovery identifiers are included. Do not add a blanket LICENSE: ownership and third-party provenance need review first.

## Quick check

With Python, NumPy and SciPy already available, from this directory:

```sh
python3 -B tests/smoke.py
```

This checks source hashes/parsing and runs 14 tiny synthetic mechanics/history tests. It does not load anatomy, run a face solve, resume a history, or establish scientific validity.

Optional native renderer smoke additionally requires Pillow and `/usr/bin/c++`:

```sh
python3 -B tests/native_render_smoke.py
```

It builds in a temporary directory and tests only synthetic pixel/math fixtures.

## Reproduction boundary

The completed milestone is v14 with both passive and zygomaticus/frontalis cases admitted through 25 ms, 24 saved states per case. This is early onset/settling, not a useful expression movie. The later v15 runtime change retains conservative query hints between processes; its source delta and adoption distinction are documented separately.

A source-only clone is **not a standalone full simulation reproduction**. Existing scripts assume a specific relative directory layout, local anatomy/source-support data, and hash-bound absolute paths in historical proofs. See [local-data contract](docs/LOCAL_DATA.md), [architecture](docs/ARCHITECTURE.md), [runtime requirements](docs/RUNTIME.md), [source/version status](docs/SOURCE_STATUS.md), and [provenance](docs/PROVENANCE.md).
