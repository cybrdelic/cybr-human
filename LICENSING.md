# Component licenses and attribution

This package is an aggregate of application source, independently licensed numerical/anatomical data and a small third-party browser library. A source-code license does not relicense the assets.

| Component | Scope and retained terms |
| --- | --- |
| Original application source, HTML, construction tooling and tests | The parent `cybr-geo` project supplied GPL version 2. Its complete existing notice is retained verbatim in `LICENSES/GPL-2.0.txt` for application code. This preserves the inherited notice; it is not a blanket license for the third-party data below. |
| Selected atlas geometry and derived anatomical/model/binding artifacts | CC BY-SA 4.0. Covers `assets/anatomy/geometry.npz`, associated manifest and derived geometry, registration and numerical artifacts under `output/neutral-head/v5`, `output/neutral-tissue/v6` and the selected runtime's asset subtree. Preserve upstream attribution, modification notices and share-alike terms; see the component LICENSE/UPSTREAM-LICENSE files and ASSET-LICENSES.md. |
| Three.js 0.169.0 | MIT. Exact notice is in `vendor/three-0.169.0/LICENSE`; retain it with copied library files. |
| Upstream anatomy documentation/notices | Preserved upstream notices and provenance in `anatomy-source/`; their upstream license applies. |

The internal selection derives from Z-Anatomy at commit `fe708ede50acf95e37bdcbaf1305a6bd5ba17586`: https://github.com/Z-Anatomy/Models-of-human-anatomy . Its retained notice credits BodyParts3D, the Database Center for Life Science, Z-Anatomy and the named upstream contributors. The local selection has 247 bone/muscle/cartilage/connective parts and 741 corresponding numerical arrays. It excludes the separately noncommercial inner-ear and kidney models and excluded brain material. The full upstream atlas ZIP/Blend and imported exterior/photographic assets are not distributed.

Modifications to anatomical material include selection/cropping, shared-frame registration and scaffold-derived support/shape/binding construction for a code-authored exterior. The exterior uses procedural numeric geometry/materials; deriving from the credited scaffold does not remove its applicable data terms. The current BodyParts3D license page states CC BY 4.0; the older upstream attribution is retained unaltered rather than erasing its history: https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html . CC BY-SA 4.0 terms: https://creativecommons.org/licenses/by-sa/4.0/ .

Consolidation and public packaging changes dated 2026-10-07 concern portable test URLs/suites, verification output paths, runtime-publisher file-copy fallback retention, documentation and privacy-safe provenance. Selected immutable runtime code, shaders and required asset bytes are preserved. Installed npm/Python dependencies are not vendored here; their pinned packages carry their own licenses.

No claim is made that one license grants rights to every component or to an excluded upstream asset. Preserve the component notices when redistributing.
