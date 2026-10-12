# Asset sources and reuse

The active skin geometry and material use numeric fields only. They do not
load an imported skin mesh, ocular mesh, photograph, face scan or external
texture library. The new foundation loads the permitted internal atlas selection in its native coordinates and provides a rest-anatomy view. That selection is separately licensed.

| Component | Reuse terms |
| --- | --- |
| Procedural skin, eyes, facial openings, hair and maps | Authored in code; no imported exterior/ocular mesh or photographic texture input |
| Selected Z-Anatomy bones, muscles, cartilage and associated support tissues | CC BY-SA 4.0, with retained upstream attribution; adaptations distributed under the same license |
| Local Three.js runtime | MIT; the upstream copyright and license are retained in `vendor/three-0.169.0/LICENSE` |

CC BY-SA 4.0 permits redistribution and adaptation for commercial purposes
under its attribution and share-alike conditions. It is not limited to research
or personal use. See the actual
[CC BY-SA terms](https://creativecommons.org/licenses/by-sa/4.0/).

For the selected anatomy, preserve `assets/anatomy/LICENSE.txt` and
`UPSTREAM-LICENSE.txt`. The source atlas contains some separately licensed
noncommercial inner-ear and kidney material; those structures are excluded
from this exported selection. The full source atlas is not shipped.

Primary source declarations:

- [Z-Anatomy repository and attribution](https://github.com/Z-Anatomy/Models-of-human-anatomy)
- [BodyParts3D current license](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)

Generating an asset with a script does not remove the terms of source material
used by that script. Rebuilding the default skin requires no such image source;
rebuilding the internal anatomical scaffold still uses the credited reference atlas.
The exterior is authored from numeric section and facial-chart controls. Its dimensions are checked against the licensed internal scaffold, and local bone/cartilage tangent anchors use that scaffold. Preserve the source attribution and share-alike terms with the derived anatomical artifacts; code generation does not remove them.

External-skin and photographic research files are excluded from the active
generator and release. The package includes only the credited internal selection,
code-generated exterior, procedural maps and local Three.js runtime.
