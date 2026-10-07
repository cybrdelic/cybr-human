# Selected prebuilt runtime versus editable source

The stage retains two deliberate identities: selected immutable runtime `4bc8b53a8f15b8b62cf6a86e` and newer editable source. Rebuilding the editable source produces `cff25f32c7645d113a4d3ff0`; it does **not** recreate the selected behavior or the historical unpromoted candidate `ef7aee6716ff02e2cc0021f9`.

The publisher hashes sorted relative input names, NUL separators and pre-URL-rewrite file bytes, then takes 24 hex characters of SHA256. It does not hash absolute staging paths, timestamps, the README or test tooling. Two rebuilds in a disposable copy produced the same `cff25f32â€¦`; an earlier run in a different temporary directory produced that identity too. Copying the 17 selected runtime source files into a disposable editable source tree recomputes the exact selected `4bc8b53â€¦` identity using the included asset inputs. This reconstruction does not alter the stage's newer source or pointer.

## Observed differences

| Comparison | Differences | Implication |
| --- | --- | --- |
| Selected `4bc8â€¦` â†’ historical candidate `ef7â€¦` | Added `fem_surface_cache.js`; changed `layered_fem_surface.js`, `full_head_fem_gpu.js`, `viewer.js` | Candidate caches dense surface positions/normals, changes rendering transfer hooks, and introduces bounded-CG early exit/readback/scheduling. These can affect performance and rendered or stepped results and need separate parity/visual acceptance. |
| Historical candidate `ef7â€¦` â†’ editable rebuild `cffâ€¦` | Changed `authored_skin_material.js` and `viewer.js` | The anatomical lip tint receives an authored upper/lower shape and smoother edge. Interactive static solves display one Newton update per iteration rather than the former 2/4 updates per viewer step. The cap changes from 120 viewer steps to 480 counted Newton iterations; the 45-second cap and force thresholds remain. |
| Selected `4bc8â€¦` â†’ editable rebuild `cffâ€¦` | Five changed/added source paths, 56 unchanged manifest paths | All differences above are inherited pre-staging work. Stage cleanup did not create them. |

All **43 model/geometry/binding asset files** have identical manifest source identities across the three builds. GLBs, NPZ-derived runtime arrays, all binary numerical fields, parameter delta arrays and model/embedding metadata are unchanged. `full_head_fem.wgsl`, `layered_fem.js`, `shell_bending.js`, CPU backend/worker and transfer math are unchanged too. There is no observed constitutive-equation or stored numerical-asset change.

That does not prove behavior equivalence. The new viewer scheduling can alter the trajectory while applied parameters ramp across updates, the number of quiet updates and the state reached before the wall-clock cap. Surface caching changes the rendering execution path, and lip shading is an actual visual change. Bounded CG changes queue submissions and host readback; its intended equivalence remains a candidate test obligation. A successful publisher run proves construction and identity consistency only.

## Operator contract

Run the included selected application without rebuilding. Treat `src/` as the preserved next-source branch. Use `tools/build_tissue_runtime.py --no-publish` only for a separate candidate and evaluate it explicitly; `manage.py build` is an intentional verified promotion operation. No candidate was promoted during consolidation. Selected-runtime visual/video verification names `4bc8b53a8f15b8b62cf6a86e` regardless of newer editable source.

The measured comparison was performed in disposable copies; raw reports containing host paths are not shipped. Temporary rebuild trees were removed. Full geometry regeneration was not performed, so this result covers runtime publication from the retained construction inputs, not regeneration of every mesh and FEM intermediate.
