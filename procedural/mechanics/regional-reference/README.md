# Preserved regional mechanics

This self-contained numerical workspace preserves the checked original right-cheek proof, reciprocal insertion and conforming perioral continuation. It is an experimental regional mechanics module, not a complete expression rig. No later unrecovered equilibrium is claimed here. The exact preferred exterior is a separate reproducible checkpoint.

## Supported scope

- Original right-cheek: 176 exact bone-root nodes plus 176 rigid collar nodes. The clamped root prism has zero root reaction; the collar carries proximal loads. Five actual internal equilibria are preserved. Only the qualified 2% and release skin snapshots are included; the failed 8% and 16% fine-skin states are excluded.
- The original fine-skin display is one-way interpolation. It adds no reciprocal skin force. Its released state has a 0.35455-design-unit skin residual at the saved finite tolerance.
- Reciprocal insertion: 72 additional passive tetrahedra share the muscle degrees of freedom. Its close perimeter is an explicit rigid truncation, not a physiological attachment.
- Perioral continuation: 288 further conforming passive cells relocate the truncation. This 3,401-cell reference has exact neutral geometry; saved 2% residual 6.540115e-8 and minimum J .757070. The superficial proxy moves .021498 design units. Release residual is 7.527008e-8 with .000859762 maximum residual motion. Twenty cut supports approximate omitted buccal/perioral tissue; they are not a jaw attachment.
- Static mandibular/contact layout and distinct route construction are preserved as unsolved geometry references. Full skull, jaw, neck and lid mechanics remain absent.

Lengths use `solver_length = 0.001 * authored_design_length` as numerical scaling. Material constants are Young 3200, Poisson .43 and fibre stiffness 24000. This model is not calibrated to human dimensions or forces; results must not be labelled Newtons or physiological smile percentages.

## Reproduce the limited original cheek sequence

Requirements used for the checked numerical source: Node 24, Python 3.12, NumPy 2.3.5, SciPy 1.17.0, Blender 4.3.2 for BVH geometry scripts. Nothing is installed automatically.

Build the exact preferred head using `checkpoints/preferred-v53/build_preferred.py`, then:

```
python mechanics/regional-reference/prepare_replay.py --head /tmp/human-preferred-replay/project/output/coordinated-orbital-scale-v53/head.npz --output /tmp/human-cheek-replay
```

The command verifies the input head and writes exact numerical commands in the new root. It performs no solve. Run each phase under a two-CPU, 900 MiB sampled owned-tree guard with at least 2 GiB available; phases must release the shared execution lock. Saved `CHECKPOINT_REPLAY_RECEIPT.json` records the earlier fresh original replay: geometry/case/binding and solved numeric values matched exactly; timing metadata is expected to differ.

The included `workspace` contains exact sources, original numeric inputs, saved states and reports. `PAYLOAD.json` pins those original files. Builders refuse existing output locations. Perioral commands, from a fresh workspace containing the saved preceding insertion inputs, are:

```
B=human-preferred-v53-mechanics-20261011
blender -b -t 2 --python "$B/src/build_perioral_insertion_v3.py"
python "$B/src/create_perioral_cheek_case_v3.py"
CYBR_BILATERAL_BATCHES=24 node "$B/src/solve_loaded_cheek_secant_v1.mjs" "$B/cheek-perioral-case-v3" "$B/cheek-perioral-equilibrium-v3" neutral-0 0
CYBR_BILATERAL_BATCHES=24 node "$B/src/solve_loaded_cheek_secant_v1.mjs" "$B/cheek-perioral-case-v3" "$B/cheek-perioral-equilibrium-v3" right-02-initial .02
```

These are historical qualified states, not the new broad-domain expression experiment. Source exactness, geometric validity and finite saved-state checks do not certify whole-head injectivity or arbitrary interpolated motion. The original failed whole-midface rest-fit appears solely because its bone arrays are an input to the independent healthy cheek remesh; it is never a selected tissue domain.

## Licensing

The preserved source notices in `workspace/human-mechanics-20261010/reference-source/LICENSING.md`, `ASSET-LICENSES.md`, and `LICENSES/GPL-2.0.txt` are unchanged. Keep the owning repository's GPL-2.0 and CC BY-SA anatomical component attribution notices. Historical notices describe a broader atlas-based application; the current numeric regional bone fields and authored muscle routes do not newly include an atlas payload. This checkpoint does not relicense any existing component.
