# Preferred-v53 cheek: independent attachment/reaction audit

## Result

The saved regional muscle is mechanically clamped. Its 176 exact bone-footprint nodes and 176 additional collar nodes are all fixed in three world-space coordinates. All 352 positions are bit-exact to rest in every supplied state. This is a static rigid attachment approximation.

The exact footprint does not carry nonzero constitutive traction in this setup. All 903 tetrahedra touching root nodes have only fixed vertices, hence stay undeformed and force-free. The additional fixed collar takes the reaction from the deforming passive adapter. No active tetrahedron touches a fixed node. Both facts follow from the actual connectivity, not only descriptive metadata.

## Extracted loads

Values below are uncalibrated solver force units; they are not measured human forces.

| State | Collar nodal-force L2 | Largest collar nodal force | Collar net-force norm | Free-DOF residual |
|---|---:|---:|---:|---:|
| neutral-0 | 0 | 0 | 0 | 1.0047561e-17 |
| right-02 | 7.8063234e-06 | 2.4746482e-06 | 5.5036506e-07 | 1.9204973e-05 |
| right-08 | 2.9578433e-05 | 1.1801375e-05 | 6.8411239e-07 | 1.4632669e-05 |
| right-16 | 5.8879744e-05 | 2.5631036e-05 | 7.1058111e-07 | 1.4624537e-05 |
| released-0 | 8.0965948e-06 | 3.866522e-06 | 6.6304286e-07 | 1.5306806e-05 |

Every exact-root reaction and every finite nodal bone-contact force is zero in all five states. Nonzero collar loads rise with activation and become smaller after release. The nonzero net-force norm is smaller than the free residual. With no active contact, gravity, springs or other external load, that residual-level net should not be presented as a resolved muscle pulling force; localized collar reactions can largely cancel.

## Honest demonstration

- Show the exact bone footprint and the larger fixed collar separately, with zero displacement at both.
- Put reaction arrows at the collar nodes where the calculated loads act. Show the force scale and local-force statistic; do not use the small net as a strength estimate.
- If treating the collar as a rigid extension of a static bone, its summed force and moment can be reported as an inferred rigid-support wrench. Label that rigid-adapter assumption; this is not a simulated stress transmission through the root cells or a reciprocal bone-dynamics solve.
- The five states demonstrate force-law-driven regional solid deformation and release under a fixed attachment. They do not establish a complete anatomical face or calibrated pose rig.

## Remaining model gaps

The source constructs one authored right-zygomatic-like active body and passive origin adapter. There is no segmented anatomy, physiological activation dynamics, tendon law, full skull/mandible mechanics, or bilateral/oral/lid/jaw/neck assembly. The active body is not embedded in a solved surrounding face volume, and the supplied case contains no separate force-bearing distal insertion constraint or surrounding skin load. Skin-transfer work is outside this audit and must be assessed independently; a visual transfer alone would not add reciprocal force coupling.

## Reproducibility and checks

`extract_reactions.mjs` loads the unchanged case and each saved state, reproduces the stored free residual exactly, then clears only the in-memory fixed mask for an energy/gradient evaluation at the same positions. It does not step or optimize. A second evaluation without contacts separates constitutive and contact gradients. Clearing the mask leaves free gradients and energy exactly unchanged. Six central-difference checks of the largest .16 collar-node reaction agree within 2.18e-13 absolute. All input SHA-256 values are unchanged before/after.

`reactions.json` contains the aggregate results, hash identities and sign conventions. `fixed_node_reactions.json` contains all fixed-node vectors for plotting. `run.log` records the shared-lock, CPU-bounded run: 3.022 seconds, 403.7 MiB peak sampled RSS, no guard stop.

Source references: `src/regional_case_io.mjs:17-40` verifies exact roots, removes root penalty contacts and creates the static bonded token; `src/build_preferred_cheek_volume_v9.py` builds fixed root/collar layers and activates only body cells; `src/create_cheek_case_v3.py:17` defines the one-muscle domain and its scope. In the original solver, `reference-source/src/layered_fem.js:99-104,298-304,488-495` implements pins, projects gradients and enforces targets. `src/volumetric_muscle_fem.mjs:127-199` implements the active constitutive gradient. These source paths under the original solver are relative to `human-mechanics-20261010`.
