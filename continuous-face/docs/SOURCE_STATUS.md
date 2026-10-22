# Frozen versions and completed milestone

- Mechanics v14: 137 files; SHA256SUMS.json digest `fbb33acbcbae8beed888346b21b8a96be905899d122d61d6029bc9d9b73179ec`. The primary src tree restores those exact bytes into their runtime-relative layout. It is the source of the completed 25 ms paired milestone (24 saved states per case).
- Renderer 0005: manifest digest `e3fcace1742522f034a836ac39086cc5b9dcc176b693e72093d23ccd980b7510`. Retained separately for earlier receipt identities; it does not recognize the v14 controller.
- Renderer 0006: manifest digest `2301542febcb843ee592aedc541e474658fc5589cd9950701be88e0bc2492d33`. Matching reader for v14 controller `728fc856b3b61ab22b3209440a9ec04919fcb636dd3fa26991c9bf1cb55d7b28`.
- Mechanics v15: 138 files; manifest digest `685eb486297586ec9457046e6d0fec953f1fde027e0ec062e75ab44df372acf7`. It was adopted prospectively at the fully admitted 25 ms boundary and began running on 2026-10-22 at 13:10:08 UTC. It is not the source of the completed 25 ms history.

`variants/mechanics-v15/mechanics/` contains only the three exact v15 delta files: contact_query_bound_hint.py, coarse_reconstruction_020_curvature_driver.py and paired_reconstruction_020_curvature_sequence.py. All other v15 files are byte-identical to the corresponding v14 entries. The variant is not installed in src. It introduces optional authenticated conservative contact-query hint persistence and the controller opt-in forwarding hook, not a different physical law. The first enabled step is cold; pre-change partial sensitivity/QP state is not migrated. Matching renderer support for later v15 receipts is not certified by this handoff.

Original complete frozen manifests are under docs/source-pins/. Renderer manifests also enumerate excluded logs, data-validation receipts and native binaries, so they are historical complete-snapshot identities, **not** the source-only export's file list. `source-manifest.json` is the exact exported-source inventory. Parent material/contact/geometry helper files are independently hashed, not silently claimed to be part of v14.

No run was stopped, source refactored, algorithm changed, production snapshot edited, or new state admitted by this export. A separate repository task controls publication and repository visibility.
