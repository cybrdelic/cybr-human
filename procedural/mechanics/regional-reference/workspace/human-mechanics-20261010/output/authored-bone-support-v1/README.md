# Original bounded zygomatic / maxillary support

This is an editable, original regional bony support for the native v16 face. It is not a scan, atlas import, full skull, calibrated anatomical reconstruction, or a completed bone/contact simulation. Native cheek v2 remains unchanged.

## Geometry

The common watertight rigid surface organizes five authored regions: zygomatic body, posteriorly extending arch, thin maxillary facial shell, inferior orbital rim, and lateral orbital rim. The maxillary facial surface has an authored canine-fossa recess; the orbital margins reserve the native eye/socket region. The arch terminates at a deliberate posterior cut. Semantic bone IDs distinguish maxillary and zygomatic portions but do not represent a modeled suture or independently moving bones.

The editable source is `src/zygomatic_support_parameters.json` and `src/zygomatic_support_build.py`. Named control curves, body radii, shell outline, depth parameters and landmark seeds drive an original implicit construction. An original marching-tetrahedra implementation extracts the mesh. The existing original renderer produces the preview directly from geometry; no generated appearance, external geometry or reference pixels are used.

Files:
- `bone_support.npz`: surface vertices/triangles, semantic region/bone/component IDs, per-triangle conservative skin-clearance bounds
- `bone_support.obj`: editable interchange mesh with named face groups
- `landmarks.json`: actual projected mesh landmarks and a zygomaticus origin patch
- `mechanics_interface.json`: coordinate/scale contract and projections of all five existing v2 origins
- `audit.json`: topology, volume, winding, native skin and globe checks
- `authored_support_cutaway.png`: actual original mesh, isolated and inside a native-skin cutaway; only display triangles are hidden

## Verification and limits

For each support-triangle centroid, an original exact triangle BVH evaluates nearest native skin distance and three independent oblique ray parities. Subtracting that triangle's maximum centroid-to-vertex distance supplies a conservative clearance bound over its entire area. Positive bounds plus interior parity certify the full surface under the closed-source-mesh assumption and ordinary floating-point tolerances. This is stronger than vertex-only containment. An independent closest-point check uses the actual globe sphere.

No attachment has yet been rebound in the FEM solve. Do not substitute the mesh for old tangent-plane contacts without implementing actual signed mesh contact, unit-consistent tolerances, and a subsequent deformed-state audit. Design coordinates are not measured millimetres.

## Primary references used for organization only

1. Sarilita et al., *Zygomaticus major muscle bony attachment site: a Thiel-embalmed cadaver study* (2021). A 52-dissection study locates the origin near the superior margin of the temporal process on the zygomatic lateral surface. This informs the named superior-lateral origin patch; no reported measurements or meshes were copied. https://pubmed.ncbi.nlm.nih.gov/32807628/
2. Song et al., *Topographic anatomy of the zygomatic arch and temporal fossa: a cadaveric study* (2009). Seventy cadavers inform the arch's long, slender lateral organization and distinction between anterior, middle and posterior portions. Its numeric measurements are not used as native-model calibration. https://pubmed.ncbi.nlm.nih.gov/18948070/
3. AO Foundation Surgery Reference, *Zygoma, zygomatic complex fracture* and its four-point reconstruction guidance. The author's own clinical reference organizes the zygomaticomaxillary buttress, arch, inferior orbital rim and lateral orbital wall as distinct structural landmarks. https://surgeryreference.aofoundation.org/cmf/trauma/midface/zygoma-complex/definition and https://surgeryreference.aofoundation.org/cmf/trauma/midface/zygoma-complex/orif-4-point-fixation-with-orbital-reconstruction

## Reproduce

From the workspace, run the build and then preview through the shared heavy-work lock. The build enforces at least 125 MiB available storage and writes only this separate output folder. Parameters are copied with the output. Native source SHA-256: `f4fa73042fab1534b45c0982b47ae668b16eb990c2029e6a06cfc9b94b01df14`.
