# Eye and eyelid coverage review

Evidence: actual neutral/passive/pull screenshots from immutable runtime
`302159721eef46e6f4a3a451`, 1280 x 900 at DPR 1, three controlled lights and four
independent optical flags. The detail sheet enlarges a 220 x 65-pixel eye crop
uniformly by 1.455; it exposes native pixel steps rather than restoring detail.
Regional pigmentation changes lips but preserves the white-eye steps. Refraction
changes the iris response but preserves lid/brow boundary speckling. This is a
limited optical improvement, not a completed large appearance upgrade.

## Supported causes and limits

* **Offscreen aliasing is confirmed in source.** `WebGLRenderer` has canvas MSAA,
  but `skin_diffusion.js` creates single-sample half-float beauty and diffuse
  targets. All geometry, including eyes and real hair tubes, is rasterized into
  that beauty target before compositing. The canvas AA cannot recover those
  sample decisions. Hair radii are 0.03–0.055 mm, taper to 15%, and use five-sided
  tubes: many hairs are subpixel at the captured scale. This supports coverage
  aliasing as a major contributor, but requires a controlled GPU ablation to
  quantify the visible contribution.
* **The limbus cut is clean in the actual asset.** Both baked sclera rings have
  radius 5.719998–5.720002 mm. The old centroid-cut jagged seam is not present.
  The iris is a recessed authored disk; its rim is about 0.052 mm behind the
  sphere. The procedural corneal shell is independently placed from the iris
  bounding box and sits about 0.225 mm behind the sphere at the limbus. Those
  independent surfaces do not enforce a continuous joined optical profile.
  These construction gaps do not, by themselves, prove a visible hole or
  explain every white pixel.
* **Iris detail is undersampled before rasterization.** The 192 angular segments
  receive a vertex-color sine with frequency 143 (above angular Nyquist 96),
  plus additional modulation. Filtering the screen later cannot undo that
  vertex-sampled alias. A later analytic, derivative-filtered iris shader is
  preferable to increasing tint or segment count blindly.
* **Contact is authored, not enforced.** Skin, lid margin and hairs transfer FEM
  motion; sclera, iris, pupil and conjunctival sleeves explicitly do not. Passive
  deformation can change their relative coverage without reciprocal contact or
  moving sleeves. The generator's current opening differs from the stored lid
  centreline by up to 1.500001 mm. The actual head's shape-delta buffer is zero.
  Rebuilding from newer editable generators would therefore change the captured
  geometry; it is not an equivalent replacement for the baked baseline.

## Minimal prepared fix

`?surfaceCoverage=msaa` requests four samples for **both** beauty and skin-diffuse
targets, retaining resolved depth for the existing occlusion gates. It intersects
the actual RGBA16F and DEPTH_COMPONENT24 supported sample counts, choosing four,
then two, then the original zero-sample path. Blur targets remain single-sample.
The default and original diffusion kernel/radius stay unchanged. Telemetry exposes
requested and selected counts. This addresses raster coverage rather than tint,
and introduces extra target memory/work that must be measured before adoption.

A later bounded GPU review paired identical optics at DPR 1 in neutral
three-quarter studio/side/overhead views. Actual four samples were selected;
both solves converged with identical sampled positions and no errors. Reviewed
pixels show reduced eye-edge steps and brow speckling. Passive/pull coverage,
closer native-pixel views, a diffusion-bypass ablation, other devices and target
memory/work still need acceptance. Do not infer anatomical contact from MSAA.

After coverage is isolated, the smallest geometry lane should derive a shared
limbus and lid aperture from the **baked** vertices, join the corneal profile to
that limbus and add a consistent lid-to-globe inner surface. Validate tangency,
non-inversion and visible occlusion under the existing held loads; add explicit
moving/reciprocal contact separately if physical coupling is requested. Neither
render-only closure nor passive numerical stability attests physiology. The
remaining waxy response also needs controlled BRDF/light/transport evaluation.
