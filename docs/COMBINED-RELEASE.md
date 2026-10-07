# Combined HUMAN release candidate

Proposed runtime: `0120e716779dc503dadfc9dd`. Selected runtime and public Site are
unchanged until combined acceptance and parent-controlled promotion.

The default Refined appearance combines transferred smooth normals, authored
regional skin/lip response, refractive corneal shells, existing diffuse transport
and format-supported 4x/2x offscreen coverage. Classic selects the former skin and
specular-shell response with single-sample targets. These are rendering choices;
eye/lid geometry and mechanical material fields are not regenerated or calibrated.

Balanced defaults to the ordered two-bank pipeline on WebGPU adapters that do not
need bounded submissions. CPU, unavailable WebGPU and bounded adapters retain the
full-state serial path. Serial fallback is selectable. Both modes keep full state
validation, actual-velocity checkpoints and recovery/picking ownership. Public
choices explain that switching reloads the face. Query overrides remain available.

`rollback.html?rollback=1` loads the exact preserved prior page and runtime
`4bc8b53a8f15b8b62cf6a86e`. CPU closure checks independently verify those preserved
bytes against the original staged provenance. The new bootstrap retains headerless
gzip normalization and visible startup error/retry. Hidden anatomy remains lazy.

`node tools/build_site.mjs --candidate` stages `dist-candidate` without changing
the selected source pointer. Its current pointer selects the candidate **inside
the package only**, and it includes the prior runtime and rollback pointer. This
is a proposed package, not a published or accepted release. HTTP checks verify
decoded hashes, MIME, both module/worker closures, headerless gzip, rollback and
exclusion of private/inactive paths. Its largest transport asset is under 25 MiB.

## Evidence and remaining acceptance

The earlier isolated pipeline `6f661aa...` had exact packed-state and actual-
velocity equality over 30 matched steps and accepted-checkpoint recovery after
discarding two outstanding solves. One ABBA reference-quality workload advanced
5.30/5.28 simulation seconds in approximately 12.2 wall seconds versus 3.35/2.98
for serial: about 1.67x aggregate advancement in that experiment. Both remained
behind real time. Pipeline map stalls reached 539/526 ms versus serial 431/416 ms;
do not claim reduced worst latency or a general 1.67x gain. Pipeline input pulses
reached accepted render submission in 63/80 ms; serial pulse polling gives only a
coarse one-second bound. Submission is not displayed FPS or photon latency.

The pipeline's owned solver GPU buffers increased from 25,419,072 to 26,305,312
bytes (+886,240). That excludes renderer/native/driver memory. MSAA target memory
and total combined application peak remain unmeasured. Do not advertise a full
memory result. Isolated MSAA neutral images used actual four samples, converged
with identical 188 sampled position vectors and no application/GPU errors; pixel
review shows reduced eye steps and brow speckling. Waxy response and authored eye
contact limitations remain. This is an incremental improvement, not completed
photorealism or physiological accuracy.

Six CPU contracts, offline runtime/privacy/hash/geometry/worker checks, calibration
fixtures and source syntax are checked independently. The combined defaults and
new package still need browser acceptance. In an exclusive GPU slot serve
`dist-candidate` on 8783 and run `node tests/combined-release-browser.mjs` from this
checkout. One owned browser has a 360-second close watchdog; request at most eight
minutes including launch/closure. It checks default packaged startup, lazy anatomy,
repeated controls, real pointer drag/release, pipeline overlap, reset during running
work, actual device loss on two fresh desktop/mobile viewport contexts, recovered
CPU controls, Classic/Serial reloads and the baseline rollback link. Physical
Android, global contact, subject calibration and all-model driver support remain
unverified. Stop and release immediately on completion/failure, before processing
media or documentation. Parent performs promotion only after combined acceptance.
