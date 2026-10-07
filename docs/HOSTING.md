# Public hosting of the preserved application

The existing Sites identity is saved in `.openai/hosting.json`. Deployment is a static HTTPS site using `dist/`. No simulation rebuild is needed.

```powershell
npm run build:site
npm run test:site
```

The build allowlists 70 reviewed entry/runtime/vendor paths, including the runtime-transfer adapter, adds the root redirect and licensing notices, and records original and transport hashes in `dist/deployment.json`. It never invokes the candidate publisher. The root opens `neutral-tissue.html`; runtime modules, model requests and CPU module workers retain their same-origin URLs. HTTPS provides the secure context needed by WebGPU. WebGL2 remains required for rendering; `?backend=cpu` preserves the worker fallback and its documented speed limits. SharedArrayBuffer is not used.

The selected `smooth-transfer.bin` is 27,038,784 bytes, exceeding the host's 25 MiB static-asset limit. Only this deployment payload uses lossless gzip transfer. The source binary, archived runtime and model checksum remain unchanged. Although `dist/_headers` declares `Content-Encoding: gzip`, actual production responses were observed to ignore this file and return compressed bytes without the header. The old loader rejected those bytes as `Stale smooth tissue transfer`, then hid its startup error, leaving a blank viewport. Successful deployment status did not prove functional startup.

The bootstrap now normalizes only this exact same-origin model asset. Native HTTP-decoded bytes pass through; headerless gzip is decoded using the browser's `DecompressionStream`. The preserved model loader still authenticates the decoded bytes against its original SHA256. If loading fails, a visible error and working retry button remain over the viewport. Loading progress remains visible until the preserved viewer reports ready. This does not change solver, rendering or model math.

The packaging test fetches all 70 application paths through actual HTTP and verifies original sizes/hashes and JavaScript MIME. It additionally serves the compressed model both with and without its encoding header and verifies the original checksum through the bootstrap adapter. The earlier header-respecting local test missed the production gateway behavior; it is not sufficient evidence of live startup. Browser checks must also inspect the actual face, controls/reset and visible failure state.

The size limit follows [Cloudflare static asset limitations](https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/). `_headers` rules follow [Cloudflare header documentation](https://developers.cloudflare.com/workers/static-assets/headers/), but this Sites gateway was observed not to apply them. Production relies on the gateway's actual JavaScript MIME and browser-side model normalization. Realtime, anatomical fidelity and physical Android device acceptance remain separate from loading checks; see [measured limits](VALIDATION.md).

Use reviewed source commits in an isolated Site checkout, build and test it, and use the Sites hosting workflow to push that exact source, package its output, save the version and deploy to the existing Site. Preserve its current public audience. Keep QA traces, profiles, backups, dependency caches and environment files outside the source push and deployment archive. The static package excludes construction arrays/tools and local QA history.
