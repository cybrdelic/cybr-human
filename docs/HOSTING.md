# Public hosting of the preserved application

The Sites identity is saved in `.openai/hosting.json`. Deployment is a static HTTPS site using `dist/`, rather than a new framework or simulation implementation. There are no hosted secrets, account connectors, server data, paid-plan changes or custom-domain purchases.

```powershell
npm run build:site
npm run test:site
```

The build allowlists the 69 reviewed entry/runtime/vendor paths, adds the root redirect and scoped licensing/attribution notices, and records decoded and transport hashes in `dist/deployment.json`. It never invokes the candidate publisher. The existing root opens `neutral-tissue.html`; runtime modules, model requests and the CPU module worker retain their original same-origin URLs. HTTPS provides the secure context needed by WebGPU. WebGL2 remains required for rendering; `?backend=cpu` preserves the worker fallback and its documented speed limits. SharedArrayBuffer is not used, so cross-origin-isolation headers are not required.

The selected `smooth-transfer.bin` is 27,038,784 bytes, exceeding Cloudflare's 25 MiB static-asset limit. Only this deployment payload uses lossless gzip transfer, with an explicit `Content-Encoding: gzip` rule in `dist/_headers`; browser HTTP decoding recovers the original bytes and SHA256 before model validation. The source binary and original runtime manifest remain unchanged. The build checks compression round-trip equality and the test fetches all 69 application paths through actual HTTP, verifying decoded sizes/hashes and JavaScript MIME. The hosting rules set JavaScript, JSON, GLB and binary MIME, an uncached runtime pointer, and immutable caching only for the content-addressed runtime directory. Missing assets receive 404 responses.

Custom static headers and the file-size limit follow [Cloudflare header documentation](https://developers.cloudflare.com/workers/static-assets/headers/) and [static asset limitations](https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/). Successful native Sites deployment status and public audience are recorded separately from local behavior validation. Existing actual scene/controls, GPU/CPU parity and worker/recovery tests apply because decoded application payloads are unchanged. Publication does not certify realtime or anatomical fidelity; see [measured limits](VALIDATION.md).

Source changes use the repository review flow. After merging, copy only tracked reviewed source to an isolated Site checkout, build and test it, and use the Sites hosting workflow to push that exact source state, package its output, save the version and deploy with public access. Keep raw QA traces, local profiles, backups, dependency caches and environment files outside both the source push and deployment archive. The static package does not contain offline anatomy-source arrays, construction tools or local QA history.
