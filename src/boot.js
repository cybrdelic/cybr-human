// One immutable build owns the renderer, workers, shaders and bound assets.
const message = document.querySelector("#message");
const status = document.querySelector("#tissueStatus");
const startup = (window.__tissueStartup = { state: "loading", error: null });
message.hidden = false;
message.textContent = "Loading face data.";
message.setAttribute("role", "status");
let monitor;
function failStartup(error) {
  clearInterval(monitor);
  startup.state = "failed";
  startup.error = error?.message || String(error);
  message.hidden = false;
  message.setAttribute("role", "alert");
  message.textContent = "The face could not load. " + startup.error + " ";
  const retry = document.createElement("button");
  retry.textContent = "Retry loading";
  retry.style.pointerEvents = "auto";
  retry.onclick = () => location.reload();
  message.append(retry);
  status.textContent = "Face loading failed: " + startup.error;
}
// The preserved viewer starts asynchronously and handles its own rejection.
// Its old catch hides the centre message, so keep startup failure visible here.
monitor = setInterval(() => {
  const debug = window.__fullHeadFEM;
  if (debug?.errors?.length && !debug.ready) {
    failStartup(Error(debug.errors.at(-1)));
  } else if (debug?.ready) {
    startup.state = "ready";
    clearInterval(monitor);
    message.hidden = true;
  } else if (message.hidden) {
    message.hidden = false;
    message.textContent = "Loading face data. " + (status.textContent || "");
  }
}, 100);
try {
  const { createRuntimeFetch } = await import("./runtime_transport.js");
  const response = await fetch(
    new URLSearchParams(location.search).get("candidate") === "1"
      ? "/output/runtime-candidate.json"
      : "/output/runtime-current.json",
    {
      cache: "no-store",
    },
  );
  if (!response.ok)
    throw Error("No published tissue runtime. Run python manage.py build.");
  const runtime = await response.json();
  const expected = new URLSearchParams(location.search).get("expectedBuild");
  if (expected && expected !== runtime.build)
    throw Error("Runtime changed during verification");
  window.__tissueRuntime = runtime;
  window.fetch = createRuntimeFetch(
    window.fetch.bind(window),
    new URL(runtime.tissueBase + "smooth-transfer.bin", location.href),
  );
  await import(runtime.entry);
} catch (error) {
  failStartup(error);
}
