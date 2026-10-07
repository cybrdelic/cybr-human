// One immutable build owns the renderer, workers, shaders and bound assets.
try {
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
  await import(runtime.entry);
} catch (error) {
  document.querySelector("#message").textContent =
    "Tissue startup failed: " + error.message;
  document.querySelector("#tissueStatus").textContent = error.message;
}
