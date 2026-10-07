// Stop submissions when the page cannot safely use its graphics session.
export function guardGPUPage({
  document,
  window,
  canvas,
  onStop,
  onDispose,
  onContextLost,
  onContextRestored,
}) {
  let failed = false,
    disposed = false,
    suspended = false,
    contextLost = false;
  const fail = (reason) => {
    if (failed || disposed) return;
    failed = true;
    onStop(reason);
  };
  const lost = (event) => {
    event.preventDefault();
    if (onContextLost) {
      contextLost = true;
      onContextLost();
    } else
      fail(
        "The graphics context was lost. Reload this page to start a new session.",
      );
  };
  const restored = () => {
    if (!disposed && !failed && contextLost) {
      contextLost = false;
      onContextRestored?.();
    }
  };
  const hide = (event) => {
    suspended = true;
    if (!event.persisted) dispose();
  };
  const show = () => {
    suspended = false;
  };
  function dispose() {
    if (disposed) return;
    disposed = true;
    canvas.removeEventListener("webglcontextlost", lost);
    canvas.removeEventListener("webglcontextrestored", restored);
    window.removeEventListener("pagehide", hide);
    window.removeEventListener("pageshow", show);
    onDispose();
  }
  canvas.addEventListener("webglcontextlost", lost);
  canvas.addEventListener("webglcontextrestored", restored);
  window.addEventListener("pagehide", hide);
  window.addEventListener("pageshow", show);
  return {
    get canSubmit() {
      return (
        !failed && !disposed && !suspended && !contextLost && !document.hidden
      );
    },
    get disposed() {
      return disposed;
    },
    fail,
    dispose,
  };
}
