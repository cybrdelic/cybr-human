/** Optional anatomy never delays face readiness. Requests share one load and
 * keep the skin visible until a verified scene is ready; a failure is retryable. */
export function createLazyAnatomy({ load, attach, show, discard, isDisposed, onState }) {
  let value = null, pending = null, wanted = false;
  const state = { phase: "idle", error: null, attempts: 0 };
  const notify = () => onState?.({ ...state });
  async function setVisible(visible) {
    wanted = visible;
    if (isDisposed()) return;
    if (!visible || value) { show(value, visible && !!value); return; }
    if (!pending) {
      state.phase = "loading"; state.error = null; state.attempts++; notify();
      pending = Promise.resolve().then(load).then(scene => {
        if (isDisposed()) { if (scene) discard(scene); return; }
        if (!scene) throw Error("Registered anatomy is unavailable");
        value = scene; attach(scene); state.phase = "ready"; notify();
      }).catch(error => {
        if (isDisposed()) return;
        wanted = false; state.phase = "failed"; state.error = error.message;
        show(value, false); notify();
      }).finally(() => { pending = null; });
    }
    await pending;
    if (!isDisposed()) show(value, wanted && !!value);
  }
  return { state, setVisible };
}
