/** Bounded measurement of render submissions. These are NOT presentation/GPU
 * completion timestamps; browser trace/captures are required for displayed FPS. */
export function createFrameMetrics(now = () => performance.now(), capacity = 240) {
  const intervals = [], costs = [];
  let last = null, count = 0, worstGap = 0, worstCost = 0;
  const append = (list, value) => { list.push(value); if (list.length > capacity) list.shift(); };
  const percentile = (list, fraction) => {
    if (!list.length) return null;
    const sorted = [...list].sort((a, b) => a - b);
    return sorted[Math.ceil(fraction * sorted.length) - 1];
  };
  return {
    submitted(begin, end = now()) {
      if (last !== null) { const gap = begin - last; append(intervals, gap); worstGap = Math.max(worstGap, gap); }
      last = begin; count++;
      const cost = end - begin; append(costs, cost); worstCost = Math.max(worstCost, cost);
    },
    suspend() { last = null; },
    snapshot(clock) {
      return {
        renderSubmissions: count, windowSamples: costs.length,
        submissionGapP50Ms: percentile(intervals, .5), submissionGapP95Ms: percentile(intervals, .95),
        renderCpuP50Ms: percentile(costs, .5), renderCpuP95Ms: percentile(costs, .95),
        worstSubmissionGapMs: worstGap, worstRenderCpuMs: worstCost,
        simulationToActiveWallRatio: clock?.activeWallSeconds > 0 ? clock.simulationSeconds / clock.activeWallSeconds : null,
        timingContract: "CPU render submissions; no claim of displayed FPS or GPU completion",
      };
    },
  };
}
