import { WrinklePlate } from "./wrinkle_plate.js";
const plate = new WrinklePlate();
self.onmessage = ({ data }) => {
  try {
    const start = performance.now(),
      report = plate.solve(data.F, { iterations: 1200 }),
      height = Float32Array.from(plate.h);
    let minimumStretch = Infinity;
    for (let j = 0; j < data.F.length; j += 4)
      minimumStretch = Math.min(minimumStretch, data.F[j], data.F[j + 3]);
    self.postMessage(
      {
        height,
        report: {
          ...report,
          elapsedMs: performance.now() - start,
          minimumStretch,
        },
      },
      [height.buffer],
    );
  } catch (e) {
    self.postMessage({ error: e.message });
  }
};
