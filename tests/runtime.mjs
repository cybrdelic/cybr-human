import fs from "node:fs";
fs.mkdirSync("output/verification", { recursive: true });
export const pointerPath = process.env.TISSUE_CANDIDATE === "1"
  ? "/output/runtime-candidate.json" : "/output/runtime-current.json";
export const baseURL = process.env.TISSUE_BASE_URL || "http://127.0.0.1:8779";
export const solverURL = new URL("/tests/solver.html", baseURL).href;
const app = new URL("/neutral-tissue.html", baseURL);
if (process.env.TISSUE_CANDIDATE === "1") app.searchParams.set("candidate", "1");
if (process.env.TISSUE_BUILD) app.searchParams.set("expectedBuild", process.env.TISSUE_BUILD);
export const appURL = app.href;
export const expectedBuild = process.env.TISSUE_BUILD || "";
