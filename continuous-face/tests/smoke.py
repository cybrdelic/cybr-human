"""Data-free source-integrity and synthetic unit smoke. Never runs a face solve."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "source-manifest.json").read_text())
for entry in manifest["files"]:
    path = ROOT / entry["target_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit("Source digest mismatch: " + entry["target_path"])
    if path.suffix == ".py":
        ast.parse(path.read_text(), filename=str(path))
base = ROOT / "src/supported-candidate/whole-face-continuum-staging"
mechanics = base / "mechanics"
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
env["PYTHONPATH"] = os.pathsep.join(str(p) for p in [mechanics, base / "contact", base / "contact/reconstruction-020-v1", base / "contact/interior-feature-tangent-v1"])
print(f"Verified {len(manifest['files'])} source files; running data-free synthetic fixtures.", flush=True)
subprocess.run([sys.executable, "-B", "-m", "unittest", "test_exterior_epigraph", "test_accepted_history"], cwd=mechanics, env=env, check=True)
