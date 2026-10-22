"""Seven synthetic renderer tests; builds only in a temporary directory."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "src/supported-candidate/whole-face-continuum-staging"
source = base / "visualization/portable-renderer-0006"
names = ["finite_projection_and_rotation", "winding_no_backface_culling", "occlusion_independent_submission_order", "intersecting_triangles_use_per_pixel_depth", "shared_triangle_edge_has_no_holes", "degenerate_and_outside_finite_triangles", "exact_shared_coordinate_reconstruction"]
with tempfile.TemporaryDirectory(prefix="cybr-human-render-smoke-") as temporary:
    target = Path(temporary) / "supported-candidate/whole-face-continuum-staging/visualization/renderer"
    target.mkdir(parents=True)
    for file in source.iterdir():
        if file.suffix in {".py", ".cpp"} or file.name == "CMakeLists.txt":
            shutil.copyfile(file, target / file.name)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(base / "mechanics")
    subprocess.run([sys.executable, "-B", "-m", "unittest", *["test_renderer.RendererTests.test_" + name for name in names]], cwd=target, env=env, check=True)
