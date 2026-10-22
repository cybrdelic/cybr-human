# Runtime, build and test requirements

## Observed, not locked

This handoff was checked on Linux with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Pillow 12.3.0. `requirements.in` records direct third-party imports; `requirements-observed.txt` records observed versions only. No clean environment installation or dependency resolution was performed. There was no project pyproject/packaging build file or proven transitive lockfile in the selected sources. NumPy/SciPy supply compiled numerical dependencies through their installed distributions.

- Mechanics/contact/geometry: Python standard library, NumPy and SciPy (sparse matrices/SuperLU, optimization, spatial trees, ndimage and linear algebra).
- Renderer: NumPy, Pillow, Python ctypes, and a C++17 compiler. Observed `/usr/bin/c++` is Debian GCC 14.2.0.
- MP4 encoding and decoded-frame verification: ffmpeg and ffprobe at hard-coded `/usr/bin/` paths. Observed ffmpeg is 7.1.5-0+deb13u1. Movie encoding is not exercised by the minimal smoke.
- Tests use standard-library unittest, not pytest. No GPU/CUDA or PyTorch import was found in the selected source closure. Optional geometry helper branches remain incomplete as described in LOCAL_DATA.md.

The actual renderer build command is:

```sh
/usr/bin/c++ -O3 -std=c++17 -fPIC -shared rasterizer.cpp -o rasterizer.so
```

This is a local shared-library C ABI consumed by ctypes, not a Python extension package. No prebuilt .so is shipped. Portability to Windows/macOS or another ABI is unverified; adapt build suffixes/flags/tool locations before claiming support. Do not reuse a binary hash across compilers.

## Verified cheap checks

`python3 -B tests/smoke.py` verifies all staged source hashes, parses every Python source and runs 14 synthetic exterior-epigraph/accepted-history tests. It passed in the handoff environment. It does not load full anatomy or run a physical solve.

`python3 -B tests/native_render_smoke.py` builds the renderer only in a temporary directory and runs seven synthetic pixel/reconstruction tests. It does not load anatomy, create a facial movie, or claim visual/scientific validation.

Run with OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 for bounded checks; the supplied scripts set these. Test outputs and exact verification metadata are retained outside the candidate repo payload by the handoff producer.
