# Portable renderer and platform checks

The derived renderer at `src/supported-candidate/whole-face-continuum-staging/visualization/portable-renderer-0006/` uses the exact frozen 0006 C++ rasterizer and the same mechanics/history readers. It changes build and tool discovery, library filenames, UTF-8 reads and build provenance. Its `PROVENANCE.json` records the source and derived Python hashes. The frozen 0005/0006 directories and all 211 original source-manifest entries remain unchanged.

## Fresh setup

Install Python 3.12 and a C++17 toolchain: Visual Studio C++ Build Tools on Windows, Xcode command-line tools on macOS, or GCC/Clang on Linux. CMake discovers that toolchain; a Visual Studio developer shell is not required when its normal CMake generator can locate the installation.

Create a local Python environment, then run from `continuous-face/`:

```sh
python -m pip install --no-compile -r requirements-observed.txt "cmake>=3.20"
python -B tests/smoke.py
python -B -m unittest discover -s tests -p test_portable_runtime.py
python -B tests/native_render_smoke.py
```

Use the environment's Python executable. On systems where only `python3` is installed, substitute that command.

The native test builds a real shared library and runs seven synthetic projection, occlusion, triangle-edge and coordinate-reconstruction tests. CMake emits `.dll`, `.dylib` or `.so` for the host. It exports the C ABI on Windows without editing the rasterizer equations. Build inputs are copied byte-for-byte to a short temporary cache to avoid MSBuild's legacy path-length limit. Compiler logs, commands, source identity and binary hash are recorded there. A per-build lock prevents concurrent writes; a cached binary is reused only when its recorded hash matches.

Set `CYBR_CMAKE` to a CMake executable, `CXX`/`CMAKE_GENERATOR` to select a supported toolchain, or `CYBR_NATIVE_BUILD_DIR` to choose the generated build-cache location. Keep custom Windows cache paths short. These settings affect only new builds, not historical proofs.

## Media tools

Install FFmpeg with ffprobe using your platform's package manager or official distribution. The derived readers use PATH discovery, with optional exact executable overrides `CYBR_FFMPEG` and `CYBR_FFPROBE`. Arguments are passed as a list, including paths with spaces; there is no shell interpolation or implicit installation.

```sh
python -B tests/media_tools_smoke.py
```

This encodes two synthetic color frames, then verifies their decoded count, size and playback rate with the actual ffprobe executable. It is a toolchain test, not a facial animation or scientific result.

Use the scripts in `portable-renderer-0006/` for new renders. Their accepted-frame, geometry/hash and output-directory checks remain active. They still require the authorized data and history described in [LOCAL_DATA.md](LOCAL_DATA.md). Platform support does not supply anatomy or make the full face solve self-contained. The fixed source snapshots remain available for historical Linux reproduction; never reuse a historical binary hash or rewrite an old proof to match a new toolchain.

## Coverage

The `Continuous face portability` workflow compiles and runs the mechanics, native pixel and media-tool checks on Windows, Linux and macOS. A workflow definition is not a passing result: inspect the run for the exact PR commit. No physical mobile device, full facial solve, anatomy reconstruction or full expression movie is exercised by these checks.

The native component is CPU C++17, with no GPU dependency. Browser projects elsewhere in this repository retain separate requirements; this change covers the migrated continuous-face source collection.

[CMake's Windows C ABI export support](https://cmake.org/cmake/help/latest/prop_tgt/WINDOWS_EXPORT_ALL_SYMBOLS.html) is used by the build.
