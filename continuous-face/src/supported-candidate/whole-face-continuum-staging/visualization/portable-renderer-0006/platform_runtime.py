"""Platform tools and CMake build for the derived renderer; no mechanics changes."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def resolve_tool(name):
    """Honor a configured executable, then PATH. Never execute a shell string."""
    variable = "CYBR_" + name.upper()
    configured = os.environ.get(variable)
    executable = shutil.which(configured or name)
    if not executable:
        if configured:
            raise RuntimeError(f"{variable} does not name an executable: {configured}")
        raise RuntimeError(f"{name} was not found on PATH. Install it or set {variable} to its executable path.")
    return str(Path(executable).resolve())

def library_suffix():
    system = platform.system()
    if system == "Windows":
        return ".dll"
    if system == "Darwin":
        return ".dylib"
    if system == "Linux":
        return ".so"
    raise RuntimeError(f"Native renderer platform is not supported: {system}")

@contextmanager
def build_lock(path, timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Native build is locked: {path}. Check for another build before removing a stale lock.")
            time.sleep(.1)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock:
            lock.write(str(os.getpid()))
        yield
    finally:
        Path(path).unlink(missing_ok=True)

def build_library():
    cmake = resolve_tool("cmake")
    identity = {
        "source_sha256": sha_file(HERE / "rasterizer.cpp"),
        "cmake_sha256": sha_file(HERE / "CMakeLists.txt"),
        "platform": platform.system(),
        "machine": platform.machine(),
        "cmake_executable": cmake,
        "cmake_generator": os.environ.get("CMAKE_GENERATOR", ""),
        "cxx": os.environ.get("CXX", ""),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    build_root = Path(os.environ.get("CYBR_NATIVE_BUILD_DIR", str(Path(tempfile.gettempdir()) / "cybr-human-native"))).expanduser().resolve()
    build = build_root / fingerprint
    build.mkdir(parents=True, exist_ok=True)
    library = build / "lib" / ("rasterizer" + library_suffix())
    record = build / "build.json"
    with build_lock(build / ".lock"):
        if library.exists() and record.exists():
            try:
                previous = json.loads(record.read_text(encoding="utf-8"))
                if previous.get("identity") == identity and previous.get("library_sha256") == sha_file(library):
                    return library
            except (OSError, ValueError):
                pass
        # Keep MSBuild intermediates and source paths below legacy Windows path limits.
        # These two build inputs are byte copies with identities recorded above.
        build_source = build / "source"
        build_source.mkdir(exist_ok=True)
        for name in ("rasterizer.cpp", "CMakeLists.txt"):
            (build_source / name).write_bytes((HERE / name).read_bytes())
        commands = [
            [cmake, "-S", str(build_source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
            [cmake, "--build", str(build), "--config", "Release", "--clean-first", "--parallel", "2"],
        ]
        logs = []
        for command in commands:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            logs.append({"command": command, "exit_code": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr})
            if completed.returncode:
                (build / "failed-build.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
                raise RuntimeError("Native CMake build failed. Install a C++17 toolchain "
                                   "(Visual Studio C++ Build Tools, Xcode command-line tools, or GCC/Clang). "
                                   f"Details: {build / 'failed-build.json'}\n{completed.stdout}\n{completed.stderr}")
        if not library.is_file():
            raise RuntimeError(f"CMake did not produce the expected native library: {library}")
        payload = {"identity": identity, "library_sha256": sha_file(library), "steps": logs}
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(record)
    return library
