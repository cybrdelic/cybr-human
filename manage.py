"""Single entry point for the supported tissue application."""

import argparse
from pathlib import Path
import subprocess
import sys
import os
import json

TESTS = [
    "compression-barrier.mjs",
    "controls.mjs",
    "slider-range.mjs",
    "build-contract.mjs",
    "recovery.mjs",
    "context.mjs",
    "fine-skin.mjs",
    "parity.mjs",
    "diffusion.mjs",
    "mechanics.mjs",
]

ROOT = Path(__file__).resolve().parent


def run(*arguments):
    subprocess.run(arguments, cwd=ROOT, check=True)


def verify(candidate=False, port=8779, suite="all"):
    (ROOT / "output/verification").mkdir(parents=True, exist_ok=True)
    pointer = (
        ROOT
        / "output"
        / ("runtime-candidate.json" if candidate else "runtime-current.json")
    )
    build = json.loads(pointer.read_text(encoding="utf-8"))["build"]
    environment = {
        **os.environ,
        "TISSUE_CANDIDATE": "1" if candidate else "0",
        "TISSUE_BUILD": build,
        "TISSUE_BASE_URL": f"http://127.0.0.1:{port}",
    }
    if suite in ("offline", "all"):
        for script in ["source-graph.py", "rest-shape.py", "head-topology.py"]:
            subprocess.run([sys.executable, str(ROOT / "tests" / script)],
                           cwd=ROOT, env=environment, check=True)
        subprocess.run(["node", str(ROOT / "tests/offline-worker.mjs")],
                       cwd=ROOT, env=environment, check=True)
    if suite in ("browser", "all"):
        for script in TESTS:
            subprocess.run(["node", str(ROOT / "tests" / script)],
                           cwd=ROOT, env=environment, check=True)
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8779)
    build = commands.add_parser("build")
    build.add_argument("--port", type=int, default=8779)
    build.add_argument(
        "--geometry",
        action="store_true",
        help="Regenerate head, volume and bindings before publishing",
    )
    verification = commands.add_parser("verify")
    verification.add_argument("--candidate", action="store_true")
    verification.add_argument("--port", type=int, default=8779)
    verification.add_argument("--suite", choices=["offline", "browser", "all"], default="all")
    args = parser.parse_args()
    if args.command == "serve":
        run(sys.executable, "-m", "http.server", str(args.port), "--bind", "127.0.0.1")
    elif args.command == "build":
        if args.geometry:
            for script in [
                "build_neutral_head.py",
                "build_neutral_tissue.py",
                "build_neutral_tissue_assets.py",
            ]:
                run(sys.executable, str(ROOT / "tools" / script))
        run(sys.executable, str(ROOT / "tools/build_tissue_runtime.py"), "--no-publish")
        candidate = ROOT / "output/runtime-candidate.json"
        expected = json.loads(candidate.read_text(encoding="utf-8"))["build"]
        verified = verify(candidate=True, port=args.port)
        raw = candidate.read_bytes()
        if verified != expected or json.loads(raw)["build"] != expected:
            raise RuntimeError(
                "Candidate changed during verification; refusing promotion"
            )
        pending = ROOT / "output/runtime-current.pending"
        pending.write_bytes(raw)
        os.replace(pending, ROOT / "output/runtime-current.json")
        print("Verified and promoted runtime " + expected)
    else:
        verify(candidate=args.candidate, port=args.port, suite=args.suite)


if __name__ == "__main__":
    main()
