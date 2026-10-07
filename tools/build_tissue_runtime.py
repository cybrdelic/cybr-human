"""Publish an immutable renderer/worker/shader and numerical asset snapshot.

The pointer is replaced only after all files are present. Existing browser sessions
keep their own build; a reload resolves the current pointer without stale modules.
"""

from pathlib import Path
import errno
import shutil
import hashlib, json, re, os, argparse

ROOT = Path(__file__).resolve().parent.parent
ENTRY = "src/viewer.js"
BASE = "output/neutral-tissue/v6/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def materialize_blob(blob, destination):
    """Preserve snapshot bytes on filesystems without hard-link support."""
    try:
        os.link(blob, destination)
    except OSError as error:
        unsupported = error.errno in (errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.ENOSYS)
        unsupported = unsupported or getattr(error, "winerror", None) in (1, 50)
        if not unsupported:
            raise
        shutil.copy2(blob, destination)


def main(base=BASE, publish=True):
    files = {}

    def code(name):
        name = name.split("?")[0]
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT / "src"):
            raise ValueError("Runtime modules must live in src/: " + name)
        name = path.relative_to(ROOT).as_posix()
        if name in files:
            return
        raw = path.read_bytes()
        files[name] = raw
        if path.suffix not in (".js", ".mjs"):
            return
        source = raw.decode("utf-8")
        references = re.findall(
            r'(?:from\s*|import\s*\(|new URL\s*\()\s*[\'"]([^\'"]+)[\'"]', source
        )
        for ref in references:
            if ref.startswith("."):
                relative = (Path(name).parent / ref.split("?")[0]).as_posix()
                if (ROOT / relative).is_file():
                    code(relative)

    code(ENTRY)
    documents = {}

    def asset(name):
        if name in files:
            return
        raw = (ROOT / name).read_bytes()
        files[name] = raw
        if name.endswith(".json"):
            document = json.loads(raw)
            documents[name] = document

            def walk(value):
                if isinstance(value, dict):
                    for item in value.values():
                        walk(item)
                elif isinstance(value, list):
                    for item in value:
                        walk(item)
                elif (
                    isinstance(value, str)
                    and value.startswith("output/")
                    and (ROOT / value).is_file()
                ):
                    asset(value)

            walk(document)

    for name in ["model.json", "surface-embedding.json", "surface-embedding.bin"]:
        asset(base + name)
    model = documents[base + "model.json"]
    skin = model["skin_url"]
    asset(skin)
    if model.get("skin_glb_sha256") != digest(files[skin]):
        raise ValueError(
            "Source skin changed after binding; rebuild volume and bindings"
        )
    model["capabilities"] = {
        "volumetric_fem": True,
        "compression_barrier": "numerical orientation barrier, active below J=.6 and singular at the J=.2 guard; not a calibrated tissue term",
        "dermal_bending": True,
        "adipose_growth": True,
        "fine_wrinkles": "one-way forehead patch",
        "anatomical_registration": False,
        "bone_contact": False,
        "muscle_volume_fem": False,
        "self_contact": False,
        "jaw_dynamics": False,
        "subsurface_scattering": "depth-gated screen-space diffuse transport; no multilayer spectral or back-face transport",
    }
    if model.get("registration_mode"):
        model["capabilities"]["anatomical_registration"] = (
            "shared native coordinates; vertex/chart containment checked, not anatomical calibration"
        )
        model["capabilities"]["local_bone_contact"] = (
            "fixed nearest bone/cartilage triangle tangents; no global or moving contact"
        )
    # Bind the added GLB identity into the build hash as well as all source bytes.
    files[base + "model.json"] = json.dumps(model, sort_keys=True).encode("utf-8")
    identity = digest(
        b"".join(name.encode() + b"\0" + files[name] for name in sorted(files))
    )[:24]
    target = ROOT / "output/runtime" / identity
    prefix = "/output/runtime/" + identity + "/assets/"

    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str) and value in files and value.startswith("output/"):
            return prefix + value
        return value

    if not target.exists():
        temporary = target.with_name(identity + ".pending")
        temporary.mkdir(parents=True, exist_ok=True)
        blob_directory = ROOT / "output/runtime-blobs"
        blob_directory.mkdir(parents=True, exist_ok=True)
        for name, raw in files.items():
            destination = temporary / (
                "assets/" + name if name.startswith("output/") else name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name in documents:
                raw = json.dumps(rewrite(documents[name]), indent=2).encode("utf-8")
            # Snapshots share immutable blobs, never mutable source-file links.
            # A rebuild writes a new digest rather than modifying an old blob.
            blob = blob_directory / digest(raw)
            if not blob.exists():
                blob_pending = blob.with_suffix(".pending")
                blob_pending.write_bytes(raw)
                os.replace(blob_pending, blob)
            elif digest(blob.read_bytes()) != blob.name:
                raise ValueError("Corrupted immutable runtime blob: " + blob.name)
            if destination.exists():
                if digest(destination.read_bytes()) != blob.name:
                    raise ValueError("Incomplete snapshot has inconsistent contents")
            else:
                materialize_blob(blob, destination)
        manifest = {
            "build": identity,
            "source_sha256": {name: digest(raw) for name, raw in sorted(files.items())},
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        temporary.rename(target)
    pointer = {
        "build": identity,
        "entry": "/output/runtime/" + identity + "/" + ENTRY,
        "tissueBase": prefix + base,
    }
    current = ROOT / (
        "output/runtime-current.json" if publish else "output/runtime-candidate.json"
    )
    pending = current.with_suffix(".pending")
    pending.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    os.replace(pending, current)
    print(
        json.dumps(
            {
                "build": identity,
                "files": len(files),
                "source_bytes": sum(map(len, files.values())),
                "pointer": str(current),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    main(args.base, not args.no_publish)
