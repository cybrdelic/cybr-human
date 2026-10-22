"""Real build is exercised by native_render_smoke; these cover platform contracts."""
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/supported-candidate/whole-face-continuum-staging/visualization/portable-renderer-0006"
spec = importlib.util.spec_from_file_location("platform_runtime", SOURCE / "platform_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

class PlatformRuntimeTests(unittest.TestCase):
    def test_frozen_and_derived_provenance(self):
        provenance = json.loads((SOURCE / "PROVENANCE.json").read_text(encoding="utf-8"))
        original = SOURCE.parent / "fixed-barrier-renderer-source-snapshot-0006"
        for item in provenance["files"]:
            with self.subTest(file=item["file"]):
                self.assertEqual(hashlib.sha256((original / item["file"]).read_bytes()).hexdigest(), item["original_sha256"])
                self.assertEqual(hashlib.sha256((SOURCE / item["file"]).read_bytes()).hexdigest(), item["derived_sha256"])
        self.assertEqual((SOURCE / "rasterizer.cpp").read_bytes(), (original / "rasterizer.cpp").read_bytes())

    def test_library_suffixes(self):
        for name, suffix in [("Windows", ".dll"), ("Linux", ".so"), ("Darwin", ".dylib")]:
            with self.subTest(name=name), patch.object(runtime.platform, "system", return_value=name):
                self.assertEqual(runtime.library_suffix(), suffix)

    def test_unknown_platform_is_explicit(self):
        with patch.object(runtime.platform, "system", return_value="Other"):
            with self.assertRaisesRegex(RuntimeError, "not supported"):
                runtime.library_suffix()

    def test_missing_tool_actionable(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(runtime.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "CYBR_FFMPEG"):
                runtime.resolve_tool("ffmpeg")

    def test_invalid_override_does_not_silently_use_path_tool(self):
        with patch.dict(os.environ, {"CYBR_FFMPEG": "missing tool"}), patch.object(runtime.shutil, "which", return_value=None) as which:
            with self.assertRaisesRegex(RuntimeError, "does not name"):
                runtime.resolve_tool("ffmpeg")
            which.assert_called_once_with("missing tool")

    def test_override_with_spaces_stays_one_executable(self):
        with patch.dict(os.environ, {"CYBR_FFMPEG": "path with spaces/ffmpeg"}), patch.object(runtime.shutil, "which", return_value=__file__) as which:
            self.assertEqual(runtime.resolve_tool("ffmpeg"), str(Path(__file__).resolve()))
            which.assert_called_once_with("path with spaces/ffmpeg")

    def test_lock_is_exclusive_and_removed_after_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / ".lock"
            with self.assertRaisesRegex(ValueError, "test"):
                with runtime.build_lock(lock):
                    self.assertTrue(lock.exists())
                    with self.assertRaisesRegex(RuntimeError, "locked"):
                        with runtime.build_lock(lock, timeout=0):
                            pass
                    raise ValueError("test")
            self.assertFalse(lock.exists())

if __name__ == "__main__":
    unittest.main()
