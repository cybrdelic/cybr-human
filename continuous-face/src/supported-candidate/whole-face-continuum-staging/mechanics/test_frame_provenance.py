"""Serialization-only regression; no actual geometry or motion."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from first_step import save_accepted_frames


class FrameProvenanceTest(unittest.TestCase):
    def test_saved_bytes_are_hashed_after_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"synthetic-serialization-test.npz"
            solid = SimpleNamespace(X=np.zeros((1, 3)), mm_per_L0=1.)
            result = save_accepted_frames(path, positions=[[[0., 0., 0.]]],
                velocities=[[[0., 0., 0.]]], times=[.001], solid=solid)
            self.assertEqual(result["accepted_npz_path"], str(path.resolve()))
            self.assertEqual(result["accepted_npz_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
