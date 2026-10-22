"""Synthetic two-frame encode/decode check; never represents a facial animation."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "src/supported-candidate/whole-face-continuum-staging"
RENDERER = BASE / "visualization/portable-renderer-0006"
sys.path[:0] = [str(RENDERER), str(BASE / "mechanics")]
from platform_runtime import resolve_tool
from render_actual import verify_encoded_movie

def main():
    ffmpeg = resolve_tool("ffmpeg")
    resolve_tool("ffprobe")
    with tempfile.TemporaryDirectory(prefix="cybr synthetic media ") as folder:
        root = Path(folder)
        for i, color in enumerate(((220, 20, 50), (20, 70, 220))):
            Image.new("RGB", (16, 16), color).save(root / f"frame-{i:02d}.png")
        output = root / "synthetic colors.mp4"
        command = [ffmpeg, "-nostdin", "-v", "error", "-framerate", "2",
                   "-i", str(root / "frame-%02d.png"), "-frames:v", "2",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
        subprocess.run(command, check=True)
        verified = verify_encoded_movie(output, 2, 2, (16, 16))
        print(json.dumps({"fixture": "synthetic colors, not anatomy or motion",
                          "verified": verified["decoded_frame_count_verified"],
                          "stream": verified["stream"]}, indent=2))

if __name__ == "__main__":
    main()
