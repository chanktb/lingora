"""Run the HyperFrames CLI to render a project directory to MP4."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class RenderError(RuntimeError):
    pass


def _ensure_ffmpeg_on_path() -> None:
    base = os.environ.get("FFMPEG_BIN", "")
    if base and base not in os.environ.get("PATH", ""):
        os.environ["PATH"] = base + os.pathsep + os.environ.get("PATH", "")


def render(
    project_dir: Path,
    output_mp4: Path,
    *,
    hyperframes_version: str = "0.6.52",
    timeout: int = 600,
) -> Path:
    """Render project_dir/index.html to output_mp4.

    Raises RenderError on non-zero exit or missing output file.
    """
    _ensure_ffmpeg_on_path()
    project_dir = project_dir.resolve()
    output_mp4 = output_mp4.resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        raise RenderError("npx not found in PATH — install Node.js >= 22")

    cmd = [
        npx,
        "--yes",
        f"hyperframes@{hyperframes_version}",
        "render",
        "--output",
        str(output_mp4),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"Render timed out after {timeout}s") from exc

    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-30:])
        raise RenderError(
            f"hyperframes render exited {result.returncode}\n--- last log ---\n{tail}",
        )
    if not output_mp4.exists():
        raise RenderError(f"render finished but {output_mp4} is missing")

    return output_mp4


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python renderer.py <project_dir> <output_mp4>")
        sys.exit(1)
    out = render(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Rendered: {out}")
