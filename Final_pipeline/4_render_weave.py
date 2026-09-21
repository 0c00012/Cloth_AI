# -*- coding: utf-8 -*-
"""
Step 4 - Yarn-based 3D weaving and rendering (paper 3.6).

Runs Blender in background mode with Blender_Large_Weave.py for every master
texture produced by step 3. The grid size (cols x rows) is read from the file name
(<garment>_master_texture_<cols>x<rows>.png) and defines the physical fabric size
(cols*10 cm x rows*10 cm), the yarn counts and the camera/light settings inside the
Blender script.

Blender is located through (in order): the BLENDER_EXE environment variable,
`blender` on PATH, or C:\\Program Files\\Blender Foundation\\Blender *\\blender.exe.

Inputs : work/03_master_textures/*_master_texture_CxR.png
Outputs: outputs/04_weave_renders/<garment>/<garment>_CxR.png  (PNG, RGBA, transparent film)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline_config import BLENDER_SCRIPT, RENDER_DIR, TEXTURE_DIR

NAME_RE = re.compile(r"^(?P<stem>.+)_master_texture_(?P<cols>\d+)x(?P<rows>\d+)\.png$", re.IGNORECASE)


def find_blender() -> str | None:
    env = os.environ.get("BLENDER_EXE")
    if env and Path(env).exists():
        return env
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    root = Path(r"C:\Program Files\Blender Foundation")
    if root.exists():
        for d in sorted(root.iterdir(), reverse=True):
            exe = d / "blender.exe"
            if exe.exists():
                return str(exe)
    return None


def render(texture: Path, cols: int, rows: int, out_dir: Path, blender: str, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{cols}x{rows}.png"
    cmd = [blender, "-b", "-P", str(BLENDER_SCRIPT), "--", str(texture), str(cols), str(rows), str(out_path)]
    print("  ", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=TEXTURE_DIR)
    ap.add_argument("--output", type=Path, default=RENDER_DIR)
    ap.add_argument("--only", nargs="*", default=None, help="garment stems to render (default: all)")
    args = ap.parse_args()

    blender = find_blender()
    if not blender:
        print("Blender executable not found. Set BLENDER_EXE or add blender to PATH.")
        return 1
    print(f"Blender: {blender}")
    textures = sorted(args.input.glob("*_master_texture_*x*.png"))
    if not textures:
        print(f"no master textures in {args.input}")
        return 1
    for tex in textures:
        m = NAME_RE.match(tex.name)
        if not m:
            continue
        stem = m.group("stem")
        if args.only and stem not in args.only:
            continue
        cols, rows = int(m.group("cols")), int(m.group("rows"))
        print(f"{stem}: {cols}x{rows} grid ({cols * 10} x {rows * 10} cm)")
        out = render(tex, cols, rows, args.output / stem, blender, stem)
        print(f"   -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
