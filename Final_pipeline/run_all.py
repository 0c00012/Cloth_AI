# -*- coding: utf-8 -*-
"""Run the whole paper pipeline in order (segment -> extract -> synthesize -> render).

usage: python run_all.py [--skip-segment] [--skip-render]
  --skip-segment  reuse the *_nobg.png already in work/01_segmented (paper inputs)
  --skip-render   stop after the master textures (no Blender needed)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["1_segment_garments.py", "2_extract_swatches.py", "3_synthesize_master_texture.py", "4_render_weave.py"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-segment", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()
    steps = list(STEPS)
    if args.skip_segment:
        steps.remove("1_segment_garments.py")
    if args.skip_render:
        steps.remove("4_render_weave.py")
    for s in steps:
        print(f"\n===== {s} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE)
        if r.returncode:
            print(f"step failed: {s} (exit {r.returncode})")
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
