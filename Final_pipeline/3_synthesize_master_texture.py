# -*- coding: utf-8 -*-
"""
Step 3 - Texture synthesis: master texture from equal-size swatches (paper 3.5).

* 3.5.1  every swatch is 376 x 376 px (nominal 10 x 10 cm); no resizing is applied.
* 3.5.2  the swatch order is randomised with a fixed seed (42) so that the result
         is reproducible; no new texture content is generated.
* 3.5.3  floor-based rectangular grid:  cols = floor(sqrt(N)),  rows = N // cols,
         N_used = cols * rows; the remaining swatches are excluded.
* 3.5.4  swatches are pasted in row-major order without gaps.

Inputs : work/02_swatches/<garment>/<garment>_crop_NN.png   (from step 2)
Outputs: work/03_master_textures/<garment>_master_texture_<cols>x<rows>.png
         work/03_master_textures/<garment>_composition.json
         work/03_master_textures/texture_composition_summary.csv  (Table 1 style)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from PIL import Image

from pipeline_config import SHUFFLE_SEED, SWATCH_DIR, SWATCH_PX, TEXTURE_DIR, garment_id

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def floor_grid(n: int) -> tuple[int, int]:
    """cols = floor(sqrt(N)), rows = N // cols (paper 3.5.3)."""
    cols = max(math.isqrt(n), 1)
    return cols, n // cols


def synthesize(swatch_folder: Path, out_dir: Path, seed: int | None, size: int) -> dict | None:
    files = sorted(p for p in swatch_folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    n = len(files)
    if n == 0:
        return None
    cols, rows = floor_grid(n)
    used = cols * rows
    order = list(files)
    rng = random.Random(seed) if seed is not None else random.Random()
    rng.shuffle(order)
    selected = order[:used]

    canvas = Image.new("RGBA", (cols * size, rows * size))
    for idx, path in enumerate(selected):
        r, c = divmod(idx, cols)
        patch = Image.open(path).convert("RGBA")
        assert patch.size == (size, size), f"{path.name} is {patch.size}, expected {(size, size)}"
        canvas.paste(patch, (c * size, r * size))

    stem = swatch_folder.name
    out_path = out_dir / f"{stem}_master_texture_{cols}x{rows}.png"
    canvas.save(out_path)
    record = {
        "garment": garment_id(stem), "stem": stem, "extracted": n, "used": used, "excluded": n - used,
        "cols": cols, "rows": rows, "grid": f"{cols}x{rows}",
        "physical_size_cm": f"{cols * 10} x {rows * 10}",
        "utilization_pct": round(100.0 * used / n, 1), "seed": seed,
        "texture_px": [cols * size, rows * size], "output": out_path.name,
        "order_used": [p.name for p in selected], "excluded_files": [p.name for p in order[used:]],
    }
    (out_dir / f"{stem}_composition.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=SWATCH_DIR, help="folder containing one sub-folder per garment")
    ap.add_argument("--output", type=Path, default=TEXTURE_DIR)
    ap.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    ap.add_argument("--size", type=int, default=SWATCH_PX)
    args = ap.parse_args()

    folders = sorted(p for p in args.input.iterdir() if p.is_dir())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in folders:
        r = synthesize(folder, args.output, args.seed, args.size)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['garment']} {r['stem']}: {r['extracted']} extracted -> {r['grid']} grid, "
              f"{r['used']} used ({r['utilization_pct']}%), {r['physical_size_cm']} cm")
    keys = ["garment", "stem", "extracted", "used", "excluded", "grid", "physical_size_cm", "utilization_pct"]
    with (args.output / "texture_composition_summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
        tot_e, tot_u = sum(r["extracted"] for r in rows), sum(r["used"] for r in rows)
        w.writerow({"garment": "Total", "extracted": tot_e, "used": tot_u, "excluded": tot_e - tot_u,
                    "utilization_pct": round(100.0 * tot_u / tot_e, 1) if tot_e else 0})
    print(f"total used {sum(r['used'] for r in rows)} / {sum(r['extracted'] for r in rows)} -> {args.output}")


if __name__ == "__main__":
    main()
