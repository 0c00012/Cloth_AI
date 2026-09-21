# -*- coding: utf-8 -*-
"""
Step 2 - Physically-constrained swatch extraction by grid-origin optimization
(paper 3.4.2 - 3.4.5).

Method
------
1. Physical scale: 3,755 px == 100 cm  ->  a 10 x 10 cm swatch is 376 x 376 px.
2. Regular grid: cell size and cell pitch are both 376 px, so cells of one grid
   never overlap (non-overlap constraint, 3.4.4).
3. Garment constraint (3.4.3): a cell is valid only if every pixel lies inside the
   binary garment mask (alpha > 10). Validity of all top-left positions is computed
   once with an integral image.
4. Grid-origin optimization (3.4.5): the whole grid is translated by
   (ox, oy) in [0, 375] x [0, 375]  ->  141,376 origins. For each origin the number
   of valid cells is counted; the origin with the largest count is selected.
   Ties are broken by the smaller y, then the smaller x.
5. All valid cells of the selected grid are saved in row-major order.

This is an exhaustive search over the translated regular-grid family. It is not a
global optimum over all free (non-grid) placements, and the code makes no such claim.

Inputs : work/01_segmented/*_nobg.png   (RGBA from step 1)
Outputs: work/02_swatches/<garment>/<garment>_crop_NN.png
         work/02_swatches/<garment>_placement.json     (origin, coordinates, counts)
         work/02_swatches/<garment>_grid_origin_counts.npy (376 x 376 count map)
         work/02_swatches/<garment>_layout.png          (real-image overlay figure)
         work/02_swatches/extraction_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline_config import ALPHA_THRESHOLD, SEGMENT_DIR, SWATCH_DIR, SWATCH_PX, garment_id


# --------------------------------------------------------------------------- core
def garment_mask(rgba: np.ndarray) -> np.ndarray:
    """Binary garment mask from the RGBA alpha channel (alpha > threshold)."""
    return rgba[:, :, 3] > ALPHA_THRESHOLD


def valid_top_left(mask: np.ndarray, size: int) -> np.ndarray:
    """valid[y, x] is True when the size x size square at (x, y) is fully inside the mask.

    Implemented with an integral image so each square costs O(1).
    """
    h, w = mask.shape
    if h < size or w < size:
        return np.zeros((max(h - size + 1, 0), max(w - size + 1, 0)), dtype=bool)
    ii = np.zeros((h + 1, w + 1), dtype=np.int64)
    ii[1:, 1:] = mask.cumsum(0).cumsum(1)
    sums = ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]
    return sums == size * size


def grid_origin_counts(valid: np.ndarray, size: int) -> np.ndarray:
    """counts[oy, ox] = number of valid cells of the regular grid with origin (ox, oy).

    Valid top-left positions are grouped by their residue modulo the pitch, which
    is exactly the set of cells belonging to that grid origin.
    """
    h, w = valid.shape
    hp, wp = -(-h // size) * size, -(-w // size) * size
    padded = np.zeros((hp, wp), dtype=bool)
    padded[:h, :w] = valid
    return padded.reshape(hp // size, size, wp // size, size).sum(axis=(0, 2), dtype=np.int64)


def grid_cells(valid: np.ndarray, size: int, origin: tuple[int, int]) -> list[tuple[int, int]]:
    """Top-left (x, y) of every valid cell of the grid anchored at origin, row-major."""
    ox, oy = origin
    rc = np.argwhere(valid[oy::size, ox::size])             # (row, col), row-major
    return [(int(c * size + ox), int(r * size + oy)) for r, c in rc]


def optimise_grid_origin(valid: np.ndarray, size: int):
    counts = grid_origin_counts(valid, size)
    oy, ox = np.unravel_index(int(np.argmax(counts)), counts.shape)   # first max: min y, then min x
    origin = (int(ox), int(oy))
    return grid_cells(valid, size, origin), origin, counts


def verify(mask: np.ndarray, cells: list[tuple[int, int]], size: int) -> None:
    """Independent pixel-level check: containment, bounds, non-overlap, area identity."""
    h, w = mask.shape
    occupied = np.zeros_like(mask)
    for x, y in cells:
        assert 0 <= x <= w - size and 0 <= y <= h - size, (x, y, "out of bounds")
        assert mask[y:y + size, x:x + size].all(), (x, y, "cell leaves the garment mask")
        assert not occupied[y:y + size, x:x + size].any(), (x, y, "overlap")
        occupied[y:y + size, x:x + size] = True
    assert int(occupied.sum()) == len(cells) * size * size


# ------------------------------------------------------------------------ outputs
def draw_layout(rgba: np.ndarray, cells, size, origin, label, path: Path) -> None:
    im = Image.new("RGBA", (rgba.shape[1], rgba.shape[0]), "white")
    im.alpha_composite(Image.fromarray(rgba))
    im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    for x, y in cells:
        d.rectangle((x, y, x + size - 1, y + size - 1), outline="#B54217", width=6)
    d.text((20, 20), f"{label}: {len(cells)} swatches, grid origin (x, y) = {origin} px", fill="black")
    im.save(path)


def extract_one(nobg_path: Path, out_dir: Path, size: int) -> dict:
    stem = nobg_path.stem.replace("_nobg", "")
    label = garment_id(stem)
    with Image.open(nobg_path) as im:
        rgba = np.array(im.convert("RGBA"))
    mask = garment_mask(rgba)
    valid = valid_top_left(mask, size)
    cells, origin, counts = optimise_grid_origin(valid, size)
    verify(mask, cells, size)

    swatch_dir = out_dir / stem
    swatch_dir.mkdir(parents=True, exist_ok=True)
    for i, (x, y) in enumerate(cells, 1):
        Image.fromarray(rgba[y:y + size, x:x + size]).save(swatch_dir / f"{stem}_crop_{i:02d}.png")

    np.save(out_dir / f"{stem}_grid_origin_counts.npy", counts)
    draw_layout(rgba, cells, size, origin, label, out_dir / f"{stem}_layout.png")

    fg = int(mask.sum())
    record = {
        "garment": label, "stem": stem, "source": nobg_path.name,
        "image_width": int(mask.shape[1]), "image_height": int(mask.shape[0]),
        "swatch_px": size, "alpha_threshold": ALPHA_THRESHOLD,
        "foreground_px": fg, "valid_top_left_positions": int(valid.sum()),
        "grid_origins_tested": int(counts.size),
        "grid_origin_xy": list(origin), "count": len(cells),
        "fixed_origin_count": int(counts[0, 0]),
        "coverage_pct_of_mask": 100.0 * len(cells) * size * size / fg,
        "cells_xy_row_major": cells,
    }
    (out_dir / f"{stem}_placement.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=SEGMENT_DIR, help="folder with *_nobg.png")
    ap.add_argument("--output", type=Path, default=SWATCH_DIR)
    ap.add_argument("--size", type=int, default=SWATCH_PX, help="swatch side in px (default 376 = 10 cm)")
    args = ap.parse_args()

    files = sorted(args.input.glob("*_nobg.png"))
    if not files:
        raise SystemExit(f"no *_nobg.png in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        r = extract_one(f, args.output, args.size)
        rows.append(r)
        print(f"{r['garment']} {r['stem']}: {r['count']} swatches at origin {tuple(r['grid_origin_xy'])} "
              f"(fixed origin (0,0): {r['fixed_origin_count']}), mask coverage {r['coverage_pct_of_mask']:.2f}%")
    keys = ["garment", "stem", "image_width", "image_height", "foreground_px", "grid_origins_tested",
            "grid_origin_xy", "count", "fixed_origin_count", "coverage_pct_of_mask"]
    with (args.output / "extraction_summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"({r[k][0]}, {r[k][1]})" if k == "grid_origin_xy" else r[k]) for k in keys})
    print(f"total {sum(r['count'] for r in rows)} swatches -> {args.output}")


if __name__ == "__main__":
    main()
