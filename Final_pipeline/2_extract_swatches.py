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
5. Tie-break (many origins reach the same maximum): among the maximum-count origins
   the one whose worst cell keeps the largest margin to the mask boundary is chosen
   (max-min cell margin, then max mean margin, then smallest y, then smallest x).
   Cells therefore sit as far as possible from hems, necklines and sleeve edges.
6. All valid cells of the selected grid are saved in row-major order and the
   placement is verified pixel-wise. A maximality check confirms that no further
   376 x 376 px square fits in the remaining garment area.

This is an exhaustive search over the translated regular-grid family. It is not a
global optimum over all free (non-grid) placements, and the code makes no such claim.

Inputs : work/01_segmented/*_nobg.png   (RGBA from step 1)
Outputs: work/02_swatches/<garment>/<garment>_crop_NN.png
         work/02_swatches/<garment>_placement.json     (origin, coordinates, counts, bounds)
         work/02_swatches/<garment>_grid_origin_counts.npy (376 x 376 count map)
         work/02_swatches/<garment>_layout.png          (real-image overlay figure)
         work/02_swatches/extraction_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, minimum_filter

from pipeline_config import ALPHA_THRESHOLD, SEGMENT_DIR, SWATCH_DIR, SWATCH_PX, garment_id


# --------------------------------------------------------------------------- core
def garment_mask(rgba: np.ndarray, threshold: int = ALPHA_THRESHOLD) -> np.ndarray:
    """Binary garment mask from the RGBA alpha channel (alpha > threshold)."""
    return rgba[:, :, 3] > threshold


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


def cell_margin_map(mask: np.ndarray, size: int) -> np.ndarray:
    """margin[y, x] = smallest distance (px) from any pixel of the square at (x, y) to the
    mask boundary, i.e. the minimum of the Euclidean distance transform over the square."""
    dt = distance_transform_edt(mask)
    m = minimum_filter(dt, size=size, mode="constant", cval=0.0, origin=-(size // 2))
    return m[: mask.shape[0] - size + 1, : mask.shape[1] - size + 1]


def optimise_grid_origin(valid: np.ndarray, size: int, margin: np.ndarray | None = None):
    """Return (cells, origin, counts, tie_info).

    Primary criterion: maximum number of valid cells.
    Secondary (when `margin` is given): maximise the minimum cell margin, then the mean
    margin; remaining ties are broken by the smaller y, then the smaller x.
    """
    counts = grid_origin_counts(valid, size)
    best = int(counts.max())
    ties = np.argwhere(counts == best)                      # (oy, ox), row-major = y then x
    info = {"max_count": best, "tied_origins": int(len(ties))}
    if margin is None or len(ties) == 1:
        oy, ox = ties[0]
        origin = (int(ox), int(oy))
        return grid_cells(valid, size, origin), origin, counts, info
    scored = []
    for oy, ox in ties:
        cells = grid_cells(valid, size, (int(ox), int(oy)))
        m = np.array([margin[y, x] for x, y in cells])
        scored.append((float(m.min()), float(m.mean()), -int(oy), -int(ox), (int(ox), int(oy))))
    scored.sort(reverse=True)
    origin = scored[0][4]
    info.update({"min_cell_margin_px": scored[0][0], "mean_cell_margin_px": scored[0][1],
                 "first_tie_origin_xy": [int(ties[0][1]), int(ties[0][0])],
                 "first_tie_min_margin_px": float(min(margin[y, x] for x, y in grid_cells(valid, size, (int(ties[0][1]), int(ties[0][0])))))})
    return grid_cells(valid, size, origin), origin, counts, info


def verify(mask: np.ndarray, cells: list[tuple[int, int]], size: int) -> np.ndarray:
    """Independent pixel-level check: containment, bounds, non-overlap, area identity."""
    h, w = mask.shape
    occupied = np.zeros_like(mask)
    for x, y in cells:
        assert 0 <= x <= w - size and 0 <= y <= h - size, (x, y, "out of bounds")
        assert mask[y:y + size, x:x + size].all(), (x, y, "cell leaves the garment mask")
        assert not occupied[y:y + size, x:x + size].any(), (x, y, "overlap")
        occupied[y:y + size, x:x + size] = True
    assert int(occupied.sum()) == len(cells) * size * size
    return occupied


def remaining_capacity(mask: np.ndarray, occupied: np.ndarray, size: int) -> int:
    """Number of top-left positions where another square would still fit (0 = maximal)."""
    return int(valid_top_left(mask & ~occupied, size).sum())


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


def extract_one(nobg_path: Path, out_dir: Path, size: int, threshold: int, tie_break: bool) -> dict:
    stem = nobg_path.stem.replace("_nobg", "")
    label = garment_id(stem)
    with Image.open(nobg_path) as im:
        rgba = np.array(im.convert("RGBA"))
    mask = garment_mask(rgba, threshold)
    valid = valid_top_left(mask, size)
    margin = cell_margin_map(mask, size) if tie_break else None
    cells, origin, counts, tie = optimise_grid_origin(valid, size, margin)
    occupied = verify(mask, cells, size)
    leftover = remaining_capacity(mask, occupied, size)

    swatch_dir = out_dir / stem
    swatch_dir.mkdir(parents=True, exist_ok=True)
    for old in swatch_dir.glob(f"{stem}_crop_*.png"):
        old.unlink()
    for i, (x, y) in enumerate(cells, 1):
        Image.fromarray(rgba[y:y + size, x:x + size]).save(swatch_dir / f"{stem}_crop_{i:02d}.png")

    np.save(out_dir / f"{stem}_grid_origin_counts.npy", counts)
    draw_layout(rgba, cells, size, origin, label, out_dir / f"{stem}_layout.png")

    fg = int(mask.sum())
    record = {
        "garment": label, "stem": stem, "source": nobg_path.name,
        "source_sha256": hashlib.sha256(nobg_path.read_bytes()).hexdigest(),
        "image_width": int(mask.shape[1]), "image_height": int(mask.shape[0]),
        "swatch_px": size, "alpha_threshold": threshold,
        "foreground_px": fg, "valid_top_left_positions": int(valid.sum()),
        "grid_origins_tested": int(counts.size),
        "grid_origin_xy": list(origin), "count": len(cells),
        "fixed_origin_count": int(counts[0, 0]),
        "origin_count_mean": float(counts.mean()), "origin_count_min": int(counts.min()),
        "area_upper_bound": fg // (size * size),
        "remaining_positions_after_grid": leftover,
        "coverage_pct_of_mask": 100.0 * len(cells) * size * size / fg,
        "tie_break": tie, "cells_xy_row_major": cells,
    }
    (out_dir / f"{stem}_placement.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=SEGMENT_DIR, help="folder with *_nobg.png")
    ap.add_argument("--output", type=Path, default=SWATCH_DIR)
    ap.add_argument("--size", type=int, default=SWATCH_PX, help="swatch side in px (default 376 = 10 cm)")
    ap.add_argument("--threshold", type=int, default=ALPHA_THRESHOLD, help="alpha > threshold is garment")
    ap.add_argument("--no-tie-break", action="store_true", help="use plain first-maximum (smallest y, then x) selection")
    args = ap.parse_args()

    files = sorted(args.input.glob("*_nobg.png"))
    if not files:
        raise SystemExit(f"no *_nobg.png in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        r = extract_one(f, args.output, args.size, args.threshold, not args.no_tie_break)
        rows.append(r)
        t = r["tie_break"]
        print(f"{r['garment']} {r['stem']}: {r['count']} swatches at origin {tuple(r['grid_origin_xy'])} "
              f"(bound {r['area_upper_bound']}, ties {t['tied_origins']}, min margin "
              f"{t.get('min_cell_margin_px', float('nan')):.0f} px, leftover positions {r['remaining_positions_after_grid']})")
    keys = ["garment", "stem", "image_width", "image_height", "foreground_px", "alpha_threshold", "grid_origins_tested",
            "grid_origin_xy", "count", "fixed_origin_count", "origin_count_mean", "origin_count_min",
            "area_upper_bound", "remaining_positions_after_grid", "coverage_pct_of_mask"]
    with (args.output / "extraction_summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["tied_origins", "min_cell_margin_px"])
        w.writeheader()
        for r in rows:
            row = {k: (f"({r[k][0]}, {r[k][1]})" if k == "grid_origin_xy" else r[k]) for k in keys}
            row["tied_origins"] = r["tie_break"]["tied_origins"]
            row["min_cell_margin_px"] = r["tie_break"].get("min_cell_margin_px", "")
            w.writerow(row)
    print(f"total {sum(r['count'] for r in rows)} swatches -> {args.output}")


if __name__ == "__main__":
    main()
