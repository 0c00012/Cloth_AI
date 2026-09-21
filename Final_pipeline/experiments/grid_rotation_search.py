# -*- coding: utf-8 -*-
"""
Pilot experiment: grid ORIENTATION + origin optimization.

The paper's method only translates an axis-aligned regular grid. Here the garment
mask is additionally rotated by theta in [0, 90) deg (a square grid repeats every
90 deg) and the same exhaustive origin search is run on the rotated mask. The best
(theta, origin) is reported per garment together with the axis-aligned result.

Rotation uses nearest-neighbour resampling of the binary mask with the canvas
enlarged so nothing is clipped; counts are therefore exact for the rotated mask,
but swatches would have to be cut on the rotated image (fabric grain rotates too),
which the current pipeline does not do. This script only measures the potential gain.

usage: python experiments/grid_rotation_search.py [--step 5] [--output experiments/results_rotation]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib  # noqa: E402

ex = importlib.import_module("2_extract_swatches")
from pipeline_config import ALPHA_THRESHOLD, SEGMENT_DIR, SWATCH_PX, garment_id  # noqa: E402


def rotated_mask(mask: np.ndarray, theta: float) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(im.rotate(theta, resample=Image.Resampling.NEAREST, expand=True)) > 127


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=SEGMENT_DIR)
    ap.add_argument("--step", type=float, default=5.0, help="angle step in degrees")
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results_rotation")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    angles = np.arange(0.0, 90.0, args.step)
    rows, summary = [], []
    for f in sorted(args.input.glob("*_nobg.png")):
        stem = f.stem.replace("_nobg", "")
        g = garment_id(stem)
        with Image.open(f) as im:
            mask = np.array(im.convert("RGBA"))[:, :, 3] > ALPHA_THRESHOLD
        best = None
        for th in angles:
            m = rotated_mask(mask, float(th))
            valid = ex.valid_top_left(m, SWATCH_PX)
            counts = ex.grid_origin_counts(valid, SWATCH_PX)
            n = int(counts.max())
            oy, ox = np.unravel_index(int(np.argmax(counts)), counts.shape)
            rows.append({"garment": g, "theta_deg": float(th), "count": n, "origin_x": int(ox), "origin_y": int(oy)})
            if best is None or n > best[1]:
                best = (float(th), n)
            print(f"{g} theta={th:5.1f} count={n}", flush=True)
        axis = next(r["count"] for r in rows if r["garment"] == g and r["theta_deg"] == 0.0)
        summary.append({"garment": g, "axis_aligned": axis, "best_theta_deg": best[0], "best_count": best[1],
                        "gain": best[1] - axis})
    for name, data in (("rotation_runs.csv", rows), ("rotation_summary.csv", summary)):
        with (args.output / name).open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print("summary:")
    for s in summary:
        print(s)
    print("total axis-aligned", sum(s["axis_aligned"] for s in summary), "best rotated", sum(s["best_count"] for s in summary))


if __name__ == "__main__":
    main()
