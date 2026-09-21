# -*- coding: utf-8 -*-
"""
Placement-strategy comparison (paper 3.4.6 / 4.4, Table 2, Figures 9-10).

All strategies share the same binary garment mask (alpha > 10), the same
376 x 376 px swatch, full containment, non-overlap and no rotation.

  greedy_10px     raster-scan greedy baseline: origin (0, 0), stride 10 px,
                  row-major scan, accept immediately (legacy/greedy_raster_packing.py)
  grid_fixed      regular 376 px grid anchored at (0, 0), all fully-contained cells
  grid_best_phase grid-origin optimization: all 376 x 376 = 141,376 integer
                  translations, pick the largest count (ties: smaller y, then x)
                  == 2_extract_swatches.py (the paper's method)
  random_1px      random sequential placement over all feasible integer positions,
                  run until no feasible position remains; seeds 0..N-1 (NumPy PCG64)
  random_10px     same as random_1px restricted to the greedy 10 px candidate lattice
  greedy_1px      raster greedy with 1 px stride (density control)

Every placement is verified pixel-wise (containment, bounds, non-overlap, area).
Outputs (never overwrites a non-empty folder): runs.csv, summary.csv, totals.csv,
placements.jsonl, <G>_all_grid_phase_counts.npy, manifest.json, validation.json.

This file is a path-cleaned copy of the experiment script that produced the
numbers in the paper (experiments/packing_comparison_20260903/run_experiment.py
in the analysis workspace); the algorithms are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import ALPHA_THRESHOLD, GARMENT_IDS, ROOT, SEGMENT_DIR, SWATCH_PX  # noqa: E402

HERE = Path(__file__).resolve().parent
SIZE = SWATCH_PX
METHODS = ["greedy_10px", "grid_fixed", "grid_best_phase", "random_1px", "random_10px", "greedy_1px"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path, rows):
    if rows:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


# ----------------------------------------------------------------- algorithms
def valid_positions(mask, size):
    h, w = mask.shape
    if min(h, w) < size:
        return np.zeros((max(0, h - size + 1), max(0, w - size + 1)), dtype=bool)
    ii = np.zeros((h + 1, w + 1), dtype=np.int32)
    ii[1:, 1:] = mask.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    sums = ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]
    return sums == size * size


def sequential(valid, xs, ys, size, seed=None):
    """Raster-first (seed None) or uniformly random sequential placement on a candidate lattice."""
    live = valid[np.ix_(ys, xs)].copy()
    counts = live.sum(axis=1, dtype=np.int64)
    total = int(counts.sum())
    rng = np.random.Generator(np.random.PCG64(seed)) if seed is not None else None
    points = []
    while total:
        if rng is None:
            iy = int(np.flatnonzero(counts)[0])
            ix = int(np.flatnonzero(live[iy])[0])
        else:
            k = int(rng.integers(total))
            cum = counts.cumsum()
            iy = int(np.searchsorted(cum, k, side="right"))
            rank = k - (int(cum[iy - 1]) if iy else 0)
            ix = int(np.flatnonzero(live[iy])[rank])
        x, y = int(xs[ix]), int(ys[iy])
        points.append((x, y))
        x0, x1 = np.searchsorted(xs, [x - size + 1, x + size])
        y0, y1 = np.searchsorted(ys, [y - size + 1, y + size])
        region = live[y0:y1, x0:x1]
        removed = region.sum(axis=1, dtype=np.int64)
        counts[y0:y1] -= removed
        total -= int(removed.sum())
        region[:] = False
    assert not live.any() and not counts.any()
    return points


def grid(valid, size, phase=(0, 0)):
    ox, oy = phase
    loc = np.argwhere(valid[oy::size, ox::size])
    return [(int(x * size + ox), int(y * size + oy)) for y, x in loc]


def best_grid(valid, size):
    h, w = valid.shape
    hp, wp = ((h + size - 1) // size) * size, ((w + size - 1) // size) * size
    padded = np.zeros((hp, wp), dtype=bool)
    padded[:h, :w] = valid
    scores = padded.reshape(hp // size, size, wp // size, size).sum(axis=(0, 2), dtype=np.int64)
    oy, ox = np.unravel_index(np.argmax(scores), scores.shape)
    return grid(valid, size, (int(ox), int(oy))), (int(ox), int(oy)), scores


def literal_raster(mask, size, step, exclusive=True):
    """Literal reproduction of the legacy greedy loops (no filesystem writes)."""
    h, w = mask.shape
    occ = np.zeros_like(mask)
    points = []
    for y in range(0, h - size + (0 if exclusive else 1), step):
        for x in range(0, w - size + (0 if exclusive else 1), step):
            if occ[y:y + size, x:x + size].any() or not mask[y:y + size, x:x + size].all():
                continue
            occ[y:y + size, x:x + size] = True
            points.append((x, y))
    return points


def verify(mask, points, size):
    occ = np.zeros_like(mask)
    h, w = mask.shape
    for x, y in points:
        assert 0 <= x <= w - size and 0 <= y <= h - size
        assert mask[y:y + size, x:x + size].all(), (x, y, "outside foreground")
        assert not occ[y:y + size, x:x + size].any(), (x, y, "overlap")
        occ[y:y + size, x:x + size] = True
    assert int(occ.sum()) == len(points) * size * size
    return occ


def self_tests():
    rng = np.random.default_rng(20260903)
    for n in range(30):
        mask = np.ones((8, 9), dtype=bool) if n == 0 else rng.random((8, 9)) > 0.12
        s = 2
        valid = valid_positions(mask, s)
        brute = np.array([[mask[y:y + s, x:x + s].all() for x in range(8)] for y in range(7)])
        assert np.array_equal(valid, brute)
        for step in (1, 2, 3):
            xs, ys = np.arange(0, 9 - s, step), np.arange(0, 8 - s, step)
            assert sequential(valid, xs, ys, s) == literal_raster(mask, s, step)
        pts, phase, scores = best_grid(valid, s)
        brute_scores = np.array([[len(grid(valid, s, (x, y))) for x in range(s)] for y in range(s)])
        assert np.array_equal(scores, brute_scores)
        assert len(pts) == int(scores.max())
        for seed in (0, 1, 2):
            pts = sequential(valid, np.arange(valid.shape[1]), np.arange(valid.shape[0]), s, seed)
            occ = verify(mask, pts, s)
            assert not valid_positions(mask & ~occ, s).any()
            assert pts == sequential(valid, np.arange(valid.shape[1]), np.arange(valid.shape[0]), s, seed)
    return {"synthetic_masks": 30, "random_saturation_and_reproducibility_checks": 90, "status": "passed"}


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=SEGMENT_DIR, help="folder with *_nobg.png")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--output", type=Path, default=HERE / "results_placement_comparison")
    ap.add_argument("--legacy-packs", type=Path, default=ROOT / "work" / "legacy_greedy_packing",
                    help="optional legacy greedy output used for pixel-level reproduction checks")
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty results: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tests = self_tests()
    print("Synthetic unit tests passed", flush=True)

    files = sorted(args.input.glob("*_nobg.png"))
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
                "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(), "seed_count": args.seeds,
                "swatch_px": SIZE, "foreground_rule": f"alpha > {ALPHA_THRESHOLD}", "inputs": []}
    runs, placements, summaries, reproductions, saturation = [], [], [], [], []
    for src in files:
        stem = src.stem.replace("_nobg", "")
        gid = GARMENT_IDS.get(stem, stem)
        with Image.open(src) as im:
            rgba = np.array(im.convert("RGBA"))
        mask = rgba[:, :, 3] > ALPHA_THRESHOLD
        h, w = mask.shape
        t0 = time.perf_counter()
        valid = valid_positions(mask, SIZE)
        prep_s = time.perf_counter() - t0
        xs10, ys10 = np.arange(0, w - SIZE, 10), np.arange(0, h - SIZE, 10)
        xs1, ys1 = np.arange(valid.shape[1]), np.arange(valid.shape[0])
        manifest["inputs"].append({"garment": gid, "stem": stem, "source": str(src), "sha256": sha(src),
                                   "width": w, "height": h, "foreground_px": int(mask.sum()),
                                   "valid_1px_candidates": int(valid.sum()),
                                   "valid_10px_candidates": int(valid[np.ix_(ys10, xs10)].sum()),
                                   "shared_feasibility_seconds": prep_s,
                                   "loose_area_upper_bound": int(mask.sum()) // (SIZE * SIZE)})
        greedy = sequential(valid, xs10, ys10, SIZE)
        assert greedy == literal_raster(mask, SIZE, 10)
        matches = 0
        for i, (x, y) in enumerate(greedy, 1):
            old = args.legacy_packs / stem / f"{stem}_crop_{i:02d}.png"
            if old.exists():
                with Image.open(old) as im:
                    matches += int(np.array_equal(np.array(im.convert("RGBA")), rgba[y:y + SIZE, x:x + SIZE]))
        reproductions.append({"garment": gid, "recomputed_greedy_count": len(greedy),
                              "literal_and_optimized_coordinates_equal": True, "saved_patch_matches": matches})
        print(f"{gid}: greedy {len(greedy)}; feasible positions {int(valid.sum())}", flush=True)
        for method in METHODS:
            trials = range(args.seeds) if method.startswith("random") else range(5)
            method_runs, method_points, phase = [], [], (0, 0)
            for trial in trials:
                seed = trial if method.startswith("random") else None
                t0 = time.perf_counter()
                if method == "greedy_10px":
                    pts = sequential(valid, xs10, ys10, SIZE)
                elif method == "greedy_1px":
                    pts = sequential(valid, xs1, ys1, SIZE)
                elif method == "grid_fixed":
                    pts = grid(valid, SIZE)
                elif method == "grid_best_phase":
                    pts, phase, scores = best_grid(valid, SIZE)
                elif method == "random_1px":
                    pts = sequential(valid, xs1, ys1, SIZE, seed)
                else:
                    pts = sequential(valid, xs10, ys10, SIZE, seed)
                elapsed = time.perf_counter() - t0
                verify(mask, pts, SIZE)
                rec = {"garment": gid, "method": method, "trial": trial, "seed": seed, "count": len(pts),
                       "foreground_px": int(mask.sum()), "packed_px": len(pts) * SIZE * SIZE,
                       "coverage_pct": 100 * len(pts) * SIZE * SIZE / int(mask.sum()),
                       "selection_seconds": elapsed, "phase_x": phase[0], "phase_y": phase[1]}
                runs.append(rec)
                method_runs.append(rec)
                method_points.append(pts)
                if seed is not None or trial == 0:
                    placements.append({"garment": gid, "method": method, "trial": trial, "seed": seed,
                                       "box_px": SIZE, "xy": pts})
            counts = np.array([r["count"] for r in method_runs])
            summaries.append({"garment": gid, "method": method, "repeats": len(method_runs),
                              "mean_count": float(counts.mean()), "sd_count": float(counts.std(ddof=1)),
                              "min_count": int(counts.min()), "max_count": int(counts.max()),
                              "median_count": float(np.median(counts)),
                              "mean_coverage_pct": float(np.mean([r["coverage_pct"] for r in method_runs])),
                              "median_selection_ms": 1000 * statistics.median(r["selection_seconds"] for r in method_runs),
                              "beats_greedy_pct": float(100 * np.mean(counts > len(greedy))),
                              "ties_greedy_pct": float(100 * np.mean(counts == len(greedy))),
                              "phase_x": phase[0], "phase_y": phase[1]})
            audit = sorted({0, int(np.argmin(counts)), int(np.argmax(counts))}) if method.startswith("random") else [0]
            for trial in audit:
                occ = verify(mask, method_points[trial], SIZE)
                remaining = valid_positions(mask & ~occ, SIZE)
                full_n, ten_n = int(remaining.sum()), int(remaining[np.ix_(ys10, xs10)].sum())
                if method in ("random_1px", "greedy_1px"):
                    assert full_n == 0
                if method in ("random_10px", "greedy_10px"):
                    assert ten_n == 0
                saturation.append({"garment": gid, "method": method, "trial": trial,
                                   "remaining_1px_candidates": full_n, "remaining_10px_candidates": ten_n})
            if method == "grid_best_phase":
                np.save(out / f"{gid}_all_grid_phase_counts.npy", scores)
            print(f"  {method}: mean {counts.mean():.2f}, range {counts.min()}-{counts.max()}", flush=True)
    save_csv(out / "runs.csv", runs)
    save_csv(out / "summary.csv", summaries)
    totals = []
    for method in METHODS:
        per_trial = {}
        for r in runs:
            if r["method"] == method:
                per_trial.setdefault(r["trial"], []).append(r)
        sums = np.array([sum(x["count"] for x in v) for v in per_trial.values()])
        fg = sum(x["foreground_px"] for x in next(iter(per_trial.values())))
        packed = np.mean([sum(x["packed_px"] for x in v) for v in per_trial.values()])
        totals.append({"method": method, "mean_total_count": float(sums.mean()),
                       "sd_total_count": float(sums.std(ddof=1)) if len(sums) > 1 else 0.0,
                       "min_total_count": int(sums.min()), "max_total_count": int(sums.max()),
                       "pooled_coverage_pct": 100 * packed / fg})
    save_csv(out / "totals.csv", totals)
    with (out / "placements.jsonl").open("w", encoding="utf-8") as f:
        for row in placements:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_json(out / "manifest.json", manifest)
    save_json(out / "validation.json", {"unit_tests": tests, "source_reproduction": reproductions,
                                        "number_of_runs": len(runs), "independent_saturation_checks": saturation})
    print("COMPLETE", out, flush=True)
    for t in totals:
        print(f"{t['method']:16s} total {t['mean_total_count']:7.2f}  coverage {t['pooled_coverage_pct']:.2f}%")


if __name__ == "__main__":
    main()
