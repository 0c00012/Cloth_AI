# -*- coding: utf-8 -*-
"""Unit tests for the grid-origin swatch extraction (run: pytest Final_pipeline/tests)."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ex = importlib.import_module("2_extract_swatches")


def brute_valid(mask, s):
    h, w = mask.shape
    return np.array([[mask[y:y + s, x:x + s].all() for x in range(w - s + 1)] for y in range(h - s + 1)])


@pytest.mark.parametrize("seed", range(12))
def test_valid_positions_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((23, 31)) > 0.15
    for s in (2, 3, 5):
        assert np.array_equal(ex.valid_top_left(mask, s), brute_valid(mask, s))


@pytest.mark.parametrize("seed", range(12))
def test_grid_origin_counts_match_direct_enumeration(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((40, 37)) > 0.2
    s = 4
    valid = ex.valid_top_left(mask, s)
    counts = ex.grid_origin_counts(valid, s)
    assert counts.shape == (s, s)
    for oy in range(s):
        for ox in range(s):
            assert counts[oy, ox] == len(ex.grid_cells(valid, s, (ox, oy)))


def test_selected_grid_is_valid_non_overlapping_and_maximal_within_family():
    rng = np.random.default_rng(3)
    mask = rng.random((60, 50)) > 0.1
    s = 6
    valid = ex.valid_top_left(mask, s)
    cells, origin, counts, info = ex.optimise_grid_origin(valid, s)
    occ = ex.verify(mask, cells, s)                      # containment / overlap / area asserts
    assert len(cells) == int(counts.max()) == info["max_count"]
    assert 0 <= origin[0] < s and 0 <= origin[1] < s
    # first-maximum rule: smallest y, then smallest x
    oy, ox = np.argwhere(counts == counts.max())[0]
    assert origin == (int(ox), int(oy))
    assert occ.sum() == len(cells) * s * s


def test_tie_break_prefers_larger_boundary_margin():
    # a full rectangle: every origin gives the same count, so the margin rule decides
    mask = np.ones((30, 30), dtype=bool)
    mask[:2, :] = False
    mask[-2:, :] = False
    s = 5
    valid = ex.valid_top_left(mask, s)
    margin = ex.cell_margin_map(mask, s)
    cells, origin, counts, info = ex.optimise_grid_origin(valid, s, margin)
    assert info["tied_origins"] > 1
    assert info["min_cell_margin_px"] >= info["first_tie_min_margin_px"]
    assert len(cells) == int(counts.max())
    ex.verify(mask, cells, s)


def test_cell_margin_map_equals_min_distance_over_cell():
    from scipy.ndimage import distance_transform_edt
    mask = np.ones((20, 24), dtype=bool)
    mask[:, :3] = False
    mask[10:, 15:] = False
    s = 4
    dt = distance_transform_edt(mask)
    m = ex.cell_margin_map(mask, s)
    for y in range(0, 20 - s + 1, 3):
        for x in range(0, 24 - s + 1, 5):
            assert m[y, x] == pytest.approx(dt[y:y + s, x:x + s].min())


def test_remaining_capacity_zero_means_no_square_fits():
    mask = np.ones((8, 8), dtype=bool)
    s = 4
    valid = ex.valid_top_left(mask, s)
    cells, origin, counts, _ = ex.optimise_grid_origin(valid, s)
    occ = ex.verify(mask, cells, s)
    assert len(cells) == 4
    assert ex.remaining_capacity(mask, occ, s) == 0
