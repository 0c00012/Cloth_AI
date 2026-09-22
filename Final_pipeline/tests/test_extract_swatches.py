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


def test_selected_grid_is_valid_non_overlapping_and_first_maximum():
    rng = np.random.default_rng(3)
    mask = rng.random((60, 50)) > 0.1
    s = 6
    valid = ex.valid_top_left(mask, s)
    cells, origin, counts = ex.optimise_grid_origin(valid, s)
    ex.verify(mask, cells, s)                            # containment / overlap / area asserts
    assert len(cells) == int(counts.max())
    oy, ox = np.argwhere(counts == counts.max())[0]      # smallest y, then smallest x
    assert origin == (int(ox), int(oy))


def test_full_square_mask_is_tiled_completely():
    mask = np.ones((8, 8), dtype=bool)
    cells, origin, counts = ex.optimise_grid_origin(ex.valid_top_left(mask, 4), 4)
    assert len(cells) == 4 and origin == (0, 0)
