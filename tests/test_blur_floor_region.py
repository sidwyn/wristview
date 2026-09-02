"""The absolute blur floor reads scene TEXTURE, not focus.

Variance of the Laplacian scales with how much detail the scene contains. The
per-pass rule in `_filter_frames` is relative for exactly that reason. The
absolute floor is the one number left that is not, and real31 showed what that
costs: 0.0323, measured over real26/d's woven mat, was carried into a room with
a bare bamboo desk, where a pass shot IN FOCUS at 1 to 3 m cleared it on 31.5
per cent of frames. A pass in focus cannot be 68 per cent blurred.

Two things were tried before the floor was re-measured, and both are recorded
here so they are not tried again blind.

1. A region-aware floor, judging the sharpest ninth. Rejected. It moved real31's
   close pass only 8.7 to 17.8 per cent, while loosening real27's known
   too-soft mat pass from 59.1 to 73.1, which is the fault the floor exists to
   catch.

2. Normalising each tile by its OWN contrast. Rejected, and it was wrong rather
   than merely unhelpful. A flat patch of shadow has almost no contrast, so
   dividing by it amplified quantisation noise into a high score. On real31 the
   winning tile was the LOW-contrast one on 72.5 per cent of frames, median
   winning contrast 5.18 grey levels against 20.14 across all tiles, contrast
   and score correlating -0.318. It reported that pass as sharper than the
   reference low pass. `best_tile_normalised_variance` now divides by the
   FRAME's contrast, which keeps exposure invariance without rewarding flat
   tiles.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from wristview.videoio import (
    best_tile_normalised_variance,
    normalised_laplacian_variance,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"
STAGE = ROOT / "src" / "wristview" / "stages" / "s00_ingest.py"


def test_tiles_one_is_exactly_the_old_measure():
    """The shipped path must be the measure the floor was calibrated on."""
    rng = np.random.default_rng(0)
    image = np.clip(rng.normal(128, 40, (360, 640)), 0, 255).astype(np.uint8)
    assert best_tile_normalised_variance(image, 1) == pytest.approx(
        normalised_laplacian_variance(image), rel=1e-9
    )


def test_a_flat_tile_cannot_win_the_region_vote():
    """The bug that made real31 look sharper than its reference.

    One tile of real texture, the rest flat. The flat tiles must score near
    zero. Dividing each tile by its own contrast made them score highest.
    """
    rng = np.random.default_rng(1)
    image = np.full((360, 640), 100, np.uint8)
    image[:120, :213] = np.clip(rng.normal(128, 50, (120, 213)), 0, 255).astype(np.uint8)

    grey = image.astype(np.float64)
    scaled = (grey - grey.mean()) / grey.std()
    textured = float(cv2.Laplacian(scaled[:120, :213], cv2.CV_64F).var())
    flat = float(cv2.Laplacian(scaled[240:360, 427:640], cv2.CV_64F).var())

    assert textured > flat * 100, (
        f"the textured tile scores {textured:.4f} and a flat one {flat:.6f}; "
        f"if these were close the region vote would pick shadow over detail"
    )
    assert best_tile_normalised_variance(image, 3) == pytest.approx(textured, rel=1e-6)


def test_the_region_measure_never_scores_below_the_whole_frame_one():
    rng = np.random.default_rng(2)
    for _ in range(4):
        image = np.clip(rng.normal(120, 35, (360, 640)), 0, 255).astype(np.uint8)
        assert best_tile_normalised_variance(image, 3) >= \
            best_tile_normalised_variance(image, 1) - 1e-9


def test_tiles_must_be_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        best_tile_normalised_variance(np.zeros((64, 64), np.uint8), 0)


def test_a_tile_too_small_to_measure_falls_back_rather_than_lying():
    tiny = np.random.default_rng(4).integers(0, 255, (20, 20), dtype=np.uint8)
    assert best_tile_normalised_variance(tiny, 8) == pytest.approx(
        best_tile_normalised_variance(tiny, 1)
    )


def test_the_shipped_config_uses_the_calibrated_pair():
    """0.0323 is a whole-frame level. It is meaningless on another grid."""
    ingest = yaml.safe_load(CONFIG.read_text())["ingest"]
    tiles = int(ingest["blur_floor_tiles"])
    floor = float(ingest["blur_normalised_floor"])
    assert tiles == 1, (
        "the region rule was measured and rejected on real31; see this "
        "module's docstring before enabling it again"
    )
    assert floor == pytest.approx(0.0323), (
        f"the shipped floor is the real26/d whole-frame calibration 0.0323, "
        f"not {floor}"
    )


def test_the_config_says_the_floor_is_per_room():
    """The failure was carrying one room's number into another in silence."""
    text = CONFIG.read_text()
    assert "PER ROOM" in text, (
        "the config must say the floor does not transfer between sessions, "
        "or real31's mistake is repeated by the next person"
    )
    assert "31.5" in text, "record the evidence: an in-focus pass cleared it 31.5% of the time"


def test_stage_0_reads_the_grid():
    """A config key nothing reads is this project's most repeated defect."""
    source = STAGE.read_text()
    assert "best_tile_normalised_variance(image, blur_floor_tiles)" in source
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.keyword) and node.arg == "blur_floor_tiles"
        for node in ast.walk(tree)
    ), "the call site must pass blur_floor_tiles through"
