"""Hero window selection, on synthetic motion where the right answer is known.

The real failure this guards against is a window that runs past the grasp. The
composited object stays where the scan left it, so every frame past the tail
shows a cube resting on a desk the operator is no longer touching, and the
illusion breaks in exactly the asset that gets shown to people.

These cover the arithmetic only. ffmpeg is not exercised anywhere here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.hero_window import (  # noqa: E402
    ClipScore,
    choose_window,
    describe_grasp,
    find_grasp_frame,
    find_reach_start,
    lateral_displacement,
    rank_clips,
    reach_search_bounds,
    right_panel,
)

UP = np.array([0.0, 0.0, 1.0])


def poses_from_centres(centres: np.ndarray) -> np.ndarray:
    """Identity-rotation poses at the given centres."""
    out = np.tile(np.eye(4), (len(centres), 1, 1))
    out[:, :3, 3] = centres
    return out


def pick_and_carry(
    rest_frames: int = 100,
    carry_frames: int = 100,
    slide_m: float = 0.12,
    lift_m: float = 0.04,
    jitter_m: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """A cube that sits still, then slides across the desk while lifting.

    The slide is along +x and the lift is along the surface normal, so the
    lateral answer is the slide alone and the 3D answer is larger.
    """
    rng = np.random.default_rng(seed)
    centres = np.zeros((rest_frames + carry_frames, 3))
    ramp = np.linspace(0.0, 1.0, carry_frames)
    centres[rest_frames:, 0] = ramp * slide_m
    centres[rest_frames:, 2] = ramp * lift_m
    if jitter_m:
        centres += rng.normal(0, jitter_m, centres.shape)
    return centres


class TestLateralDisplacement:
    def test_measures_the_slide_and_ignores_the_lift(self):
        centres = pick_and_carry(slide_m=0.12, lift_m=0.04)
        poses = poses_from_centres(centres)
        valid = np.ones(len(poses), dtype=bool)

        lateral = lateral_displacement(poses, valid, UP)
        assert lateral[:100].max() == pytest.approx(0.0, abs=1e-12)
        assert lateral[-1] == pytest.approx(0.12, abs=1e-9)

    def test_without_a_normal_it_falls_back_to_3d_distance(self):
        centres = pick_and_carry(slide_m=0.12, lift_m=0.04)
        poses = poses_from_centres(centres)
        valid = np.ones(len(poses), dtype=bool)

        full = lateral_displacement(poses, valid, None)
        assert full[-1] == pytest.approx(np.hypot(0.12, 0.04), abs=1e-9)

    def test_the_resting_place_is_the_median_of_the_settle_window(self):
        """One wild frame inside the settle window must not move the origin."""
        centres = pick_and_carry(slide_m=0.10, lift_m=0.0)
        centres[3] = [5.0, 5.0, 5.0]
        poses = poses_from_centres(centres)
        valid = np.ones(len(poses), dtype=bool)

        lateral = lateral_displacement(poses, valid, UP)
        assert lateral[-1] == pytest.approx(0.10, abs=1e-9)

    def test_invalid_frames_report_nan_not_zero(self):
        centres = pick_and_carry()
        poses = poses_from_centres(centres)
        valid = np.ones(len(poses), dtype=bool)
        valid[120:130] = False

        lateral = lateral_displacement(poses, valid, UP)
        assert np.isnan(lateral[120:130]).all()
        assert np.isfinite(lateral[130:]).all()

    def test_rejects_a_pose_array_of_the_wrong_shape(self):
        with pytest.raises(ValueError, match=r"\(N, 4, 4\)"):
            lateral_displacement(np.zeros((10, 3)), np.ones(10, bool), UP)

    def test_rejects_mismatched_validity(self):
        with pytest.raises(ValueError, match="validity"):
            lateral_displacement(poses_from_centres(np.zeros((10, 3))), np.ones(9, bool), UP)


class TestFindGraspFrame:
    def test_finds_the_frame_the_object_crosses_the_threshold(self):
        # The cube slides 1 mm per frame from frame 100, so frame 100 + i sits
        # i mm out. The threshold falls between samples on purpose: at a round
        # 20 mm the sample at frame 120 lands on it to within one float ulp, and
        # the test would be asserting a rounding mode rather than the search.
        centres = pick_and_carry(rest_frames=100, carry_frames=100, slide_m=0.099, lift_m=0.0)
        lateral = lateral_displacement(poses_from_centres(centres), np.ones(200, bool), UP)

        assert find_grasp_frame(lateral, threshold_m=0.0205, min_hold_frames=1) == 121
        assert find_grasp_frame(lateral, threshold_m=0.0305, min_hold_frames=1) == 131

    def test_a_single_outlier_frame_does_not_trigger_it(self):
        """One bad pose is common in this tracker. A run of three is not."""
        lateral = np.zeros(200)
        lateral[40] = 0.5           # a lone spike
        lateral[150:] = 0.08        # the real pick

        assert find_grasp_frame(lateral, threshold_m=0.02, min_hold_frames=3) == 150

    def test_it_reports_the_start_of_the_run_not_its_end(self):
        lateral = np.zeros(50)
        lateral[30:] = 0.09
        assert find_grasp_frame(lateral, threshold_m=0.02, min_hold_frames=5) == 30

    def test_a_dropped_track_breaks_the_run(self):
        """An unobserved frame is not evidence that the object moved."""
        lateral = np.zeros(60)
        lateral[20:25] = 0.09
        lateral[22] = np.nan
        lateral[40:] = 0.09

        assert find_grasp_frame(lateral, threshold_m=0.02, min_hold_frames=4) == 40

    def test_returns_none_when_nothing_is_ever_picked_up(self):
        assert find_grasp_frame(np.full(300, 0.004), threshold_m=0.02) is None

    def test_rejects_a_nonsense_hold(self):
        with pytest.raises(ValueError, match="min_hold_frames"):
            find_grasp_frame(np.zeros(10), min_hold_frames=0)


class TestDescribeGrasp:
    def test_reports_the_margin_over_the_resting_noise(self):
        rng = np.random.default_rng(0)
        lateral = np.abs(rng.normal(0, 0.002, 300))   # about 4 mm at the 95th
        lateral[200:] = 0.10

        event = describe_grasp(lateral, grasp_frame=200, threshold_m=0.02)
        assert event.frame == 200
        assert event.peak_displacement_m == pytest.approx(0.10, abs=1e-9)
        assert 0.002 < event.rest_noise_p95_m < 0.006
        assert event.margin > 3.0

    def test_the_margin_collapses_when_the_tracker_is_noisy(self):
        """A threshold near the jitter floor is a coin toss, and must read as one."""
        rng = np.random.default_rng(1)
        lateral = np.abs(rng.normal(0, 0.012, 300))
        lateral[200:] = 0.10

        event = describe_grasp(lateral, grasp_frame=200, threshold_m=0.02)
        assert event.margin < 1.5


class TestChooseWindow:
    def test_the_real04_demo_0_window(self):
        """426 frames at 20 fps with the grasp at 200 gives frames 101 to 221."""
        window = choose_window("demo_0", grasp_frame=200, frame_count=426, fps=20.0)

        assert (window.start, window.end) == (101, 221)
        assert window.frames == 120
        assert window.duration_s == pytest.approx(6.0)
        assert window.at_target_length

    def test_it_keeps_exactly_one_second_past_the_grasp(self):
        window = choose_window("c", grasp_frame=200, frame_count=426, fps=20.0, tail_s=1.0)
        frames_after_grasp = window.end - 1 - window.grasp_frame
        assert frames_after_grasp == 20
        assert frames_after_grasp / window.fps == pytest.approx(1.0)

    def test_it_never_runs_past_the_end_of_the_clip(self):
        """The tail is capped by the footage, never by padding it out."""
        window = choose_window("c", grasp_frame=200, frame_count=205, fps=20.0)
        assert window.end == 205

    def test_a_reach_onset_lengthens_the_window(self):
        window = choose_window(
            "c", grasp_frame=200, frame_count=426, fps=20.0, reach_frame=60
        )
        assert window.start == 60
        assert window.duration_s == pytest.approx(8.05)
        assert window.at_target_length

    def test_a_very_early_reach_is_clamped_to_the_maximum(self):
        window = choose_window(
            "c", grasp_frame=200, frame_count=426, fps=20.0, reach_frame=5, max_s=10.0
        )
        assert window.start == 21           # end 221 minus 200 frames
        assert window.duration_s == pytest.approx(10.0)
        assert window.at_target_length

    def test_a_very_late_reach_is_clamped_to_the_minimum(self):
        window = choose_window(
            "c", grasp_frame=200, frame_count=426, fps=20.0, reach_frame=190, min_s=6.0
        )
        assert window.start == 101
        assert window.duration_s == pytest.approx(6.0)

    def test_a_short_clip_gives_a_short_window_and_says_so(self):
        """The tail is the hard rule. A clip too short to fill the band shortens."""
        window = choose_window("c", grasp_frame=10, frame_count=50, fps=20.0)

        assert window.start == 0
        assert window.end == 31
        assert window.duration_s == pytest.approx(1.55)
        assert not window.at_target_length

    def test_the_window_lands_in_the_band_at_a_different_frame_rate(self):
        window = choose_window("c", grasp_frame=600, frame_count=1200, fps=60.0)
        assert window.end == 661
        assert window.duration_s == pytest.approx(6.0)
        assert window.frames == 360

    def test_rejects_a_nonsense_frame_rate(self):
        with pytest.raises(ValueError, match="fps"):
            choose_window("c", grasp_frame=10, frame_count=100, fps=0.0)

    def test_rejects_an_inverted_length_band(self):
        with pytest.raises(ValueError, match="exceeds"):
            choose_window("c", grasp_frame=10, frame_count=100, fps=20.0, min_s=9.0, max_s=7.0)


class TestReachSearchBounds:
    def test_the_band_matches_the_window_the_clamp_allows(self):
        lo, hi = reach_search_bounds(grasp_frame=200, frame_count=426, fps=20.0)
        assert (lo, hi) == (21, 101)

    def test_the_band_never_runs_before_the_first_frame(self):
        lo, hi = reach_search_bounds(grasp_frame=120, frame_count=252, fps=20.0)
        assert lo == 0
        assert hi == 21

    def test_the_band_collapses_rather_than_inverting_on_a_short_clip(self):
        lo, hi = reach_search_bounds(grasp_frame=10, frame_count=50, fps=20.0)
        assert lo == 0
        assert hi >= lo


class TestFindReachStart:
    def test_finds_the_frame_the_hand_starts_moving(self):
        # Still until frame 60, then 20 cm/s along +x at 20 fps.
        positions = np.zeros((200, 3))
        positions[60:, 0] = np.arange(140) * 0.01
        measured = np.ones(200, dtype=bool)

        onset = find_reach_start(positions, measured, fps=20.0, search_lo=0, search_hi=120)
        assert onset == 60

    def test_it_refuses_when_the_hand_track_is_too_sparse(self):
        """A filled pose reports zero speed, which would read as a quiet hand."""
        positions = np.zeros((200, 3))
        positions[60:, 0] = np.arange(140) * 0.01
        measured = np.zeros(200, dtype=bool)
        measured[100:] = True       # only a fifth of the search band is measured

        assert find_reach_start(
            positions, measured, fps=20.0, search_lo=0, search_hi=120
        ) is None

    def test_it_refuses_a_band_shorter_than_the_quiet_run(self):
        positions = np.zeros((200, 3))
        measured = np.ones(200, dtype=bool)
        assert find_reach_start(
            positions, measured, fps=20.0, search_lo=50, search_hi=53, quiet_frames=6
        ) is None

    def test_a_hand_that_never_settles_yields_no_onset(self):
        positions = np.zeros((200, 3))
        positions[:, 0] = np.arange(200) * 0.02      # 40 cm/s throughout
        measured = np.ones(200, dtype=bool)

        assert find_reach_start(
            positions, measured, fps=20.0, search_lo=0, search_hi=120
        ) is None


def score(coverage: float, sharpness: float, near_black: int, frames: int = 120) -> ClipScore:
    return ClipScore(
        coverage_mean=coverage,
        coverage_min=coverage / 2,
        sharpness_mean=sharpness,
        near_black_frames=near_black,
        frames_measured=frames,
    )


class TestRankClips:
    def test_it_reproduces_the_real04_ordering(self):
        """Measured on real04's clean renders over each clip's own hero window."""
        ranked = rank_clips({
            "demo_0": score(0.695, 6335.4, 0),
            "demo_1": score(0.572, 5224.0, 0),
            "demo_3": score(0.483, 4412.8, 16),
        })

        assert [clip for clip, _ in ranked] == ["demo_0", "demo_1", "demo_3"]
        assert ranked[0][1] == pytest.approx(0.5 * 0.695 + 0.3 * 1.0 + 0.2 * 1.0, abs=1e-9)
        assert ranked[2][1] == pytest.approx(
            0.5 * 0.483 + 0.3 * (4412.8 / 6335.4) + 0.2 * (1 - 16 / 120), abs=1e-9
        )

    def test_the_scores_are_strictly_ordered(self):
        ranked = rank_clips({
            "demo_0": score(0.695, 6335.4, 0),
            "demo_1": score(0.572, 5224.0, 0),
            "demo_3": score(0.483, 4412.8, 16),
        })
        totals = [total for _, total in ranked]
        assert totals == sorted(totals, reverse=True)

    def test_near_black_frames_can_overturn_a_coverage_lead(self):
        """A clip that is dark half the time loses to a slightly emptier clip."""
        ranked = rank_clips({
            "dark": score(0.60, 6000.0, 60),
            "clean": score(0.55, 6000.0, 0),
        })
        assert ranked[0][0] == "clean"

    def test_a_tie_breaks_on_the_clip_name_so_reruns_agree(self):
        ranked = rank_clips({
            "demo_9": score(0.5, 100.0, 0),
            "demo_2": score(0.5, 100.0, 0),
        })
        assert ranked[0][0] == "demo_2"

    def test_an_empty_group_ranks_to_nothing(self):
        assert rank_clips({}) == []

    def test_zero_sharpness_everywhere_does_not_divide_by_zero(self):
        ranked = rank_clips({"a": score(0.5, 0.0, 0), "b": score(0.4, 0.0, 0)})
        assert ranked[0][0] == "a"
        assert np.isfinite([total for _, total in ranked]).all()


class TestRightPanel:
    def test_it_takes_the_synthesised_half_not_the_source_half(self):
        frame = np.zeros((480, 1284, 3), dtype=np.uint8)
        frame[:, 644:] = 200            # the wrist view, as laid out by the renderer

        panel = right_panel(frame)
        assert panel.shape == (480, 640, 3)
        assert (panel == 200).all()

    def test_the_divider_is_excluded(self):
        frame = np.zeros((480, 1284, 3), dtype=np.uint8)
        frame[:, 640:644] = 40          # the grey divider
        assert not (right_panel(frame) == 40).any()


class TestClipScore:
    def test_the_near_black_fraction_is_out_of_the_frames_measured(self):
        assert score(0.5, 100.0, 30, frames=120).near_black_fraction == pytest.approx(0.25)

    def test_it_does_not_divide_by_zero_on_an_empty_window(self):
        assert score(0.0, 0.0, 0, frames=0).near_black_fraction == 0.0
