"""Contact-based grasp detection.

The width test scored 57.9% against ground truth on a handle grasp, which is
about what guessing the majority class scores. These cover the cases that
distinguish contact-and-carry from finger separation.
"""

from __future__ import annotations

import numpy as np

from wristview.grasp import (
    detect_grasp_by_contact,
    fingertip_distances,
    relative_pose_stability,
)


def hand_at(position, spread=0.05):
    """A 21-joint hand whose palm is axis-aligned, placed at `position`."""
    position = np.asarray(position, dtype=float)
    lm = np.zeros((21, 3))
    lm[0] = position                          # wrist
    lm[5] = position + np.array([0.04, 0.0, 0.09])   # index MCP
    lm[9] = position + np.array([0.0, 0.0, 0.10])    # middle MCP
    lm[17] = position + np.array([-0.04, 0.0, 0.085])
    lm[4] = position + np.array([spread, 0.0, 0.10])     # thumb tip
    lm[8] = position + np.array([-spread, 0.0, 0.12])    # index tip
    lm[12] = position + np.array([0.0, 0.0, 0.13])       # middle tip
    return lm


def sequence(n, hand_path, object_path):
    lm = np.array([hand_at(p) for p in hand_path])
    return lm, np.ones(n, bool), np.array(object_path), np.ones(n, bool)


class TestFingertipDistances:
    def test_zero_at_the_surface(self):
        lm, v, obj, ov = sequence(1, [[0, 0, 0]], [[0.0, 0.0, 0.13]])
        d = fingertip_distances(lm, v, obj, ov, 0.0)
        assert abs(d[0]) < 1e-9

    def test_radius_is_subtracted(self):
        lm, v, obj, ov = sequence(1, [[0, 0, 0]], [[0.0, 0.0, 0.23]])
        assert abs(fingertip_distances(lm, v, obj, ov, 0.04)[0] - 0.06) < 1e-9

    def test_nan_where_the_object_is_unknown(self):
        lm, v, obj, _ = sequence(1, [[0, 0, 0]], [[0, 0, 0.2]])
        assert np.isnan(fingertip_distances(lm, v, obj, np.zeros(1, bool), 0.0)[0])


class TestRelativeStability:
    def test_a_carried_object_reads_still(self):
        """Hand and object translate together: motion in world, none in hand."""
        n = 20
        path = [[i * 0.02, 0.0, 0.0] for i in range(n)]
        obj = [[p[0], p[1], p[2] + 0.13] for p in path]
        lm, v, o, ov = sequence(n, path, obj)
        w = relative_pose_stability(lm, v, o, ov, 5)
        assert np.nanmax(w) < 1e-6

    def test_a_stationary_object_reads_unstable_when_the_hand_moves(self):
        n = 20
        path = [[i * 0.02, 0.0, 0.0] for i in range(n)]
        obj = [[0.0, 0.0, 0.13]] * n
        lm, v, o, ov = sequence(n, path, obj)
        w = relative_pose_stability(lm, v, o, ov, 5)
        assert np.nanmedian(w) > 0.01


class TestDetectGraspByContact:
    def test_a_handle_grasp_is_detected_though_the_fingers_never_close(self):
        """The case the width test cannot see.

        Finger separation is held constant at a wide 5 cm for the whole
        episode, exactly as it is when holding a mug by the handle. Contact and
        carry still identify the grasp.
        """
        n = 60
        path = [[0.0, 0.0, 0.0]] * 20 + [[i * 0.01, 0.0, 0.0] for i in range(20)] \
            + [[0.19, 0.0, 0.0]] * 20
        obj = []
        for i, p in enumerate(path):
            obj.append([p[0], 0.0, p[2] + 0.13] if i >= 20 else [0.0, 0.0, 0.13])
        lm, v, o, ov = sequence(n, path, obj)
        closed, diag = detect_grasp_by_contact(lm, v, o, ov, 20.0,
                                               {"object_radius_m": 0.0})
        assert diag["signal"] == "contact_and_carry"
        assert closed.sum() > 20
        assert diag["grasp_onset_frame"] is not None

    def test_a_hand_moving_away_from_a_static_object_is_not_a_grasp(self):
        n = 40
        path = [[i * 0.02, 0.0, 0.0] for i in range(n)]
        obj = [[0.0, 0.0, 0.13]] * n
        lm, v, o, ov = sequence(n, path, obj)
        closed, _ = detect_grasp_by_contact(lm, v, o, ov, 20.0, {"object_radius_m": 0.0})
        # It may touch at the start, but it must not stay closed.
        assert closed.sum() < n // 2

    def test_no_object_reports_unavailable_rather_than_guessing(self):
        n = 10
        lm, v, o, _ = sequence(n, [[0, 0, 0]] * n, [[0, 0, 0.13]] * n)
        closed, diag = detect_grasp_by_contact(lm, v, o, np.zeros(n, bool), 20.0, {})
        assert diag["signal"] == "unavailable"
        assert not closed.any()
        assert "guessed" in diag["reason"]

    def test_the_flag_does_not_chatter(self):
        n = 60
        rng = np.random.default_rng(0)
        path = [[0.0, 0.0, 0.0]] * n
        obj = [[0.0, 0.0, 0.13 + rng.normal(0, 0.001)] for _ in range(n)]
        lm, v, o, ov = sequence(n, path, obj)
        _, diag = detect_grasp_by_contact(lm, v, o, ov, 20.0, {"object_radius_m": 0.0})
        assert diag["transitions"] <= 2
