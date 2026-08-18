"""The hand identity guard counts crossings, not label changes.

The first version counted label changes alone. On session 6 it rejected all
four clips, including the single-hand control, which flipped its label once in
149 frames with no second hand anywhere in the clip. Rejecting a clip for a
detector mislabelling one hand is a false alarm, and a guard that measures a
proxy for the defect it exists to catch is its own class of defect.

The rule these pin down: a switch counts only where two hands were actually
detected on one side of it, because the selector can only cross between two
hands when two hands are in frame.
"""

from __future__ import annotations


def count(sides: list[str], seen: list[int]) -> tuple[int, int]:
    """The guard's arithmetic, lifted out of Stage 3 so it can be tested.

    Mirrors the loop in s03_estimate exactly. Returns crossings and mislabels.
    """
    switches = 0
    mislabels = 0
    for a, b, na, nb in zip(sides, sides[1:], seen, seen[1:], strict=False):
        if a == b:
            continue
        if max(na, nb) >= 2:
            switches += 1
        else:
            mislabels += 1
    return switches, mislabels


class TestGuardArithmetic:
    def test_a_stable_single_hand_clip_is_clean(self):
        assert count(["Right"] * 20, [1] * 20) == (0, 0)

    def test_one_flip_with_only_one_hand_present_is_a_mislabel(self):
        """Session 6's control clip: 148 Right, 1 Left, never two hands."""
        sides = ["Right"] * 80 + ["Left"] + ["Right"] * 68
        switches, mislabels = count(sides, [1] * len(sides))
        assert switches == 0
        assert mislabels == 2      # into the bad frame and back out of it

    def test_a_flip_with_two_hands_present_is_a_real_crossing(self):
        sides = ["Right"] * 10 + ["Left"] * 10
        seen = [2] * 20
        switches, mislabels = count(sides, seen)
        assert switches == 1
        assert mislabels == 0

    def test_a_bimanual_clip_that_crosses_repeatedly_is_caught(self):
        sides = ["Right", "Left"] * 15
        switches, _ = count(sides, [2] * 30)
        assert switches == 29

    def test_two_hands_on_either_side_of_the_change_is_enough(self):
        """The crossing frame itself may report one hand while the other is
        briefly occluded. Either side seeing two hands is what matters."""
        assert count(["Right", "Left"], [2, 1]) == (1, 0)
        assert count(["Right", "Left"], [1, 2]) == (1, 0)

    def test_mislabels_do_not_mask_a_genuine_crossing(self):
        sides = ["Right", "Left", "Left", "Right"]
        seen = [1, 1, 2, 2]
        switches, mislabels = count(sides, seen)
        assert switches == 1       # the Left to Right change, two hands present
        assert mislabels == 1      # the first flip, one hand only

    def test_an_empty_clip_does_not_raise(self):
        assert count([], []) == (0, 0)
