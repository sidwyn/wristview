"""Isolated hand-side labels must not read as the selector crossing hands.

`drop_isolated` judges hand_valid, detected against not detected, and says
nothing about which hand was labelled. A single "Left" inside a run of "Right"
therefore survived it and then scored TWO crossings, Right->Left and
Left->Right, which invalidated the clip. About 11 of 30 clips were lost that
way in one session.
"""

from __future__ import annotations

import numpy as np

from wristview.handqc import drop_isolated, drop_isolated_labels


def count_switches(sides, seen):
    """The identity check from s03_estimate, reproduced."""
    switches = 0
    for a, b, na, nb in zip(sides, sides[1:], seen, seen[1:], strict=False):
        if a != b and max(na, nb) >= 2:
            switches += 1
    return switches


class TestDropIsolatedLabels:
    def test_one_stray_label_is_removed(self):
        sides = ["Right"] * 5 + ["Left"] + ["Right"] * 5
        trusted, report = drop_isolated_labels(sides)
        assert report["labels_dropped"] == 1
        assert report["dropped_indices"] == [5]
        assert not trusted[5]

    def test_the_regression_it_exists_for(self):
        """One stray label scored two crossings and invalidated the clip."""
        sides = ["Right"] * 5 + ["Left"] + ["Right"] * 5
        seen = [2] * 11
        assert count_switches(sides, seen) == 2

        trusted, _ = drop_isolated_labels(sides)
        cleaned = [s for s, ok in zip(sides, trusted, strict=True) if ok]
        cleaned_seen = [n for n, ok in zip(seen, trusted, strict=True) if ok]
        assert count_switches(cleaned, cleaned_seen) == 0

    def test_a_real_crossing_survives(self):
        """A genuine hand change is a sustained run, not one frame."""
        sides = ["Right"] * 5 + ["Left"] * 5
        trusted, report = drop_isolated_labels(sides)
        assert report["labels_dropped"] == 0
        assert count_switches(sides, [2] * 10) == 1

    def test_two_adjacent_strays_are_kept(self):
        """Two in a row support each other, so this is not an isolated label."""
        sides = ["Right"] * 4 + ["Left", "Left"] + ["Right"] * 4
        _, report = drop_isolated_labels(sides)
        assert report["labels_dropped"] == 0

    def test_several_separate_strays(self):
        sides = ["Right", "Right", "Left", "Right", "Right", "Left", "Right", "Right"]
        _, report = drop_isolated_labels(sides)
        assert report["dropped_indices"] == [2, 5]

    def test_a_stray_label_at_either_end_is_dropped(self):
        """The case that got past the first version and cost a real clip.

        demo_13 read {'Right': 250, 'Left': 1} with the Left at position 250,
        the final frame, and was declared INVALID for a hand that never
        changed. One-sided evidence is weaker in principle and overwhelming
        here: a real crossing is sustained, not one boundary frame.
        """
        _, report = drop_isolated_labels(["Right"] * 250 + ["Left"])
        assert report["dropped_indices"] == [250]

        _, report = drop_isolated_labels(["Left"] + ["Right"] * 250)
        assert report["dropped_indices"] == [0]

    def test_demo_13_scores_no_crossing_after_the_fix(self):
        sides = ["Right"] * 250 + ["Left"]
        seen = [2] * 251
        assert count_switches(sides, seen) == 1
        trusted, _ = drop_isolated_labels(sides)
        cleaned = [s for s, ok in zip(sides, trusted, strict=True) if ok]
        cleaned_seen = [n for n, ok in zip(seen, trusted, strict=True) if ok]
        assert count_switches(cleaned, cleaned_seen) == 0

    def test_a_sustained_run_at_the_end_is_not_dropped(self):
        """Two frames support each other, so this stays a real crossing."""
        _, report = drop_isolated_labels(["Right"] * 10 + ["Left", "Left"])
        assert report["labels_dropped"] == 0

    def test_alternating_labels_are_left_alone(self):
        """Neighbours never agree, so nothing is isolated. This is a real fault
        and must stay visible rather than being cleaned away."""
        sides = ["Left", "Right"] * 5
        _, report = drop_isolated_labels(sides)
        assert report["labels_dropped"] == 0
        assert count_switches(sides, [2] * 10) == 9

    def test_too_short_to_judge(self):
        _, report = drop_isolated_labels(["Left", "Right"])
        assert report["labels_dropped"] == 0
        assert "too few" in report["note"]

    def test_report_names_what_it_removed(self):
        _, report = drop_isolated_labels(["Right"] * 3 + ["Left"] + ["Right"] * 3)
        assert report["labels_before"] == 7
        assert report["labels_after"] == 6
        assert report["check"] == "isolated_hand_labels"


class TestSiblingsAreDistinct:
    def test_drop_isolated_cannot_see_a_label_problem(self):
        """The reason a second function was needed at all."""
        valid = np.ones(11, dtype=bool)
        kept, report = drop_isolated(valid)
        assert report["frames_dropped"] == 0
        assert kept.all()
        # Same frames, one stray label, which drop_isolated cannot detect.
        sides = ["Right"] * 5 + ["Left"] + ["Right"] * 5
        _, label_report = drop_isolated_labels(sides)
        assert label_report["labels_dropped"] == 1
