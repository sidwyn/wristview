"""The ArUco surface this pipeline depends on must exist and must work.

Every metric measurement here traces back to a printed ArUco marker of known
side. `_fit_aruco_scale` turns COLMAP's arbitrary units into metres with it, and
`qc.marker_agreement` uses it as the independent check on Stage 2's poses. If
`cv2.aruco` is missing the loss is quiet in the way that matters: `import cv2`
still succeeds, Stage 0 runs, Stage 1 runs for the best part of an hour, and the
failure only surfaces deep inside scale estimation.

The constraint is a version floor, not a distribution choice. ArUco moved out of
opencv_contrib into the main `objdetect` module in **OpenCV 4.7.0**, so plain
`opencv-python` has carried it since then and `pyproject.toml` pins `>=4.9`.
Below 4.7 the contrib build would be required. That floor is the thing worth
protecting, because it is invisible in the manifest without a comment and
nothing else in the codebase would fail if someone relaxed it.

Measured, not assumed: on a clean install of `opencv-python` 4.11.0.86 with no
contrib package present (checked against the contrib-only canaries below), the
marker fit over real31scan's 745 registered frames returned the same 390
detections of id 0 and a byte-identical SHA-256 of every detected corner as both
a contrib-only install and this machine's mixed install.

These tests have no fixtures and run in milliseconds, so the diagnosis arrives
before anything slower has a chance to fail confusingly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

DICT = "DICT_4X4_50"
MARKER_ID = 0

# Exactly what the pipeline calls: `s01_scene._fit_aruco_scale` and
# `qc.estimate_marker_world_pose` use the detector, `marker.generate` builds the
# printable sheet. Nothing here needs anything else from the module.
REQUIRED_SYMBOLS = (
    "getPredefinedDictionary",
    "ArucoDetector",
    "DetectorParameters",
    "generateImageMarker",
    DICT,
)

FIX = (
    "Install an OpenCV of at least 4.7, which is where ArUco moved into the "
    "main objdetect module:\n"
    "    pip install 'opencv-python>=4.9'\n"
    "On a headless server use opencv-python-headless at the same version."
)


class TestArucoAvailability:
    def test_the_aruco_module_is_present(self):
        assert hasattr(cv2, "aruco"), (
            f"cv2 {cv2.__version__} has no `aruco` module, so this install "
            f"cannot read the printed marker that gives the reconstruction "
            f"metric scale. Stage 1 would produce an unscaled world and every "
            f"distance downstream would be wrong by an unknown factor.\n{FIX}"
        )

    @pytest.mark.parametrize("symbol", REQUIRED_SYMBOLS)
    def test_required_symbol_exists(self, symbol):
        """Presence of the module is not the same as the API we call."""
        assert hasattr(cv2.aruco, symbol), (
            f"cv2.aruco has no `{symbol}` in OpenCV {cv2.__version__}. The "
            f"pipeline calls it directly, so the marker path cannot run.\n{FIX}"
        )

    def test_opencv_is_at_least_the_version_that_carries_aruco(self):
        """Pin the reason the floor in pyproject.toml exists.

        Without this, someone relaxing `opencv-python>=4.9` to something older
        gets a green test suite and a pipeline with no metric anchor.
        """
        major, minor = (int(part) for part in cv2.__version__.split(".")[:2])
        assert (major, minor) >= (4, 7), (
            f"OpenCV {cv2.__version__} predates 4.7, where ArUco moved from "
            f"opencv_contrib into the main objdetect module. At this version "
            f"plain opencv-python has no `cv2.aruco` and the contrib build is "
            f"mandatory.\n{FIX}"
        )

    def test_a_known_marker_round_trips(self):
        """Generate a marker of a known id, then read it back."""
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
        tile = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, 200)

        # The quiet zone is part of the marker specification. Detection against
        # a flush edge is unreliable, so omitting it would make this the test's
        # failure rather than the install's.
        canvas = np.full((320, 320), 255, np.uint8)
        canvas[60:260, 60:260] = tile

        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(canvas)

        assert ids is not None, (
            f"a freshly generated {DICT} marker was not detected at all, so the "
            f"aruco module is present but not working in cv2 {cv2.__version__}."
        )
        assert [int(v) for v in ids.ravel()] == [MARKER_ID], (
            f"expected to decode id {MARKER_ID}, got {ids.ravel().tolist()}"
        )
        assert corners[0].reshape(-1, 2).shape == (4, 2), (
            "a detected marker must come back as four corners"
        )

    def test_detected_corners_land_on_the_drawn_square(self):
        """The corners must be the marker's, not merely four numbers.

        A detector returning plausible-looking garbage would satisfy the checks
        above. The marker was drawn at a known place, so pin the geometry: this
        is what makes the measured side length trustworthy.
        """
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
        canvas = np.full((320, 320), 255, np.uint8)
        canvas[60:260, 60:260] = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, 200)

        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, _, _ = detector.detectMarkers(canvas)
        found = corners[0].reshape(-1, 2)
        expected = np.array([[60.0, 60.0], [259.0, 60.0], [259.0, 259.0], [60.0, 259.0]])

        def ordered(points: np.ndarray) -> np.ndarray:
            """Sort by angle about the centroid, so corner-ordering conventions
            cannot make a correct detection look wrong."""
            centre = points.mean(axis=0)
            angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
            return points[np.argsort(angles)]

        assert np.allclose(ordered(found), ordered(expected), atol=2.0), (
            f"detected corners {found.tolist()} are not the square drawn at "
            f"{expected.tolist()}"
        )
