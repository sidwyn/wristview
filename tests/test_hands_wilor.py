"""WiLoR against the ten verification stills.

These are the frames from `wristview-videos/a3/`, chosen earlier to test one
question: does a hand estimator keep the fingers on the glass when they wrap
around it and occlude themselves, or does it flatten the hand open. HaMeR was
the original candidate and does not install on Apple Silicon.

Skipped when the frames or the model are absent, so a fresh checkout still
runs green. Marked slow: it loads WiLoR and runs it on full-resolution
frames.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from wristview.backends import hands as hand_backend
from wristview.camera import Intrinsics


def _find_frames_dir() -> Path:
    """The verification stills, wherever the capture folders put them.

    Sessions are filed by date, so the path moved once and silently skipped
    seven tests. Search rather than hardcode.
    """
    root = Path(__file__).resolve().parents[1] / "wristview-videos"
    direct = root / "a3" / "frames"
    if direct.exists():
        return direct
    for candidate in sorted(root.glob("*/a3/frames")):
        return candidate
    return direct


FRAMES_DIR = _find_frames_dir()

# The roles come from a3/README.md. Control frames show an open hand, test
# frames show the fingers wrapped around the glass.
CONTROL = [
    "C005_1-pregrasp_t1.00s",
    "C006_1-pregrasp_t0.80s",
    "C007_4-release-openhand_t5.60s",
]
CONTACT = [
    "C005_2-contact_t1.50s",
    "C005_2b-contact-alt-tightwrap_t2.20s",
    "C005_3-transport_t2.75s",
    "C006_2-contact_t1.20s",
    "C006_3-transport_t2.45s",
    "C007_2-contact_t4.50s",
    "C007_3-transport_t2.50s",
]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not FRAMES_DIR.exists(), reason="verification frames not present"),
]


@pytest.fixture(scope="module")
def estimator():
    available, reason = hand_backend.wilor_available()
    if not available:
        pytest.skip(f"WiLoR unavailable: {reason}")
    return hand_backend.WiLoRHands(device="mps")


@pytest.fixture(scope="module")
def intrinsics() -> Intrinsics:
    # The clips carry no EXIF focal length, so this is Stage 0's prior.
    return Intrinsics(width=2160, height=1214, fx=1836.0, fy=1836.0, cx=1080.0, cy=607.0)


@pytest.fixture(scope="module")
def results(estimator, intrinsics) -> dict:
    out = {}
    for name in CONTROL + CONTACT:
        path = FRAMES_DIR / f"{name}.jpg"
        if not path.exists():
            continue
        out[name] = estimator.process(cv2.imread(str(path)), intrinsics)
    return out


def test_detects_every_verification_frame(results):
    """The headline check. HaMeR's replacement has to find the hand every time."""
    missed = [name for name, frame in results.items() if not frame.detected]
    assert not missed, f"WiLoR missed {len(missed)} of {len(results)} frames: {missed}"


def test_holds_the_hand_through_a_wrapped_grasp(results):
    """The reason these frames were chosen.

    A detector that flattens the hand open when the fingers disappear behind
    the glass is useless here: the grasp is exactly the moment that matters.
    """
    for name in CONTACT:
        frame = results.get(name)
        if frame is None:
            continue
        assert frame.detected, f"lost the hand on the wrapped grasp {name}"


def test_hand_size_is_anatomically_plausible(results):
    for name, frame in results.items():
        if not frame.detected:
            continue
        length = np.linalg.norm(frame.landmarks_cam[12] - frame.landmarks_cam[0])
        assert 0.10 < length < 0.25, f"{name}: hand is {length * 100:.1f} cm wrist to fingertip"


def test_hand_sits_at_arm_length_from_a_head_mounted_camera(results):
    for name, frame in results.items():
        if not frame.detected:
            continue
        depth = float(frame.landmarks_cam[:, 2].mean())
        assert 0.15 < depth < 1.2, f"{name}: hand is {depth:.2f} m from the camera"


def test_landmarks_project_near_the_frame(results, intrinsics):
    """Most landmarks land in frame, and none land absurdly far outside.

    Not all of them: in the C006 pre-grasp the hand genuinely runs off the
    left edge, so five of twenty-one landmarks are legitimately outside.
    Verified by overlaying them on the frame. The check is that the estimate
    is anchored to the image, not that the hand is fully contained.
    """
    for name, frame in results.items():
        if not frame.detected:
            continue
        pixels = frame.landmarks_px
        inside = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < intrinsics.width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < intrinsics.height)
        ).mean()
        assert inside > 0.5, f"{name}: only {inside * 100:.0f}% of landmarks land in frame"

        # Nothing should sit a whole frame away, which is what a bad camera
        # translation or a units mix-up would produce.
        assert pixels[:, 0].min() > -intrinsics.width, f"{name}: landmarks far off-image"
        assert pixels[:, 0].max() < 2 * intrinsics.width, f"{name}: landmarks far off-image"


def test_grasp_landmarks_are_visible_during_contact(results, intrinsics):
    """Stage 4 reads the thumb and index tips, so those two have to be in frame."""
    for name in CONTACT:
        frame = results.get(name)
        if frame is None or not frame.detected:
            continue
        for label, index in (("thumb tip", 4), ("index tip", 8)):
            x, y = frame.landmarks_px[index]
            assert 0 <= x < intrinsics.width and 0 <= y < intrinsics.height, (
                f"{name}: {label} projects outside the frame at ({x:.0f}, {y:.0f})"
            )


def test_grasp_width_separates_open_from_wrapped(results):
    """The signal Stage 4 keys on, measured rather than assumed.

    On these clips an open hand reads about 12 cm across and a hand wrapped
    on the glass about 6 cm. That gap is what makes grasp detection possible,
    and it is why the thresholds are derived per episode: a fixed 4.5 cm sits
    below the wrapped width and never fires.
    """
    open_widths = [
        hand_backend.grasp_width(results[n].landmarks_cam)
        for n in CONTROL
        if n in results and results[n].detected
    ]
    wrapped_widths = [
        hand_backend.grasp_width(results[n].landmarks_cam)
        for n in CONTACT
        if n in results and results[n].detected
    ]
    assert open_widths and wrapped_widths

    assert np.median(open_widths) > np.median(wrapped_widths), (
        f"open {np.median(open_widths) * 100:.1f} cm is not wider than "
        f"wrapped {np.median(wrapped_widths) * 100:.1f} cm"
    )
    # And the gap has to be big enough for a threshold to sit inside it.
    assert np.median(open_widths) - np.median(wrapped_widths) > 0.02
