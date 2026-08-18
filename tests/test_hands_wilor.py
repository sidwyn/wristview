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
from wristview.reprojection import MAX_HAND_REPROJECTION_PX


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


@pytest.fixture(scope="module")
def calibrated_results(estimator) -> dict:
    """The same frames, lifted with a focal length that is not a guess.

    The module fixture uses fx=1836, which is Stage 0's prior for this capture.
    Stage 1 refined the same camera to 2819, a factor of 1.5, and s03_estimate
    records that lifting the hand with the prior put it about 20 cm below the
    desk. Depth scales with focal length, so a reprojection check fed the prior
    is measuring the prior.

    Worth stating plainly: at 2819 these stills reproject to about 22 px, and
    the figure keeps falling as the focal is raised further, to about 15 px at
    4000. So a3 has no calibrated ground truth and this is a coarse check here.
    It is a tight one on real06, where the camera is refined and the same code
    measures 13 to 16 px.
    """
    out = {}
    intrinsics = Intrinsics(
        width=2160, height=1214, fx=2819.0, fy=2819.0, cx=1080.0, cy=607.0
    )
    for name in CONTROL + CONTACT:
        path = FRAMES_DIR / f"{name}.jpg"
        if path.exists():
            out[name] = (estimator.process(cv2.imread(str(path)), intrinsics), intrinsics)
    return out


def test_the_3d_hand_reprojects_onto_its_own_2d_detection(calibrated_results):
    """The check that was missing while grasp silently failed on every session.

    `test_landmarks_project_near_the_frame` above reads `landmarks_px`, which
    comes straight from WiLoR and was always right. Nothing compared it against
    `landmarks_cam`, the 3D estimate everything downstream actually uses, so a
    3D hand sitting metres from where the 2D hand plainly was raised no failure
    anywhere.

    Until 17 August the translation was rescaled onto our intrinsics by scaling
    all three components. Uniform scaling leaves X/Z alone while the focal
    changes, and the projected offset is `f * X/Z`, so the hand collapsed
    toward the optical axis by the focal ratio, about 25x on session 6. The
    wrist swept 2221 px across the image while its 3D position accounted for
    94, reprojection missed by a median of 614 px, and the whole hand lived in
    a 2.6 x 1.8 x 16.9 cm box. Grasp detection had never worked on any session
    and this is why.

    Reprojection is the cheapest possible statement of the thing that matters:
    the 3D hand has to be where the 2D hand is.
    """
    worst = []
    for name, (frame, intrinsics) in calibrated_results.items():
        if not frame.detected:
            continue
        cam = frame.landmarks_cam
        ahead = cam[:, 2] > 1e-6
        assert ahead.all(), f"{name}: {int((~ahead).sum())} landmarks behind the camera"

        u = intrinsics.fx * cam[:, 0] / cam[:, 2] + intrinsics.cx
        v = intrinsics.fy * cam[:, 1] / cam[:, 2] + intrinsics.cy
        error = np.hypot(u - frame.landmarks_px[:, 0], v - frame.landmarks_px[:, 1])
        worst.append((name, float(np.median(error))))

    assert worst, "no frames were estimated, so nothing was checked"

    # Compared the same way the production gate compares it, on the median
    # across frames rather than frame by frame. Two of these stills are
    # pre-grasp frames where the hand runs off the left edge, which the test
    # above already documents as legitimate; a partial hand gets a poor crop
    # and reprojects worse, and holding every individual frame to the gate's
    # bound would be testing the fixture's framing rather than the lift.
    values = np.array([value for _, value in worst])
    median = float(np.median(values))
    assert median <= MAX_HAND_REPROJECTION_PX, (
        f"median reprojection {median:.0f} px exceeds "
        f"{MAX_HAND_REPROJECTION_PX} px: "
        + ", ".join(f"{name} off by {value:.0f} px" for name, value in worst)
    )

    # And nothing may be catastrophically wrong even on the worst frame. The
    # defect this test exists for measured 614 px; the worst uncalibrated
    # partial-hand frame here measures under 100.
    name, value = max(worst, key=lambda pair: pair[1])
    assert value < 150.0, f"{name} reprojects {value:.0f} px away"
