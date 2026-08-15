"""The synthetic fixture.

Its job is to be ground truth for the geometry stages. It is a test asset, not
a pipeline input: the real runs use the clips in `wristview-videos/`.

A note recorded here because it decided the design. The fixture renders the
hand as smooth capsules, which is out of distribution for detectors trained on
photographs. MediaPipe found it on 17 percent of frames, so the fixture cannot
validate hand estimation. It validates the parts that are pure geometry:
projection, trajectories, rigid transforms, and rendering.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.synthetic_scene import (  # noqa: E402
    HAND_BONES,
    build_room,
    canonical_hand,
    curl_hand,
    hand_to_world,
    make_texture,
)


class TestTexture:
    def test_shape_and_range(self):
        texture = make_texture(np.random.default_rng(0), 64, (0.5, 0.5, 0.5))
        assert texture.shape == (64, 64, 3)
        assert texture.min() >= 0.0
        assert texture.max() <= 1.0

    def test_has_local_contrast(self):
        # Flat paint defeats COLMAP. The fixture must not hide that failure by
        # producing a scene with no features to find.
        texture = make_texture(np.random.default_rng(0), 128, (0.5, 0.5, 0.5))
        gradient = np.abs(np.diff(texture.mean(axis=2), axis=1))
        assert gradient.mean() > 0.005

    def test_is_deterministic(self):
        a = make_texture(np.random.default_rng(3), 32, (0.4, 0.5, 0.6))
        b = make_texture(np.random.default_rng(3), 32, (0.4, 0.5, 0.6))
        assert np.array_equal(a, b)


class TestRoom:
    def test_has_geometry(self):
        scene = build_room(seed=1, texture_size=32)
        assert len(scene.quads) > 20
        assert len(scene.object_quads) == 6

    def test_is_deterministic(self):
        a = build_room(seed=1, texture_size=32)
        b = build_room(seed=1, texture_size=32)
        assert len(a.quads) == len(b.quads)
        assert np.allclose(a.quads[0].origin, b.quads[0].origin)

    def test_table_is_at_the_documented_height(self):
        scene = build_room(seed=1, texture_size=32)
        assert scene.table_height == pytest.approx(0.75)


class TestHandModel:
    def test_landmark_count_matches_mediapipe(self):
        assert canonical_hand().shape == (21, 3)

    def test_bones_reference_valid_landmarks(self):
        for a, b in HAND_BONES:
            assert 0 <= a < 21
            assert 0 <= b < 21

    def test_hand_is_a_plausible_size(self):
        hand = canonical_hand()
        # Wrist to middle fingertip: an adult hand is roughly 17 to 20 cm.
        length = np.linalg.norm(hand[12] - hand[0])
        assert 0.13 < length < 0.22, f"hand is {length * 100:.1f} cm long"

    def test_curling_closes_the_thumb_index_gap(self):
        hand = canonical_hand()
        open_gap = np.linalg.norm(hand[4] - hand[8])
        closed_gap = np.linalg.norm(curl_hand(hand, 0.6)[4] - curl_hand(hand, 0.6)[8])
        assert closed_gap < open_gap

    def test_open_gap_spans_a_gripper_width(self):
        hand = canonical_hand()
        gap = np.linalg.norm(hand[4] - hand[8])
        assert 0.06 < gap < 0.12, f"open thumb-index gap is {gap * 100:.1f} cm"

    def test_curl_preserves_bone_lengths(self):
        # Curling rotates joints; it must not stretch the fingers.
        hand = canonical_hand()
        curled = curl_hand(hand, 0.5)
        for base in (5, 9, 13, 17):
            for joint in range(base + 1, base + 4):
                before = np.linalg.norm(hand[joint] - hand[joint - 1])
                after = np.linalg.norm(curled[joint] - curled[joint - 1])
                assert after == pytest.approx(before, rel=1e-6)

    def test_curl_amount_is_clamped(self):
        hand = canonical_hand()
        assert np.allclose(curl_hand(hand, 5.0), curl_hand(hand, 1.0))
        assert np.allclose(curl_hand(hand, -3.0), curl_hand(hand, 0.0))

    def test_fingers_splay_apart(self):
        # Without splay all four fingers curl into one mass and a detector
        # sees a blob rather than a hand.
        curled = curl_hand(canonical_hand(), 0.5)
        tips = curled[[8, 12, 16, 20]]
        spread = np.linalg.norm(tips.max(axis=0) - tips.min(axis=0))
        assert spread > 0.03, f"fingertips span only {spread * 100:.1f} cm"


class TestHandToWorld:
    def test_identity_transform_is_a_translation(self):
        hand = canonical_hand()
        offset = np.array([1.0, 2.0, 3.0])
        moved = hand_to_world(hand, np.eye(3), offset)
        assert np.allclose(moved, hand + offset)

    def test_rigid_transform_preserves_distances(self):
        from scipy.spatial.transform import Rotation

        hand = canonical_hand()
        rotation = Rotation.from_euler("xyz", [20, -35, 60], degrees=True).as_matrix()
        moved = hand_to_world(hand, rotation, np.array([0.5, -0.2, 1.1]))
        before = np.linalg.norm(hand[4] - hand[8])
        after = np.linalg.norm(moved[4] - moved[8])
        assert after == pytest.approx(before, rel=1e-9)


@pytest.mark.slow
def test_renders_a_frame_end_to_end():
    """The fixture renderer produces a usable image with real geometry."""
    torch = pytest.importorskip("torch")
    from tools.make_fixture import look_at_pose
    from tools.synthetic_render import render_frame

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scene = build_room(seed=1, texture_size=64)
    pose = look_at_pose(np.array([1.4, -1.4, 1.3]), np.array([0.0, 0.0, 0.75]))

    image, depth = render_frame(
        pose, scene.quads, [], fx=400.0, fy=400.0, cx=160.0, cy=120.0,
        width=320, height=240, device=device, noise_sigma=0.0,
    )
    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8
    # A camera inside a closed room must hit a surface on every ray.
    assert (depth > 0).mean() > 0.95
    # And the image must not be a flat colour.
    assert image.std() > 10
