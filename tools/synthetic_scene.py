"""A synthetic room, hand, and task, rendered to video.

Why this exists. The pipeline needs a scan clip and demo clips to run. This
machine has neither, and the pipeline cannot be shown to work on footage that
does not exist. So this module builds a room with strong texture, animates a
hand doing a pick-and-place, and renders both a scan orbit and demo clips.

It also writes ground truth: the ARKit-style camera trajectory in metres, the
true hand landmarks, and the true object pose. That turns "the pipeline ran"
into "the pipeline recovered the right answer to within X".

Two renderers, composited by depth:
  a textured triangle rasterizer for the room, which gives the crisp local
  texture COLMAP needs to find features, and
  an analytic capsule ray tracer for the hand, which gives smooth limbs
  without a mesh.

Everything runs on torch, so it uses the same MPS device as the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# MediaPipe hand landmark order. Fixed by that model, reused here so the
# ground truth and the estimator speak the same language.
LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Bones as index pairs, used to draw the hand and nothing else.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

BONE_RADII = {
    "palm": 0.011,
    "proximal": 0.009,
    "middle": 0.0075,
    "distal": 0.0065,
}


# --------------------------------------------------------------------------
# Procedural texture
# --------------------------------------------------------------------------

def _value_noise(rng: np.random.Generator, height: int, width: int, cells: int) -> np.ndarray:
    """Smooth random field by bilinear upsampling of a coarse grid."""
    coarse = rng.random((cells + 1, cells + 1))
    ys = np.linspace(0, cells, height)
    xs = np.linspace(0, cells, width)
    y0 = np.floor(ys).astype(int).clip(0, cells - 1)
    x0 = np.floor(xs).astype(int).clip(0, cells - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    # Smoothstep, so the field has no visible grid creases.
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    top = coarse[y0][:, x0] * (1 - fx) + coarse[y0][:, x0 + 1] * fx
    bottom = coarse[y0 + 1][:, x0] * (1 - fx) + coarse[y0 + 1][:, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def make_texture(
    rng: np.random.Generator,
    size: int,
    base_rgb: tuple[float, float, float],
    marks: int = 40,
    speckle: float = 0.05,
) -> np.ndarray:
    """Build a texture with the kind of detail a feature detector can lock on.

    Three layers: a soft colour field, hard-edged marks that give corners, and
    fine speckle that gives gradient everywhere. Flat paint defeats COLMAP,
    which is exactly the real-world failure this fixture must not hide.
    """
    field = _value_noise(rng, size, size, 6)[..., None]
    image = np.array(base_rgb)[None, None, :] * (0.72 + 0.5 * field)

    for _ in range(marks):
        shape = rng.integers(0, 3)
        color = rng.random(3) * 0.85 + 0.05
        cx, cy = rng.integers(0, size, 2)
        radius = int(rng.integers(size // 40, size // 9))
        ys, xs = np.mgrid[0:size, 0:size]

        if shape == 0:
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 < radius**2
        elif shape == 1:
            mask = (np.abs(xs - cx) < radius) & (np.abs(ys - cy) < radius * rng.uniform(0.3, 1.0))
        else:
            # A thin diagonal stripe. Strong oriented gradient, which is what
            # SuperPoint and SIFT both key on.
            angle = rng.uniform(0, np.pi)
            dist = np.abs((xs - cx) * np.sin(angle) - (ys - cy) * np.cos(angle))
            extent = np.abs((xs - cx) * np.cos(angle) + (ys - cy) * np.sin(angle))
            mask = (dist < radius * 0.18) & (extent < radius * 1.8)
        image[mask] = color

    image = image + rng.normal(0, speckle, image.shape)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Scene geometry
# --------------------------------------------------------------------------

@dataclass
class Quad:
    """A textured rectangle, given by a corner and two edge vectors."""

    origin: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    texture: np.ndarray
    # How many times the texture repeats along each edge.
    tiling: tuple[float, float] = (1.0, 1.0)


@dataclass
class Scene:
    quads: list[Quad] = field(default_factory=list)
    # Object pose is tracked separately, because it moves and the room does not.
    object_quads: list[Quad] = field(default_factory=list)
    object_center_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    table_height: float = 0.75


def _box_quads(
    center: np.ndarray,
    half: np.ndarray,
    textures: list[np.ndarray],
    tiling: tuple[float, float] = (1.0, 1.0),
) -> list[Quad]:
    """Six textured faces of an axis-aligned box."""
    cx, cy, cz = center
    hx, hy, hz = half
    faces = [
        # (origin, edge_u, edge_v)
        ([cx - hx, cy - hy, cz + hz], [2 * hx, 0, 0], [0, 2 * hy, 0]),   # top
        ([cx - hx, cy - hy, cz - hz], [2 * hx, 0, 0], [0, 2 * hy, 0]),   # bottom
        ([cx - hx, cy - hy, cz - hz], [2 * hx, 0, 0], [0, 0, 2 * hz]),   # -y
        ([cx - hx, cy + hy, cz - hz], [2 * hx, 0, 0], [0, 0, 2 * hz]),   # +y
        ([cx - hx, cy - hy, cz - hz], [0, 2 * hy, 0], [0, 0, 2 * hz]),   # -x
        ([cx + hx, cy - hy, cz - hz], [0, 2 * hy, 0], [0, 0, 2 * hz]),   # +x
    ]
    return [
        Quad(np.array(o, dtype=np.float64), np.array(u, dtype=np.float64),
             np.array(v, dtype=np.float64), textures[i % len(textures)], tiling)
        for i, (o, u, v) in enumerate(faces)
    ]


def build_room(seed: int = 7, texture_size: int = 512) -> Scene:
    """A 4 by 4 metre room with a table and a graspable block on it.

    World frame: z is up, the floor is z equals 0, the table top is at 0.75 m.
    Those are real metres, which is what makes the metric-scale check meaningful.
    """
    rng = np.random.default_rng(seed)
    scene = Scene()

    floor_tex = make_texture(rng, texture_size, (0.45, 0.38, 0.32), marks=55)
    ceiling_tex = make_texture(rng, texture_size, (0.82, 0.82, 0.80), marks=18, speckle=0.03)
    wall_texes = [make_texture(rng, texture_size, (0.62, 0.64, 0.68), marks=48) for _ in range(4)]

    half = 2.0
    height = 2.6

    scene.quads.append(
        Quad(np.array([-half, -half, 0.0]), np.array([2 * half, 0, 0]),
             np.array([0, 2 * half, 0]), floor_tex, (3.0, 3.0))
    )
    scene.quads.append(
        Quad(np.array([-half, -half, height]), np.array([2 * half, 0, 0]),
             np.array([0, 2 * half, 0]), ceiling_tex, (2.0, 2.0))
    )
    walls = [
        ([-half, -half, 0], [2 * half, 0, 0], [0, 0, height]),
        ([-half, half, 0], [2 * half, 0, 0], [0, 0, height]),
        ([-half, -half, 0], [0, 2 * half, 0], [0, 0, height]),
        ([half, -half, 0], [0, 2 * half, 0], [0, 0, height]),
    ]
    for i, (o, u, v) in enumerate(walls):
        scene.quads.append(
            Quad(np.array(o, dtype=np.float64), np.array(u, dtype=np.float64),
                 np.array(v, dtype=np.float64), wall_texes[i], (2.5, 1.6))
        )

    # Table: a 1.2 by 0.8 metre top at 0.75 m, on four legs.
    table_tex = [make_texture(rng, texture_size, (0.55, 0.42, 0.30), marks=42) for _ in range(3)]
    scene.quads += _box_quads(
        np.array([0.0, 0.0, 0.735]), np.array([0.6, 0.4, 0.015]), table_tex, (2.0, 1.5)
    )
    leg_tex = [make_texture(rng, texture_size, (0.35, 0.28, 0.22), marks=14)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            scene.quads += _box_quads(
                np.array([sx * 0.55, sy * 0.35, 0.36]),
                np.array([0.025, 0.025, 0.36]),
                leg_tex,
            )

    # Clutter. Real rooms have it, and it gives the reconstruction parallax.
    for _ in range(6):
        pos = np.array([rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6), 0.0])
        if abs(pos[0]) < 0.8 and abs(pos[1]) < 0.6:
            continue
        size = rng.uniform(0.08, 0.22)
        tex = [make_texture(rng, 256, tuple(rng.random(3) * 0.6 + 0.2), marks=12)]
        scene.quads += _box_quads(
            np.array([pos[0], pos[1], size]), np.array([size, size, size]), tex
        )

    # The graspable object: an 8 cm block, sized for a parallel-jaw gripper.
    obj_tex = [make_texture(rng, 256, (0.80, 0.22, 0.18), marks=16, speckle=0.03)]
    scene.object_quads = _box_quads(
        np.array([0.0, 0.0, 0.0]), np.array([0.035, 0.035, 0.045]), obj_tex
    )
    scene.object_center_local = np.zeros(3)
    return scene


# --------------------------------------------------------------------------
# Hand model and animation
# --------------------------------------------------------------------------

def canonical_hand() -> np.ndarray:
    """A right hand at rest, in a local frame, in metres.

    Local frame: origin at the wrist, +y toward the fingertips, +x toward the
    thumb, +z out of the back of the hand. Bone lengths are adult-average.
    """
    hand = np.zeros((21, 3))
    hand[0] = [0.0, 0.0, 0.0]

    # Thumb, splayed away from the palm.
    hand[1] = [0.026, 0.022, 0.008]
    hand[2] = [0.048, 0.055, 0.014]
    hand[3] = [0.058, 0.086, 0.018]
    hand[4] = [0.064, 0.111, 0.020]

    # Four fingers. Knuckle row across the palm, then three phalanges.
    knuckles = {
        "index": (0.022, 0.095),
        "middle": (0.000, 0.099),
        "ring": (-0.021, 0.095),
        "pinky": (-0.041, 0.088),
    }
    lengths = {
        "index": (0.040, 0.025, 0.021),
        "middle": (0.045, 0.028, 0.022),
        "ring": (0.042, 0.026, 0.021),
        "pinky": (0.033, 0.020, 0.018),
    }
    for offset, name in enumerate(["index", "middle", "ring", "pinky"]):
        base = 5 + offset * 4
        kx, ky = knuckles[name]
        hand[base] = [kx, ky, 0.0]
        cursor = np.array([kx, ky, 0.0])
        for j, length in enumerate(lengths[name]):
            cursor = cursor + np.array([kx * 0.06, length, 0.0])
            hand[base + 1 + j] = cursor
    return hand


def curl_hand(hand: np.ndarray, amount: float) -> np.ndarray:
    """Curl the fingers and close the thumb. `amount` runs 0 open to 1 closed.

    Each joint rotates about the palm's x axis by a share of the total, which
    is a crude but visually correct flexion.
    """
    out = hand.copy()
    amount = float(np.clip(amount, 0.0, 1.0))

    # Fingers fan out sideways as well as curling. Without the splay all four
    # curl into one merged mass and a hand detector sees a blob, not a hand.
    splay = np.deg2rad([14.0, 4.0, -6.0, -16.0])

    for offset in range(4):
        base = 5 + offset * 4
        angles = np.array([0.0, 1.05, 1.35, 0.95]) * amount
        spread = splay[offset]
        origin = out[base].copy()
        cumulative = 0.0
        previous = origin
        for j in range(1, 4):
            cumulative += angles[j]
            vector = hand[base + j] - hand[base + j - 1]
            length = np.linalg.norm(vector)
            direction = np.array(
                [
                    np.sin(spread) * np.cos(cumulative),
                    np.cos(spread) * np.cos(cumulative),
                    -np.sin(cumulative),
                ]
            )
            previous = previous + direction * length
            out[base + j] = previous

    # The thumb swings across the palm to meet the index finger.
    thumb_angle = 0.85 * amount
    rot = np.array(
        [
            [np.cos(thumb_angle), -np.sin(thumb_angle), 0.0],
            [np.sin(thumb_angle), np.cos(thumb_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pivot = hand[1]
    for i in (2, 3, 4):
        out[i] = pivot + rot @ (hand[i] - pivot)
        out[i][2] -= 0.012 * amount
    return out


def hand_to_world(hand_local: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return hand_local @ rotation.T + translation


def bone_radius(a: int, b: int) -> float:
    """Pick a capsule radius from which bone this is."""
    if a == 0 or (a, b) in [(5, 9), (9, 13), (13, 17)]:
        return BONE_RADII["palm"]
    tip_indices = {4, 8, 12, 16, 20}
    if b in tip_indices:
        return BONE_RADII["distal"]
    return BONE_RADII["proximal"] if b % 4 == 2 else BONE_RADII["middle"]
