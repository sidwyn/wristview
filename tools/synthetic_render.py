"""Ray-traced renderer for the synthetic fixture.

Two primitives, both analytic, both fully vectorized over pixels:

  textured quads   for the room, the table, and the object
  capsules         for the hand

A ray tracer rather than a rasterizer, because analytic intersection gives
exact silhouettes and exact depth. That matters: the fixture's whole job is to
be ground truth, and a ground truth with rasterization artifacts in it is not
worth much.

Runs on MPS through torch.
"""

from __future__ import annotations

import numpy as np
import torch

from .synthetic_scene import Quad


def _t(array, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), device=device, dtype=dtype)


def camera_rays(
    pose: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    width: int, height: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray origin and per-pixel direction in world coordinates.

    `pose` is camera-to-world. Camera axes are OpenCV: x right, y down,
    z forward.
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dirs_cam = torch.stack(
        [(xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy, torch.ones_like(xs)], dim=-1
    )
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

    rotation = _t(pose[:3, :3], device)
    origin = _t(pose[:3, 3], device)
    dirs_world = dirs_cam.reshape(-1, 3) @ rotation.T
    return origin, dirs_world


def intersect_quads(
    origin: torch.Tensor, dirs: torch.Tensor, quads: list[Quad], device: str,
    far: float = 1e6,
):
    """Nearest quad hit per ray.

    Returns depth, albedo, normal, and a hit mask. One vectorized pass per
    quad, which is cheap because a room is a few dozen quads.
    """
    num = dirs.shape[0]
    best_t = torch.full((num,), far, device=device)
    albedo = torch.zeros(num, 3, device=device)
    normal = torch.zeros(num, 3, device=device)

    for quad in quads:
        edge_u = _t(quad.edge_u, device)
        edge_v = _t(quad.edge_v, device)
        quad_origin = _t(quad.origin, device)

        face_normal = torch.linalg.cross(edge_u, edge_v)
        face_normal = face_normal / face_normal.norm().clamp_min(1e-12)

        denominator = dirs @ face_normal
        # Rays parallel to the plane never hit it.
        parallel = denominator.abs() < 1e-9
        t_hit = ((quad_origin - origin) @ face_normal) / torch.where(
            parallel, torch.ones_like(denominator), denominator
        )

        valid = (~parallel) & (t_hit > 1e-4) & (t_hit < best_t)
        if not bool(valid.any()):
            continue

        point = origin[None, :] + t_hit[:, None] * dirs
        local = point - quad_origin[None, :]
        alpha = (local @ edge_u) / edge_u.dot(edge_u)
        beta = (local @ edge_v) / edge_v.dot(edge_v)
        valid = valid & (alpha >= 0) & (alpha <= 1) & (beta >= 0) & (beta <= 1)
        if not bool(valid.any()):
            continue

        texture = _t(quad.texture, device)
        size_v, size_u = texture.shape[0], texture.shape[1]
        tile_u, tile_v = quad.tiling
        # `% 1.0` wraps the texture, so tiling repeats instead of clamping.
        tex_u = ((alpha * tile_u) % 1.0 * (size_u - 1)).long().clamp(0, size_u - 1)
        tex_v = ((beta * tile_v) % 1.0 * (size_v - 1)).long().clamp(0, size_v - 1)
        color = texture[tex_v, tex_u]

        best_t = torch.where(valid, t_hit, best_t)
        albedo = torch.where(valid[:, None], color, albedo)
        # Face the normal back toward the ray, so shading works from both sides.
        oriented = torch.where(denominator[:, None] < 0, face_normal[None, :], -face_normal[None, :])
        normal = torch.where(valid[:, None], oriented, normal)

    return best_t, albedo, normal, best_t < far


def intersect_capsules(
    origin: torch.Tensor,
    dirs: torch.Tensor,
    segments: list[tuple[np.ndarray, np.ndarray, float]],
    device: str,
    far: float = 1e6,
):
    """Nearest capsule hit per ray.

    Standard analytic capsule intersection: solve the infinite cylinder, then
    fall back to the end sphere when the hit lands past either cap.
    """
    num = dirs.shape[0]
    best_t = torch.full((num,), far, device=device)
    normal = torch.zeros(num, 3, device=device)

    for start_np, end_np, radius in segments:
        seg_a = _t(start_np, device)
        seg_b = _t(end_np, device)
        axis = seg_b - seg_a
        axis_dot = float(axis.dot(axis))
        if axis_dot < 1e-12:
            continue

        to_start = origin[None, :] - seg_a[None, :]
        axis_dir = dirs @ axis
        axis_start = to_start @ axis
        dir_start = (dirs * to_start).sum(dim=-1)
        start_start = (to_start * to_start).sum(dim=-1)

        quad_a = axis_dot - axis_dir * axis_dir
        quad_b = axis_dot * dir_start - axis_start * axis_dir
        quad_c = axis_dot * start_start - axis_start * axis_start - radius * radius * axis_dot
        disc = quad_b * quad_b - quad_a * quad_c

        t_hit = torch.full((num,), far, device=device)
        hit_normal = torch.zeros(num, 3, device=device)

        body = disc > 0
        if bool(body.any()):
            root = torch.sqrt(disc.clamp_min(0))
            t_body = (-quad_b - root) / torch.where(quad_a.abs() < 1e-12, torch.ones_like(quad_a), quad_a)
            along = axis_start + t_body * axis_dir
            on_body = body & (along > 0) & (along < axis_dot) & (t_body > 1e-4)
            point = origin[None, :] + t_body[:, None] * dirs
            surface = point - seg_a[None, :] - axis[None, :] * (along / axis_dot)[:, None]
            t_hit = torch.where(on_body, t_body, t_hit)
            hit_normal = torch.where(on_body[:, None], surface / radius, hit_normal)

        # End caps. Pick whichever sphere the ray passes nearest.
        for cap_center, is_start in ((seg_a, True), (seg_b, False)):
            to_center = origin[None, :] - cap_center[None, :]
            sphere_b = (dirs * to_center).sum(dim=-1)
            sphere_c = (to_center * to_center).sum(dim=-1) - radius * radius
            sphere_disc = sphere_b * sphere_b - sphere_c
            valid = sphere_disc > 0
            if not bool(valid.any()):
                continue
            t_cap = -sphere_b - torch.sqrt(sphere_disc.clamp_min(0))
            valid = valid & (t_cap > 1e-4) & (t_cap < t_hit)
            if not bool(valid.any()):
                continue
            point = origin[None, :] + t_cap[:, None] * dirs
            t_hit = torch.where(valid, t_cap, t_hit)
            hit_normal = torch.where(
                valid[:, None], (point - cap_center[None, :]) / radius, hit_normal
            )
            del is_start

        closer = t_hit < best_t
        best_t = torch.where(closer, t_hit, best_t)
        normal = torch.where(closer[:, None], hit_normal, normal)

    return best_t, normal, best_t < far


LIGHTS = [
    (np.array([1.6, -1.4, 2.35]), 0.55),
    (np.array([-1.7, 1.5, 2.35]), 0.35),
    (np.array([0.0, -1.9, 1.9]), 0.25),
]
AMBIENT = 0.38
SKIN_RGB = (0.82, 0.63, 0.52)


def shade(
    albedo: torch.Tensor, normal: torch.Tensor, points: torch.Tensor, device: str
) -> torch.Tensor:
    """Lambert shading from a few point lights, plus ambient."""
    intensity = torch.full(albedo.shape[:1], AMBIENT, device=device)
    for position, power in LIGHTS:
        light_dir = _t(position, device)[None, :] - points
        distance = light_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        light_dir = light_dir / distance
        lambert = (normal * light_dir).sum(dim=-1).clamp_min(0.0)
        # Mild falloff. Full inverse-square makes an indoor scene too contrasty.
        intensity = intensity + power * lambert / (1.0 + 0.12 * distance[:, 0] ** 2)
    return (albedo * intensity[:, None]).clamp(0.0, 1.0)


def render_frame(
    pose: np.ndarray,
    quads: list[Quad],
    capsules: list[tuple[np.ndarray, np.ndarray, float]],
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    device: str,
    noise_sigma: float = 0.006,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one frame. Returns an RGB uint8 image and a float32 depth map."""
    origin, dirs = camera_rays(pose, fx, fy, cx, cy, width, height, device)

    quad_t, quad_albedo, quad_normal, quad_hit = intersect_quads(origin, dirs, quads, device)
    color = torch.zeros_like(quad_albedo)
    normal = torch.zeros_like(quad_normal)
    depth = quad_t.clone()
    albedo = quad_albedo

    if capsules:
        cap_t, cap_normal, cap_hit = intersect_capsules(origin, dirs, capsules, device)
        nearer = cap_hit & (cap_t < depth)
        depth = torch.where(nearer, cap_t, depth)
        normal = torch.where(nearer[:, None], cap_normal, quad_normal)
        skin = _t(np.array(SKIN_RGB), device)[None, :].expand_as(albedo)
        albedo = torch.where(nearer[:, None], skin, quad_albedo)
        hit = quad_hit | nearer
    else:
        normal = quad_normal
        hit = quad_hit

    points = origin[None, :] + depth[:, None] * dirs
    color = shade(albedo, normal, points, device)
    # Rays that hit nothing see the void. In a closed room this is rare.
    color = torch.where(hit[:, None], color, torch.zeros_like(color))

    image = color.reshape(height, width, 3).cpu().numpy()
    depth_map = torch.where(hit, depth, torch.zeros_like(depth)).reshape(height, width).cpu().numpy()

    if noise_sigma > 0:
        generator = rng or np.random.default_rng()
        # Sensor noise. Without it the footage is unnaturally clean and the
        # blur and quality checks in Stage 0 have nothing to measure.
        image = image + generator.normal(0, noise_sigma, image.shape)

    return (np.clip(image, 0, 1) * 255).astype(np.uint8), depth_map.astype(np.float32)


def transform_quads(quads: list[Quad], pose: np.ndarray) -> list[Quad]:
    """Rigidly move a set of quads. Used to place the object each frame."""
    rotation, translation = pose[:3, :3], pose[:3, 3]
    return [
        Quad(
            origin=rotation @ quad.origin + translation,
            edge_u=rotation @ quad.edge_u,
            edge_v=rotation @ quad.edge_v,
            texture=quad.texture,
            tiling=quad.tiling,
        )
        for quad in quads
    ]
