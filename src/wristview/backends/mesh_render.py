"""Z-buffered rasterizer for the gripper and the object.

Stage 5 composites three things into one wrist view: the splat, which is the
scene; the gripper, which is a handful of boxes; and the object, which is the
point model Stage 3 fitted. The splat renderer returns depth, so solid
geometry composites against it with an ordinary depth test.

Small and self-contained on purpose. A full mesh pipeline would be a
dependency, a build step, and another thing that does not install on Apple
Silicon, for perhaps forty triangles.
"""

from __future__ import annotations

import numpy as np

from ..logging_setup import get

log = get(__name__)

# Cube corners and the two triangles per face, in the order `box_mesh` uses.
_CUBE_CORNERS = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=np.float64,
)
_CUBE_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2],   # -z
        [4, 5, 6], [4, 6, 7],   # +z
        [0, 1, 5], [0, 5, 4],   # -y
        [3, 7, 6], [3, 6, 2],   # +y
        [0, 4, 7], [0, 7, 3],   # -x
        [1, 2, 6], [1, 6, 5],   # +x
    ],
    dtype=np.int64,
)


def box_mesh(center: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles for an axis-aligned box."""
    return _CUBE_CORNERS * half[None, :] + center[None, :], _CUBE_FACES.copy()


def combine(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate meshes, offsetting each one's face indices."""
    vertices, faces, offset = [], [], 0
    for verts, tris in meshes:
        vertices.append(verts)
        faces.append(tris + offset)
        offset += len(verts)
    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(vertices), np.concatenate(faces)


def rasterize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    view_matrix: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    color: tuple[float, float, float],
    near: float = 0.01,
    light_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a triangle mesh. Returns RGB and a depth buffer.

    Depth is infinite where nothing was drawn. Shading is a single headlight
    term, which is right for a wrist camera: the light is at the camera.
    """
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)
    color_buffer = np.zeros((height, width, 3), dtype=np.float64)
    if len(faces) == 0:
        return color_buffer, depth_buffer

    rotation = view_matrix[:3, :3]
    translation = view_matrix[:3, 3]
    cam = vertices @ rotation.T + translation

    depths = cam[:, 2]
    safe = np.where(np.abs(depths) < 1e-9, 1e-9, depths)
    pixels = np.stack([fx * cam[:, 0] / safe + cx, fy * cam[:, 1] / safe + cy], axis=1)

    if light_direction is None:
        light_direction = np.array([0.0, 0.0, -1.0])

    base = np.asarray(color, dtype=np.float64)

    for face in faces:
        tri_depth = depths[face]
        # Clip anything crossing the near plane rather than projecting it.
        if np.any(tri_depth <= near):
            continue
        tri = pixels[face]

        x0 = max(int(np.floor(tri[:, 0].min())), 0)
        x1 = min(int(np.ceil(tri[:, 0].max())) + 1, width)
        y0 = max(int(np.floor(tri[:, 1].min())), 0)
        y1 = min(int(np.ceil(tri[:, 1].max())) + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue

        area = (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (
            tri[2, 0] - tri[0, 0]
        ) * (tri[1, 1] - tri[0, 1])
        if abs(area) < 1e-9:
            continue

        ys, xs = np.mgrid[y0:y1, x0:x1]
        px = xs + 0.5
        py = ys + 0.5

        w0 = ((tri[1, 0] - px) * (tri[2, 1] - py) - (tri[2, 0] - px) * (tri[1, 1] - py)) / area
        w1 = ((tri[2, 0] - px) * (tri[0, 1] - py) - (tri[0, 0] - px) * (tri[2, 1] - py)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        # Perspective-correct depth: interpolate 1/z, not z.
        inverse = w0 / tri_depth[0] + w1 / tri_depth[1] + w2 / tri_depth[2]
        inverse = np.where(inverse <= 1e-9, np.inf, inverse)
        pixel_depth = 1.0 / inverse

        window_depth = depth_buffer[y0:y1, x0:x1]
        nearer = inside & (pixel_depth < window_depth)
        if not nearer.any():
            continue

        edge_a = vertices[face[1]] - vertices[face[0]]
        edge_b = vertices[face[2]] - vertices[face[0]]
        normal = np.cross(edge_a, edge_b)
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        normal_cam = rotation @ normal
        shade = 0.35 + 0.65 * abs(float(np.dot(normal_cam, light_direction)))

        window_depth[nearer] = pixel_depth[nearer]
        color_buffer[y0:y1, x0:x1][nearer] = base * shade

    return color_buffer, depth_buffer


def rasterize_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    view_matrix: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    point_radius_m: float = 0.004,
    near: float = 0.01,
    default_color: tuple[float, float, float] = (0.85, 0.25, 0.2),
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a point cloud as depth-tested discs.

    The object model from Stage 3 is a point cloud, not a mesh. Drawing each
    point as a disc whose pixel radius follows its depth gives a solid-looking
    surface without meshing anything.
    """
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)
    color_buffer = np.zeros((height, width, 3), dtype=np.float64)
    if len(points) == 0:
        return color_buffer, depth_buffer

    rotation = view_matrix[:3, :3]
    translation = view_matrix[:3, 3]
    cam = points @ rotation.T + translation
    depths = cam[:, 2]

    visible = depths > near
    if not visible.any():
        return color_buffer, depth_buffer
    cam, depths = cam[visible], depths[visible]

    px = fx * cam[:, 0] / depths + cx
    py = fy * cam[:, 1] / depths + cy
    radii = np.maximum(1.0, fx * point_radius_m / depths)

    if colors is None:
        point_colors = np.tile(np.asarray(default_color), (len(cam), 1))
    else:
        point_colors = colors[visible]

    # Draw far to near, so the nearest point wins without a per-pixel test.
    order = np.argsort(-depths)
    for index in order:
        radius = int(np.ceil(radii[index]))
        cx_i, cy_i = int(round(px[index])), int(round(py[index]))
        x0, x1 = max(cx_i - radius, 0), min(cx_i + radius + 1, width)
        y0, y1 = max(cy_i - radius, 0), min(cy_i + radius + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue

        ys, xs = np.mgrid[y0:y1, x0:x1]
        disc = (xs - px[index]) ** 2 + (ys - py[index]) ** 2 <= radii[index] ** 2
        if not disc.any():
            continue
        window = depth_buffer[y0:y1, x0:x1]
        nearer = disc & (depths[index] < window)
        window[nearer] = depths[index]
        color_buffer[y0:y1, x0:x1][nearer] = point_colors[index]

    return color_buffer, depth_buffer


def composite(
    layers: list[tuple[np.ndarray, np.ndarray]], background: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Depth-order any number of (color, depth) layers into one image."""
    if not layers:
        height, width = 1, 1
        return np.zeros((height, width, 3)), np.full((height, width), np.inf)

    height, width = layers[0][1].shape
    out_color = np.tile(np.asarray(background, dtype=np.float64), (height, width, 1))
    out_depth = np.full((height, width), np.inf)

    for color, depth in layers:
        nearer = depth < out_depth
        out_depth[nearer] = depth[nearer]
        out_color[nearer] = color[nearer]
    return out_color, out_depth
