"""The MPS Gaussian splat rasterizer.

This is the component with no upstream to fall back on: gsplat is CUDA-only,
gsplat-mlx does not build, and Brush cannot be called per pose from Python. So
it gets tested on the properties Stages 1 and 5 rely on.
"""

import numpy as np
import pytest
import torch

from wristview.backends.splat_mps import (
    SH_C0,
    GaussianModel,
    eval_sh,
    quat_to_rotmat_torch,
    render,
    sh_bands,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture
def model() -> GaussianModel:
    rng = np.random.default_rng(0)
    points = rng.normal(0, 0.4, (600, 3)).astype(np.float32)
    points[:, 2] += 3.0
    colors = rng.uniform(0.2, 0.9, (600, 3)).astype(np.float32)
    return GaussianModel(
        torch.from_numpy(points), torch.from_numpy(colors), sh_degree=1, device=DEVICE
    )


def identity_view() -> torch.Tensor:
    return torch.eye(4, device=DEVICE)


def test_sh_bands_count():
    assert sh_bands(0) == 1
    assert sh_bands(1) == 4
    assert sh_bands(2) == 9


def test_dc_band_reproduces_the_input_colour():
    # Stage 1 stores colour as the DC spherical harmonic. Degree 0 must give
    # it back exactly, or the splat starts from the wrong colours.
    colors = torch.tensor([[0.2, 0.5, 0.9]])
    sh = torch.zeros(1, 1, 3)
    sh[:, 0] = (colors - 0.5) / SH_C0
    assert torch.allclose(eval_sh(sh, 0, torch.zeros(1, 3)) + 0.5, colors, atol=1e-6)


def test_quat_to_rotmat_is_a_rotation():
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0]])
    matrices = quat_to_rotmat_torch(quats)
    for matrix in matrices:
        assert torch.allclose(matrix @ matrix.T, torch.eye(3), atol=1e-5)
        assert float(torch.det(matrix)) == pytest.approx(1.0, abs=1e-5)


def test_identity_quaternion_gives_identity_rotation():
    matrix = quat_to_rotmat_torch(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))[0]
    assert torch.allclose(matrix, torch.eye(3), atol=1e-6)


class TestRender:
    def test_output_shapes_and_ranges(self, model):
        result = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 320, 240)
        assert result.rgb.shape == (240, 320, 3)
        assert result.alpha.shape == (240, 320)
        assert result.depth.shape == (240, 320)
        assert float(result.rgb.min()) >= 0.0
        assert float(result.alpha.max()) <= 1.0 + 1e-5

    def test_renders_something(self, model):
        result = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 320, 240)
        assert float(result.alpha.mean()) > 0.01

    def test_empty_view_returns_background(self, model):
        # Look the other way: every Gaussian is behind the camera.
        view = torch.eye(4, device=DEVICE)
        view[2, 3] = 10.0
        view[:3, :3] = torch.tensor(
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], device=DEVICE
        )
        background = torch.tensor([0.1, 0.2, 0.3], device=DEVICE)
        far_view = torch.eye(4, device=DEVICE)
        far_view[2, 3] = -100.0
        result = render(
            model, far_view, 250.0, 250.0, 160.0, 120.0, 320, 240, background=background
        )
        assert float(result.alpha.max()) < 1e-4
        assert torch.allclose(result.rgb[0, 0], background, atol=1e-5)
        del view

    def test_depth_is_positive_where_covered(self, model):
        result = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 320, 240)
        covered = result.alpha > 0.5
        if bool(covered.any()):
            assert float(result.depth[covered].min()) > 0.0

    def test_nearer_gaussian_occludes_the_farther_one(self):
        # Two opaque Gaussians on the optical axis. The near one must win.
        points = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0]])
        colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        scales = torch.full((2, 3), 0.15)
        one = GaussianModel(points, colors, scales, sh_degree=0, device=DEVICE)
        with torch.no_grad():
            one.opacity_logit.fill_(6.0)  # effectively opaque

        result = render(one, identity_view(), 300.0, 300.0, 64.0, 64.0, 128, 128)
        centre = result.rgb[64, 64]
        assert float(centre[0]) > float(centre[2]), "the near red Gaussian must occlude the far blue one"

    def test_is_deterministic(self, model):
        a = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 320, 240)
        b = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 320, 240)
        assert torch.allclose(a.rgb, b.rgb, atol=1e-6)

    def test_tile_size_does_not_change_the_image(self, model):
        a = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 320, 240, tile_size=16)
        b = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 320, 240, tile_size=8)
        # Tiling is an implementation detail, not a visual parameter.
        assert float((a.rgb - b.rgb).abs().mean()) < 0.02

    def test_non_multiple_resolution_is_cropped_correctly(self, model):
        result = render(model, identity_view(), 250.0, 250.0, 100.0, 75.0, 201, 149, tile_size=16)
        assert result.rgb.shape == (149, 201, 3)


class TestGradients:
    def test_every_parameter_group_receives_gradient(self, model):
        model.train()
        # Isotropic Gaussians have no rotation gradient by construction, since
        # R S S^T R^T is scale times identity. Make them anisotropic first.
        with torch.no_grad():
            model.log_scales += torch.tensor([0.0, 0.4, -0.4], device=DEVICE)

        result = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 160, 120)
        result.rgb.mean().backward()

        for name in ("means", "log_scales", "quats", "opacity_logit", "sh"):
            grad = getattr(model, name).grad
            assert grad is not None, f"{name} has no gradient"
            assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
            assert float(grad.abs().sum()) > 0, f"{name} gradient is identically zero"

    def test_screen_space_gradient_is_retained_for_densification(self, model):
        model.train()
        result = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 160, 120)
        result.rgb.mean().backward()
        uv = result.__dict__["_uv"]
        assert uv.grad is not None
        assert float(uv.grad.abs().sum()) > 0

    def test_isotropic_gaussians_have_no_rotation_gradient(self, model):
        # Documents why the check above perturbs the scales first.
        model.train()
        result = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 160, 120)
        result.rgb.mean().backward()
        assert float(model.quats.grad.abs().sum()) == pytest.approx(0.0, abs=1e-9)


class TestSaveLoad:
    def test_roundtrip_preserves_parameters(self, model, tmp_path):
        path = model.save(tmp_path / "splat.pt")
        loaded = GaussianModel.load(path, device=DEVICE)
        assert loaded.count == model.count
        assert torch.allclose(loaded.means, model.means, atol=1e-6)
        assert torch.allclose(loaded.sh, model.sh, atol=1e-6)
        assert torch.allclose(loaded.opacity_logit, model.opacity_logit, atol=1e-6)

    def test_roundtrip_renders_the_same_image(self, model, tmp_path):
        before = render(model, identity_view(), 250.0, 250.0, 160.0, 120.0, 160, 120)
        loaded = GaussianModel.load(model.save(tmp_path / "splat.pt"), device=DEVICE)
        after = render(loaded, identity_view(), 250.0, 250.0, 160.0, 120.0, 160, 120)
        assert torch.allclose(before.rgb, after.rgb, atol=1e-5)

    def test_export_ply_is_readable(self, model, tmp_path):
        path = model.export_ply(tmp_path / "points.ply")
        header = path.read_text().split("end_header")[0]
        assert "ply" in header
        assert "element vertex" in header


def test_overfits_a_single_view():
    """The end-to-end property Stage 1 needs: the loss must actually fall."""
    rng = np.random.default_rng(0)
    points = rng.normal(0, 0.5, (800, 3)).astype(np.float32)
    points[:, 2] += 3.0
    colors = rng.uniform(0, 1, (800, 3)).astype(np.float32)
    model = GaussianModel(
        torch.from_numpy(points), torch.from_numpy(colors), sh_degree=0, device=DEVICE
    )
    model.train()

    target = torch.zeros(120, 160, 3, device=DEVICE)
    target[:, :80, 0] = 1.0
    target[:, 80:, 1] = 1.0

    optimizer = torch.optim.Adam(
        [
            {"params": [model.means], "lr": 0.001},
            {"params": [model.sh], "lr": 0.02},
            {"params": [model.opacity_logit], "lr": 0.05},
            {"params": [model.log_scales], "lr": 0.005},
            {"params": [model.quats], "lr": 0.001},
        ],
        eps=1e-15,
    )

    first = None
    for step in range(60):
        result = render(model, identity_view(), 150.0, 150.0, 80.0, 60.0, 160, 120)
        loss = (result.rgb - target).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first = float(loss)

    assert float(loss) < first * 0.85, f"loss did not fall: {first:.4f} to {float(loss):.4f}"


class TestDensification:
    """Adaptive density control has to actually fire.

    It failed silently once: the screen-space gradient was accumulated in
    pixels while `grad_threshold` follows the 3DGS convention and is
    calibrated against normalized device coordinates. The two differ by about
    300x at 720p, so nothing ever crossed the threshold. A room-scale splat
    stayed at its 13k initial points and only shrank through pruning, and the
    only symptom was a splat that looked thin.
    """

    def _underfit_scene(self, count: int = 400):
        rng = np.random.default_rng(0)
        points = rng.normal(0, 0.4, (count, 3)).astype(np.float32)
        points[:, 2] += 3.0
        colors = rng.uniform(0, 1, (count, 3)).astype(np.float32)
        model = GaussianModel(
            torch.from_numpy(points), torch.from_numpy(colors), sh_degree=0, device=DEVICE
        )
        model.train()
        return model

    def test_gradient_is_accumulated_in_ndc_units(self):
        from wristview.backends.splat_mps import Densifier

        model = self._underfit_scene()
        width, height = 320, 240
        target = torch.rand(height, width, 3, device=DEVICE)
        result = render(model, identity_view(), 300.0, 300.0, width / 2, height / 2, width, height)
        (result.rgb - target).abs().mean().backward()

        densifier = Densifier(scene_extent=1.0)
        densifier.accumulate(model, result)

        pixel_grad = result.__dict__["_uv"].grad.norm(dim=-1).mean()
        accumulated = (model.grad_accum.sum() / model.grad_count.clamp_min(1).sum())
        # Roughly half the image diagonal larger than the pixel-space value.
        assert float(accumulated) > float(pixel_grad) * 50

    def test_densification_grows_an_underfit_splat(self):
        from wristview.backends.splat_mps import Densifier

        model = self._underfit_scene()
        start = model.count
        width, height = 320, 240
        target = torch.rand(height, width, 3, device=DEVICE)

        optimizer = torch.optim.Adam(
            [
                {"params": [model.means], "lr": 0.001},
                {"params": [model.sh], "lr": 0.02},
                {"params": [model.opacity_logit], "lr": 0.05},
                {"params": [model.log_scales], "lr": 0.005},
                {"params": [model.quats], "lr": 0.001},
            ],
            eps=1e-15,
        )
        densifier = Densifier(grad_threshold=0.0004, scene_extent=1.0, max_gaussians=100000)

        for _ in range(30):
            result = render(
                model, identity_view(), 300.0, 300.0, width / 2, height / 2, width, height
            )
            (result.rgb - target).abs().mean().backward()
            densifier.accumulate(model, result)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        stats = densifier.step(model, optimizer)
        assert stats["cloned"] + stats["split"] > 0, (
            f"densification never fired: {stats}. The splat cannot grow to fit a scene."
        )
        assert model.count > start

    def test_densification_respects_the_gaussian_cap(self):
        from wristview.backends.splat_mps import Densifier

        model = self._underfit_scene()
        width, height = 320, 240
        target = torch.rand(height, width, 3, device=DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        cap = model.count + 20
        densifier = Densifier(grad_threshold=0.0, scene_extent=1.0, max_gaussians=cap)

        for _ in range(5):
            result = render(
                model, identity_view(), 300.0, 300.0, width / 2, height / 2, width, height
            )
            (result.rgb - target).abs().mean().backward()
            densifier.accumulate(model, result)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        densifier.step(model, optimizer)
        assert model.count <= cap + 1, f"cap {cap} exceeded: {model.count}"


class TestDensityScaling:
    """Rendering must not blow up as the splat grows.

    Compositing every tile slot at once held the whole (pixels, slots) tensor
    for the backward pass. A 51k-Gaussian room splat at 720p exhausted 30 GB
    of MPS memory partway through training, and before that it degraded:
    PSNR fell from 22.7 dB at 13k Gaussians to 17.9 dB at 44k, because slots
    past the cap were dropped. The loop now walks the list in blocks and stops
    once every pixel is saturated.
    """

    def _dense_model(self, count: int) -> GaussianModel:
        rng = np.random.default_rng(0)
        points = rng.normal(0, 0.6, (count, 3)).astype(np.float32)
        points[:, 2] += 3.0
        colors = rng.uniform(0, 1, (count, 3)).astype(np.float32)
        return GaussianModel(
            torch.from_numpy(points), torch.from_numpy(colors), sh_degree=0, device=DEVICE
        )

    def test_renders_a_dense_splat_at_training_resolution(self):
        model = self._dense_model(60_000)
        result = render(
            model, identity_view(), 700.0, 700.0, 360.0, 202.0, 720, 405,
            tile_size=16, max_per_tile=128,
        )
        assert result.rgb.shape == (405, 720, 3)
        assert torch.isfinite(result.rgb).all()
        assert float(result.alpha.mean()) > 0.01

    def test_backward_through_a_dense_splat(self):
        model = self._dense_model(40_000)
        model.train()
        result = render(
            model, identity_view(), 700.0, 700.0, 360.0, 202.0, 720, 405,
            tile_size=16, max_per_tile=128,
        )
        result.rgb.mean().backward()
        assert torch.isfinite(model.means.grad).all()
        assert float(model.means.grad.abs().sum()) > 0

    def test_depth_block_size_does_not_change_the_image(self):
        """Blocking is an implementation detail, so it must be invisible."""
        model = self._dense_model(4_000)
        common = dict(tile_size=16, max_per_tile=128)
        a = render(model, identity_view(), 300.0, 300.0, 160.0, 120.0, 320, 240,
                   depth_block=16, **common)
        b = render(model, identity_view(), 300.0, 300.0, 160.0, 120.0, 320, 240,
                   depth_block=64, **common)
        assert torch.allclose(a.rgb, b.rgb, atol=1e-5), (
            f"max difference {float((a.rgb - b.rgb).abs().max()):.2e}"
        )
        assert torch.allclose(a.alpha, b.alpha, atol=1e-5)

    def test_opaque_foreground_saturates_and_hides_what_is_behind(self):
        # The property the early exit relies on.
        near = torch.tensor([[0.0, 0.0, 1.0]]).repeat(40, 1)
        far = torch.tensor([[0.0, 0.0, 6.0]]).repeat(40, 1)
        points = torch.cat([near, far])
        colors = torch.cat([
            torch.tensor([[1.0, 0.0, 0.0]]).repeat(40, 1),
            torch.tensor([[0.0, 0.0, 1.0]]).repeat(40, 1),
        ])
        model = GaussianModel(points, colors, torch.full((80, 3), 0.2),
                              sh_degree=0, device=DEVICE)
        with torch.no_grad():
            model.opacity_logit.fill_(4.0)
        result = render(model, identity_view(), 300.0, 300.0, 64.0, 64.0, 128, 128)
        centre = result.rgb[64, 64]
        assert float(centre[0]) > 0.5
        assert float(centre[2]) < 0.1
