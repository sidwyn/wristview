"""Stage 0 frame filtering.

The regression this file exists for: a standalone COLMAP run registered 182 of
182 scan frames, while ingest was discarding 41 percent of them before COLMAP
ever saw them. Blur rejection was absolute, and demo clips were being
deduplicated.
"""

import cv2
import numpy as np
import pytest

from wristview.stages.s00_ingest import _filter_frames
from wristview.videoio import hamming, laplacian_variance, phash


def write_frame(path, seed: int, blur: int = 0, size=(240, 320)):
    """A textured frame, optionally blurred."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
    # Add structure so the perceptual hash has something to key on.
    cv2.rectangle(image, (40, 40), (120, 160), (255, 0, 0), -1)
    cv2.circle(image, (200, 120), 40 + seed % 20, (0, 255, 0), -1)
    if blur:
        image = cv2.GaussianBlur(image, (blur * 2 + 1, blur * 2 + 1), 0)
    cv2.imwrite(str(path), image)
    return path


class TestBlurMeasure:
    def test_blurring_lowers_the_score(self, tmp_path):
        sharp = cv2.imread(str(write_frame(tmp_path / "a.png", 0)))
        blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
        assert laplacian_variance(blurred) < laplacian_variance(sharp)

    def test_a_flat_image_scores_near_zero(self):
        assert laplacian_variance(np.full((100, 100, 3), 128, np.uint8)) < 1.0


class TestPhash:
    def test_identical_images_hash_the_same(self, tmp_path):
        image = cv2.imread(str(write_frame(tmp_path / "a.png", 1)))
        assert hamming(phash(image), phash(image.copy())) == 0

    def test_different_images_differ(self, tmp_path):
        a = cv2.imread(str(write_frame(tmp_path / "a.png", 1)))
        b = cv2.imread(str(write_frame(tmp_path / "b.png", 99)))
        assert hamming(phash(a), phash(b)) > 0


class TestFilterFrames:
    def test_keeps_uniformly_sharp_frames(self, tmp_path):
        # The regression: a clip COLMAP can fully register must survive ingest.
        paths = [write_frame(tmp_path / f"f{i:03d}.png", i) for i in range(40)]
        kept, stats = _filter_frames(paths, 0.30, 3.0, 0.15, phash_min_distance=2)
        assert len(kept) == 40, f"dropped {40 - len(kept)} sharp frames: {stats}"
        assert stats["dropped_blur"] == 0

    def test_drops_a_genuinely_blurred_frame(self, tmp_path):
        paths = [write_frame(tmp_path / f"f{i:03d}.png", i) for i in range(20)]
        paths.append(write_frame(tmp_path / "blurred.png", 500, blur=12))
        kept, _ = _filter_frames(paths, 0.30, 3.0, 0.15, phash_min_distance=0)
        assert paths[-1] not in kept

    def test_never_drops_more_than_the_cap(self, tmp_path):
        # A uniformly soft clip must not lose everything.
        paths = [write_frame(tmp_path / f"f{i:03d}.png", i, blur=9) for i in range(30)]
        kept, _ = _filter_frames(paths, 0.95, 1e9, 0.15, phash_min_distance=0)
        assert len(kept) >= int(30 * 0.85)

    def test_threshold_is_relative_to_the_clip(self, tmp_path):
        # Two clips with very different texture must keep the same share.
        rng = np.random.default_rng(0)
        busy, calm = [], []
        for i in range(20):
            image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"busy{i}.png"), image)
            busy.append(tmp_path / f"busy{i}.png")

            soft = cv2.GaussianBlur(image, (9, 9), 0)
            cv2.imwrite(str(tmp_path / f"calm{i}.png"), soft)
            calm.append(tmp_path / f"calm{i}.png")

        kept_busy, _ = _filter_frames(busy, 0.30, 0.0, 0.15, 0)
        kept_calm, _ = _filter_frames(calm, 0.30, 0.0, 0.15, 0)
        assert len(kept_busy) == len(kept_calm) == 20

    def test_dedup_disabled_keeps_near_duplicates(self, tmp_path):
        # Demo clips at 60 fps: adjacent frames are near-identical by nature,
        # and every one is a distinct moment in the trajectory.
        base = write_frame(tmp_path / "base.png", 7)
        image = cv2.imread(str(base))
        paths = []
        for i in range(15):
            path = tmp_path / f"dup{i}.png"
            cv2.imwrite(str(path), image)
            paths.append(path)
        kept, stats = _filter_frames(paths, 0.30, 0.0, 0.15, phash_min_distance=0)
        assert len(kept) == 15
        assert stats["dropped_duplicate"] == 0

    def test_dedup_enabled_removes_identical_frames(self, tmp_path):
        image = cv2.imread(str(write_frame(tmp_path / "base.png", 7)))
        paths = []
        for i in range(15):
            path = tmp_path / f"dup{i}.png"
            cv2.imwrite(str(path), image)
            paths.append(path)
        kept, stats = _filter_frames(paths, 0.30, 0.0, 0.15, phash_min_distance=4)
        assert len(kept) == 1
        assert stats["dropped_duplicate"] == 14

    def test_empty_input(self):
        kept, stats = _filter_frames([], 0.3, 3.0, 0.15, 2)
        assert kept == []
        assert stats["kept"] == 0

    def test_reports_the_threshold_it_used(self, tmp_path):
        paths = [write_frame(tmp_path / f"f{i}.png", i) for i in range(10)]
        _, stats = _filter_frames(paths, 0.30, 3.0, 0.15, 0)
        assert len(stats["blur_per_pass"]) == 1
        entry = stats["blur_per_pass"][0]
        assert entry["threshold"] == pytest.approx(
            max(0.30 * entry["sharpness_median"], 3.0), abs=0.02
        )

    def test_each_pass_is_judged_against_its_own_median(self, tmp_path):
        """A soft close pass must not be culled by a sharp pass beside it.

        On real26 a third, sharper pass raised the clip median from 132.8 to
        149.6 and cost the low pass 18 frames it had previously kept. The
        frames it cost were the lowest, because those are the blurriest, and
        the reconstructed floor rose from 8.47 cm to 12.15 cm on identical
        footage.
        """
        import numpy as np

        # A sharp pass, and a soft pass whose own spread straddles the sharp
        # pass's threshold. Pooled, the soft pass is judged against a median
        # it did not set.
        sharp = [write_frame(tmp_path / f"s{i}.png", i, blur=0) for i in range(10)]
        soft = [write_frame(tmp_path / f"b{i}.png", 100 + i, blur=3 + i % 4)
                for i in range(10)]
        paths = sharp + soft
        labels = np.array([0] * 10 + [1] * 10)

        _, pooled = _filter_frames(paths, 0.60, 0.0, 0.90, 0)
        _, split = _filter_frames(paths, 0.60, 0.0, 0.90, 0, segment_of=labels)

        assert len(split["blur_per_pass"]) == 2
        soft_entry = next(e for e in split["blur_per_pass"] if e["pass"] == 1)
        sharp_entry = next(e for e in split["blur_per_pass"] if e["pass"] == 0)
        # The soft pass sets a lower bar for itself than the sharp pass sets.
        assert soft_entry["threshold"] < sharp_entry["threshold"]
        # And it therefore keeps frames the pooled median would have culled.
        assert split["kept"] > pooled["kept"]

    def test_mislabelled_segments_raise(self, tmp_path):
        import numpy as np

        paths = [write_frame(tmp_path / f"f{i}.png", i) for i in range(5)]
        with pytest.raises(ValueError, match="segment_of labels"):
            _filter_frames(paths, 0.30, 0.0, 0.15, 0, segment_of=np.zeros(3, dtype=int))
