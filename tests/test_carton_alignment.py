"""The carton's pixel height must move with the action that lifts it.

A second signal beside optical flow, because the two fail differently. Flow
survives an episode where the object is occluded; this survives an episode
where the camera barely moves. real27's head camera moved 3 to 5 cm across a
whole take, which is exactly the case that weakens the flow signal.

These run on a synthetic episode, not a render, so the gate is ready before a
render exists. The shift is injected, so a pass means the gate detects a lag it
was told to create, not that a number came out small.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristview.lerobot_export import carton_height_lag, carton_row, landmark_frames

UP = np.array([0.0, 0.0, 1.0])
COLOUR = np.array([0.20, 0.70, 0.80])


class FakeDataset:
    """The three fields the gate reads, with a controllable lag."""

    def __init__(self, frames: int = 80, shift: int = 0, visible: slice | None = None):
        self.frames = frames
        t = np.linspace(0.0, 1.0, frames)
        self.height = 0.18 * np.sin(np.pi * t)          # a lift and a lower
        self.grip = np.where((t > 0.25) & (t < 0.75), 0.02, 0.085)
        self.shift = shift
        self.visible = visible if visible is not None else slice(0, frames)

    def __len__(self):
        return self.frames

    def _image(self, index: int) -> np.ndarray:
        image = np.zeros((360, 640, 3), dtype=np.float32)
        image[:, :, :] = 0.05
        source = min(max(index + self.shift, 0), self.frames - 1)
        if not (self.visible.start <= index < self.visible.stop):
            return image
        # The carton falls in the image as it rises in the world.
        row = int(280 - 900 * self.height[source])
        image[row:row + 30, 300:340] = COLOUR
        return image

    def __getitem__(self, index: int) -> dict:
        import torch

        following = min(index + 1, self.frames - 1)
        action = np.zeros(7, dtype=np.float32)
        action[2] = self.height[following] - self.height[index]
        state = np.zeros(7, dtype=np.float32)
        state[2] = self.height[index]
        state[6] = self.grip[index]
        return {
            "observation.images.wrist": torch.from_numpy(
                self._image(index).transpose(2, 0, 1)),
            "action": torch.from_numpy(action),
            "observation.state": torch.from_numpy(state),
        }


def test_carton_row_finds_the_block():
    image = np.zeros((360, 640, 3)) + 0.05
    image[100:130, 300:340] = COLOUR
    assert carton_row(image, COLOUR) == pytest.approx(114.5, abs=1.0)


def test_carton_row_returns_none_when_the_carton_is_gone():
    assert carton_row(np.zeros((360, 640, 3)) + 0.05, COLOUR) is None


def test_an_aligned_episode_peaks_at_lag_zero():
    report = carton_height_lag(FakeDataset(shift=0), COLOUR, UP)
    assert report["measurable"] is True
    assert report["peak_lag"] == 0
    # The carton falls in the image as the action lifts it, so the sign is
    # negative. A positive correlation here would mean the image is upside down.
    assert report["r_at_zero"] < -0.9


@pytest.mark.parametrize("shift", [-3, -1, 1, 2, 4])
def test_an_injected_shift_is_detected_and_named(shift):
    report = carton_height_lag(FakeDataset(shift=shift), COLOUR, UP)
    # The image at frame i shows the carton from frame i+shift, so the gate
    # should say the image leads the action by exactly `shift`.
    assert report["peak_lag"] == shift, (
        f"injected a {shift:+d} frame shift and the gate reported "
        f"{report['peak_lag']}"
    )


def test_an_invisible_carton_reports_rather_than_passes():
    """Could-not-measure must never render as fine."""
    report = carton_height_lag(FakeDataset(visible=slice(0, 5)), COLOUR, UP)
    assert report["measurable"] is False
    assert report["peak_lag"] is None
    assert "either way" in report["note"]


def test_the_four_landmark_frames_are_written(tmp_path):
    written = landmark_frames(FakeDataset(), tmp_path, UP)
    assert set(written) == {"first", "contact", "maximum_lift", "release"}
    assert written["first"]["frame"] == 0
    assert written["contact"]["frame"] < written["maximum_lift"]["frame"]
    assert written["maximum_lift"]["frame"] < written["release"]["frame"]
    # The lift really is the highest, and the grip really is closed at contact.
    assert written["maximum_lift"]["height_m"] > written["first"]["height_m"]
    assert written["contact"]["gripper_m"] < written["first"]["gripper_m"]
    for entry in written.values():
        assert (tmp_path / f"{entry['path'].split('/')[-1]}").exists()
