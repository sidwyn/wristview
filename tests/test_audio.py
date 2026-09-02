"""Whistle detection and clock fitting, on signals whose truth is known.

Real block footage did not exist when these were written, so every fixture is
synthesised: a whistle is a 2.5 kHz tone burst, the noise floor is broadband,
and the distractors are the things that actually defeated a level-based
detector, namely broadband impacts and low-frequency speech-band energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristview.audio import (
    band_envelope,
    coarse_offset,
    find_whistles,
    fit_clock,
    robust_z,
)

SR = 16000


def tone(duration_s: float, hz: float = 2500.0, amplitude: float = 0.6) -> np.ndarray:
    t = np.arange(int(duration_s * SR)) / SR
    return amplitude * np.sin(2 * np.pi * hz * t)


def block(whistle_times_s: list[float], total_s: float, duration_s: float = 0.75,
          noise: float = 0.002, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, noise, int(total_s * SR))
    for start in whistle_times_s:
        burst = tone(duration_s)
        at = int(start * SR)
        signal[at:at + len(burst)] += burst
    return signal


class TestFindWhistles:
    def test_finds_every_whistle_at_the_right_time(self):
        truth = [1.0, 5.0, 9.0, 13.0]
        found = find_whistles(block(truth, total_s=16.0))
        assert len(found) == len(truth)
        for whistle, start in zip(found, truth, strict=True):
            # Centre of a 750 ms burst starting at `start`.
            assert whistle.centre_s == pytest.approx(start + 0.375, abs=0.05)

    def test_duration_is_recovered(self):
        found = find_whistles(block([2.0], total_s=6.0, duration_s=0.75))
        assert found[0].duration_s == pytest.approx(0.75, abs=0.08)

    def test_a_block_of_ten_takes_has_eleven_whistles(self):
        truth = [1.0 + 6.0 * i for i in range(11)]
        found = find_whistles(block(truth, total_s=72.0))
        assert len(found) == 11

    def test_broadband_impact_is_not_a_whistle(self):
        """A dropped cube is loud and wideband, but it is not sustained."""
        rng = np.random.default_rng(1)
        signal = rng.normal(0.0, 0.002, int(10.0 * SR))
        at = int(4.0 * SR)
        signal[at:at + int(0.02 * SR)] += rng.normal(0.0, 0.9, int(0.02 * SR))
        assert find_whistles(signal) == []

    def test_low_frequency_energy_is_not_a_whistle(self):
        """Speech-band rumble is sustained but sits below the band."""
        rng = np.random.default_rng(2)
        signal = rng.normal(0.0, 0.002, int(10.0 * SR))
        at = int(3.0 * SR)
        burst = tone(1.0, hz=200.0, amplitude=0.8)
        signal[at:at + len(burst)] += burst
        assert find_whistles(signal) == []

    def test_too_short_a_tone_is_rejected(self):
        assert find_whistles(block([2.0], total_s=6.0, duration_s=0.05)) == []

    def test_too_long_a_tone_is_rejected(self):
        assert find_whistles(block([1.0], total_s=10.0, duration_s=4.0)) == []

    def test_expect_keeps_the_strongest_when_the_threshold_is_loose(self):
        """Absolute z does not transfer between rooms; a known count does."""
        rng = np.random.default_rng(3)
        signal = rng.normal(0.0, 0.002, int(30.0 * SR))
        for index, start in enumerate([2.0, 8.0, 14.0, 20.0]):
            burst = tone(0.75, amplitude=0.6 if index < 3 else 0.05)
            at = int(start * SR)
            signal[at:at + len(burst)] += burst
        found = find_whistles(signal, z_min=3.0, expect=3)
        assert len(found) == 3
        # The weak fourth is the one dropped, and order stays chronological.
        assert [w.centre_s for w in found] == sorted(w.centre_s for w in found)
        assert all(w.centre_s < 16.0 for w in found)

    def test_silence_finds_nothing(self):
        assert find_whistles(np.zeros(SR * 5)) == []


class TestEnvelope:
    def test_envelope_peaks_where_the_sound_is(self):
        envelope, hop = band_envelope(block([3.0], total_s=8.0))
        assert envelope[int(3.3 / hop)] > 10 * envelope[int(1.0 / hop)]

    def test_zero_phase_filter_does_not_delay_the_peak(self):
        """A causal filter would push the peak later and bias every offset."""
        envelope, hop = band_envelope(block([2.0], total_s=6.0, duration_s=0.5))
        peak_s = int(np.argmax(envelope)) * hop
        assert 2.0 <= peak_s <= 2.5

    def test_robust_z_survives_a_silent_floor(self):
        """MAD is 0 on digital silence; the score must stay finite."""
        signal = np.zeros(SR * 4)
        signal[SR:SR + int(0.75 * SR)] += tone(0.75)
        scores = robust_z(band_envelope(signal)[0])
        assert np.all(np.isfinite(scores))
        # A silent floor is the easy case: the score should be enormous, not
        # marginal. An earlier fallback divided each whistle by its own
        # loudness and scored 5.3, under every sane threshold.
        assert scores.max() > 1000

    def test_a_quiet_room_still_detects(self):
        """The quieter the room, the higher the score should be, not the lower."""
        loud = find_whistles(block([2.0], total_s=8.0, noise=0.02, seed=7))
        quiet = find_whistles(block([2.0], total_s=8.0, noise=0.0002, seed=7))
        assert len(loud) == len(quiet) == 1
        assert quiet[0].peak_z > loud[0].peak_z


class TestFitClock:
    def test_recovers_a_known_rate_and_offset(self):
        ego = np.arange(0.0, 60.0, 6.0)
        wrist = 1.0019 * ego + 0.42
        rate, offset, residual = fit_clock(ego, wrist)
        assert rate == pytest.approx(1.0019, rel=1e-9)
        assert offset == pytest.approx(0.42, abs=1e-9)
        assert np.abs(residual).max() < 1e-9

    def test_the_305_ms_sample_drift_is_representable(self):
        """+305 ms over 16 s was measured on a real pair; a pure offset cannot."""
        ego = np.linspace(0.0, 16.0, 11)
        wrist = ego + np.linspace(0.0, 0.305, 11)
        rate, offset, residual = fit_clock(ego, wrist)
        assert np.abs(residual).max() < 1e-6
        assert (rate - 1.0) * 16.0 == pytest.approx(0.305, abs=1e-6)

    def test_a_pure_offset_model_would_be_wrong_by_the_drift(self):
        ego = np.linspace(0.0, 16.0, 11)
        wrist = ego + np.linspace(0.0, 0.305, 11)
        best_offset = float(np.mean(wrist - ego))
        assert np.abs(wrist - (ego + best_offset)).max() > 0.15

    def test_one_bad_event_shows_up_as_one_large_residual(self):
        ego = np.arange(0.0, 60.0, 6.0)
        wrist = 1.001 * ego + 0.2
        wrist[4] += 0.4
        _, _, residual = fit_clock(ego, wrist)
        assert np.argmax(np.abs(residual)) == 4
        assert np.abs(residual).max() > 0.2

    def test_needs_two_events_to_fit_a_rate(self):
        with pytest.raises(ValueError, match="at least two"):
            fit_clock(np.array([1.0]), np.array([1.5]))

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="against"):
            fit_clock(np.array([1.0, 2.0]), np.array([1.0]))


class TestCoarseOffset:
    @pytest.mark.parametrize("shift_s", [0.5, 1.25, 2.0])
    def test_recovers_a_known_shift(self, shift_s):
        truth = [2.0, 8.0, 14.0]
        ego = block(truth, total_s=20.0, seed=4)
        wrist = block([t + shift_s for t in truth], total_s=20.0, seed=5)
        assert coarse_offset(ego, wrist) == pytest.approx(shift_s, abs=0.05)

    def test_zero_shift_reads_zero(self):
        ego = block([2.0, 8.0], total_s=12.0, seed=6)
        assert coarse_offset(ego, ego) == pytest.approx(0.0, abs=0.02)
