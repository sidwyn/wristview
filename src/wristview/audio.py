"""Whistle detection and audio-clock fitting for continuously recorded blocks.

Ego and wrist each record straight through a block of takes. A whistle marks
every take boundary, so the whistles are both the cut points and the only
common clock the two cameras share.

Two things here are not obvious and cost real data when they are got wrong.

WHY A BAND, NOT A LEVEL. A whistle is loud, but so is a dropped cube and so is
a chair. What separates a whistle is that its energy sits in a narrow band
around 1 to 5 kHz and stays there for the best part of a second. Detecting on
broadband level finds every impact in the room; detecting on band energy with
a duration floor finds whistles.

WHY PTS, NEVER index/fps. The wrist container declares 30 fps and 509 frames
while a frame-by-frame scan finds 527 at 31.06 fps, reproducibly. Anything
computing a frame's time as index/30 is 3.5 per cent wrong, which is 500 ms
over a 16 s take, dwarfing the drift this module exists to measure. Frame
times come from the container's presentation timestamps or not at all.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .logging_setup import get
from .videoio import ffmpeg_binary, ffprobe_binary

log = get(__name__)

# The whistle band. Below 1 kHz sits speech and room rumble; above 5 kHz there
# is little whistle energy and a lot of handling noise.
WHISTLE_BAND_HZ = (1000.0, 5000.0)

# Audio is decoded at this rate. It only has to clear 2x the top of the band.
SAMPLE_RATE_HZ = 16000

# Envelope resolution. 5 ms is far finer than the ~700 ms whistles being found
# and keeps the edge of a whistle located to well inside one video frame.
HOP_S = 0.005
WINDOW_S = 0.025


@dataclass(frozen=True)
class Whistle:
    """One detected whistle, in the file's own timebase."""

    start_s: float
    end_s: float
    peak_z: float

    @property
    def centre_s(self) -> float:
        return (self.start_s + self.end_s) / 2.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "centre_s": round(self.centre_s, 4),
            "duration_s": round(self.duration_s, 4),
            "peak_z": round(self.peak_z, 2),
        }


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Decode a file's first audio stream to mono float32 in [-1, 1]."""
    path = Path(path)
    command = [
        ffmpeg_binary(), "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-",
    ]
    done = subprocess.run(command, capture_output=True)
    if done.returncode != 0 or not done.stdout:
        raise RuntimeError(
            f"no audio decoded from {path.name}. "
            f"ffmpeg said: {done.stderr.decode('utf-8', 'replace').strip()[:300]}"
        )
    return np.frombuffer(done.stdout, dtype=np.float32).astype(np.float64)


def band_envelope(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE_HZ,
    band_hz: tuple[float, float] = WHISTLE_BAND_HZ,
    hop_s: float = HOP_S,
    window_s: float = WINDOW_S,
) -> tuple[np.ndarray, float]:
    """Band-limit, then take short-time RMS. Returns the envelope and its hop.

    The filter is zero-phase, so an envelope peak sits where the sound is
    rather than a filter delay later. That matters: the whole point is to
    compare event times between two recordings.
    """
    nyquist = sample_rate / 2.0
    low = max(band_hz[0] / nyquist, 1e-6)
    high = min(band_hz[1] / nyquist, 0.999)
    if low >= high:
        raise ValueError(f"band {band_hz} is empty at {sample_rate} Hz")

    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    # `sosfiltfilt` needs more samples than its padding length. A clip too
    # short for the filter is a caller error worth naming.
    if len(samples) < 3 * 8:
        raise ValueError(f"only {len(samples)} audio samples; too short to filter")
    filtered = signal.sosfiltfilt(sos, samples)

    hop = max(int(round(hop_s * sample_rate)), 1)
    window = max(int(round(window_s * sample_rate)), hop)
    power = filtered * filtered
    # Cumulative sum gives a boxcar moving average in one pass.
    padded = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, max(len(power) - window, 0) + 1, hop)
    if len(starts) == 0:
        return np.zeros(0), hop / sample_rate
    envelope = np.sqrt((padded[starts + window] - padded[starts]) / window)
    return envelope, hop / sample_rate


def robust_z(envelope: np.ndarray) -> np.ndarray:
    """Score an envelope in robust standard deviations above its own floor.

    Median and MAD, not mean and standard deviation: a block is mostly silence
    punctuated by the very events being detected, and those events would drag
    a mean-based score toward themselves and hide the quietest whistle.
    """
    if envelope.size == 0:
        return envelope
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    scale = 1.4826 * mad
    if scale <= 0:
        # A digitally silent floor gives MAD 0, and then any energy at all is
        # unboundedly many deviations above it. Scale by a millionth of the
        # loudest point: the division stays defined and the answer stays
        # honest, which is that these events are unambiguous.
        #
        # Falling back to the MEAN deviation here was wrong. On a quiet
        # recording that mean is dominated by the whistles themselves, so each
        # whistle was divided by its own loudness and scored about 5 -- under
        # any sane threshold. A near-silent room is the EASY case and must not
        # be the one that fails.
        peak = float(np.max(np.abs(envelope - median)))
        scale = peak * 1e-6 if peak > 0 else 1.0
    return (envelope - median) / scale


def find_whistles(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE_HZ,
    z_min: float = 20.0,
    min_ms: float = 300.0,
    max_ms: float = 1500.0,
    merge_gap_ms: float = 120.0,
    expect: int | None = None,
    band_hz: tuple[float, float] = WHISTLE_BAND_HZ,
) -> list[Whistle]:
    """Find whistles as sustained runs of band energy above the noise floor.

    `expect` takes the N strongest candidates instead of trusting `z_min`.
    Absolute z depends on how quiet the room was, so it does not transfer
    between sessions; a known whistle count does.
    """
    envelope, hop_s = band_envelope(samples, sample_rate, band_hz)
    if envelope.size == 0:
        return []
    scores = robust_z(envelope)

    above = scores >= z_min
    if not above.any():
        return []

    # Contiguous runs, then bridge the short dips inside one whistle.
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        stops.append(len(above))

    merge_gap = merge_gap_ms / 1000.0
    runs: list[list[int]] = []
    for start, stop in zip(starts, stops, strict=True):
        if runs and (start - runs[-1][1]) * hop_s <= merge_gap:
            runs[-1][1] = stop
        else:
            runs.append([start, stop])

    found = []
    for start, stop in runs:
        duration_ms = (stop - start) * hop_s * 1000.0
        if duration_ms < min_ms or duration_ms > max_ms:
            continue
        found.append(
            Whistle(
                start_s=start * hop_s,
                end_s=stop * hop_s,
                peak_z=float(scores[start:stop].max()),
            )
        )

    if expect is not None and len(found) > expect:
        strongest = sorted(found, key=lambda w: w.peak_z, reverse=True)[:expect]
        found = sorted(strongest, key=lambda w: w.start_s)
    return found


def fit_clock(
    ego_s: np.ndarray, wrist_s: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Least-squares fit of `t_wrist = a * t_ego + b` over matched events.

    Returns the rate, the offset, and the per-event residual in seconds.

    The rate term is not decoration. Measured drift over ~16 s was +23 ms on
    one sample pair and +305 ms on another, so a single additive offset fitted
    on one whistle is wrong by up to a third of a second by the end of a take,
    and wrong by a different amount in every block.
    """
    ego_s = np.asarray(ego_s, dtype=np.float64)
    wrist_s = np.asarray(wrist_s, dtype=np.float64)
    if ego_s.shape != wrist_s.shape:
        raise ValueError(f"{ego_s.size} ego events against {wrist_s.size} wrist events")
    if ego_s.size < 2:
        raise ValueError("need at least two matched events to fit a rate")

    rate, offset = np.polyfit(ego_s, wrist_s, 1)
    residual = wrist_s - (rate * ego_s + offset)
    return float(rate), float(offset), residual


def coarse_offset(
    ego: np.ndarray, wrist: np.ndarray, sample_rate: int = SAMPLE_RATE_HZ
) -> float:
    """Cross-correlate two band envelopes for a starting offset, in seconds.

    Positive means the wrist recording starts later in absolute time, so a
    given event sits at a larger timestamp in the ego file.
    """
    ego_env, hop_s = band_envelope(ego, sample_rate)
    wrist_env, _ = band_envelope(wrist, sample_rate)
    if ego_env.size == 0 or wrist_env.size == 0:
        raise ValueError("an empty envelope cannot be correlated")

    a = ego_env - ego_env.mean()
    b = wrist_env - wrist_env.mean()
    correlation = signal.correlate(b, a, mode="full")
    lag = int(np.argmax(correlation)) - (len(a) - 1)
    return lag * hop_s


def video_frame_times(path: str | Path) -> np.ndarray:
    """Every video frame's presentation time, in seconds, from the container.

    Never index/fps. The wrist clips declare 30 fps and a frame count that is
    both wrong: 509 declared against 527 real, a true 31.06 fps. Times derived
    from the declared rate drift 3.5 per cent, which is 500 ms across a take.
    """
    path = Path(path)
    command = [
        ffprobe_binary(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {done.stderr.strip()[:300]}")

    frames = json.loads(done.stdout or "{}").get("frames", [])
    times = [
        float(f["best_effort_timestamp_time"])
        for f in frames
        if f.get("best_effort_timestamp_time") not in (None, "N/A")
    ]
    if not times:
        raise RuntimeError(f"{path.name} reported no frame timestamps")
    return np.asarray(sorted(times), dtype=np.float64)
