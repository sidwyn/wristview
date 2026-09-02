"""Video probing, frame extraction, and frame-quality measures.

ffmpeg does the decoding. It handles iPhone HEVC and rotation metadata, which
OpenCV's capture path gets wrong often enough to matter.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .logging_setup import get

log = get(__name__)


def ffmpeg_binary() -> str:
    """Locate ffmpeg, preferring the system build over the bundled wheel."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 - report the real cause to the caller
        raise RuntimeError("ffmpeg not found. Run `brew install ffmpeg`.") from exc


def ffprobe_binary() -> str:
    found = shutil.which("ffprobe")
    if found:
        return found
    raise RuntimeError("ffprobe not found. Run `brew install ffmpeg`.")


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    rotation_deg: int
    codec: str
    focal_35mm: float | None
    focal_source: str = "none"

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 3),
            "rotation_deg": self.rotation_deg,
            "codec": self.codec,
            "focal_35mm": self.focal_35mm,
            "focal_source": self.focal_source,
        }


# Tags that carry a 35mm-equivalent focal length as a bare number.
FOCAL_NUMERIC_TAGS = (
    "focal_length_in_35mm_film",
    "com.apple.quicktime.lens",
    "focal_length",
)

# Tags that name the lens in prose, with the 35mm equivalent on the end:
# "iPhone 16 Pro 24mm". Blackmagic Camera writes both of these and none of the
# numeric tags, which is why an iPhone clip can look like it has no EXIF at all.
FOCAL_LABEL_TAGS = (
    "com.blackmagic-design.camera.lensType",
    "com.apple.quicktime.model",
)

_LENS_MM = re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE)


def focal_35mm_from_tags(tags: dict) -> tuple[float | None, str]:
    """Read a 35mm-equivalent focal length out of the container tags.

    Returns the focal length and where it came from. A numeric tag is a
    measurement, so it is trusted. A lens label is the lens's nominal focal
    length, which video does not deliver because it crops the sensor: real27's
    24mm-labelled clip refined to the equivalent of 27mm. Label reads carry
    their own source so Stage 1 keeps refining them instead of freezing them.
    """
    for key in FOCAL_NUMERIC_TAGS:
        if key in tags:
            try:
                return float(str(tags[key]).split()[0]), "exif_focal35"
            except (ValueError, IndexError):
                continue

    for key in FOCAL_LABEL_TAGS:
        match = _LENS_MM.search(str(tags.get(key, "")))
        if match:
            return float(match.group(1)), "exif_lens_label"

    return None, "none"


def probe(path: str | Path) -> VideoInfo:
    """Read stream metadata with ffprobe."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    result = subprocess.run(
        [
            ffprobe_binary(),
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError(f"no video stream in {path}")
    stream = streams[0]

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key, "0/0")
        if "/" in raw:
            num, den = raw.split("/")
            if float(den) > 0 and float(num) > 0:
                fps = float(num) / float(den)
                break

    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    frame_count = int(stream.get("nb_frames") or 0)
    if frame_count == 0 and fps > 0 and duration > 0:
        frame_count = int(round(fps * duration))

    # Rotation lives in a side-data block on iPhone footage.
    rotation = 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    if rotation == 0:
        rotation = int(float(stream.get("tags", {}).get("rotate", 0) or 0))
    rotation = rotation % 360

    # ffprobe surfaces EXIF only sometimes. Absent is normal and handled.
    tags = {**payload.get("format", {}).get("tags", {}), **stream.get("tags", {})}
    focal_35mm, focal_source = focal_35mm_from_tags(tags)

    # A 90 or 270 degree rotation swaps the stored dimensions.
    if rotation in (90, 270):
        width, height = height, width

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration,
        rotation_deg=rotation,
        codec=str(stream.get("codec_name", "unknown")),
        focal_35mm=focal_35mm,
        focal_source=focal_source,
    )


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    fps: float | None = None,
    quality: int = 2,
    max_frames: int | None = None,
    prefix: str = "frame",
) -> list[Path]:
    """Extract frames to JPEG. `fps` of None keeps the native rate.

    Returns the extracted paths in order.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{prefix}_*.jpg"):
        stale.unlink()

    command = [ffmpeg_binary(), "-y", "-loglevel", "error", "-i", str(video)]
    filters = []
    if fps and fps > 0:
        filters.append(f"fps={fps}")
    if filters:
        command += ["-vf", ",".join(filters)]
    if max_frames:
        command += ["-frames:v", str(max_frames)]
    command += ["-q:v", str(quality), str(out_dir / f"{prefix}_%05d.jpg")]

    log.debug("ffmpeg: %s", " ".join(command))
    subprocess.run(command, check=True, capture_output=True, text=True)

    frames = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    log.info("extracted %d frames from %s to %s", len(frames), video.name, out_dir)
    return frames


def laplacian_variance(image: np.ndarray) -> float:
    """Blur measure. Higher is sharper. The standard variance-of-Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def best_tile_normalised_variance(image: np.ndarray, tiles: int = 3) -> float:
    """Sharpness of the sharpest `tiles` x `tiles` region, on the FRAME's contrast.

    `normalised_laplacian_variance` averages over the whole frame, which is the
    right measure when the frame is uniformly in focus and the wrong one when
    it is not. Close to a surface the depth of field is a centimetre or two, so
    a band of the frame is properly sharp and the rest falls away.

    **Why the divisor is the whole frame and not the tile.** Normalising each
    tile by its OWN standard deviation rewards flat tiles: a patch of shadow
    has almost no contrast, so dividing by it amplifies quantisation noise into
    a high score. Measured on real31's scan_full, a per-tile divisor made the
    winning tile the LOW-contrast one on 72.5 per cent of frames, median
    winning contrast 5.18 grey levels against 20.14 across all tiles, with
    contrast and score correlating -0.318. The measure was ranking shadow above
    detail, and it reported that pass as sharper than real26/d's low pass.

    Dividing every tile by the frame's contrast keeps the exposure invariance
    the floor needs, since a darker take of the same scene scales both, while
    leaving a flat tile scoring near zero where it belongs.

    `tiles=1` reduces to `normalised_laplacian_variance` exactly.
    """
    if tiles < 1:
        raise ValueError(f"tiles must be at least 1, got {tiles}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float64)
    spread = gray.std()
    if spread < 1e-6:
        return 0.0
    scaled = (gray - gray.mean()) / spread
    if tiles == 1:
        return float(cv2.Laplacian(scaled, cv2.CV_64F).var())
    height, width = scaled.shape[:2]
    step_y, step_x = height // tiles, width // tiles
    if step_y < 8 or step_x < 8:
        return float(cv2.Laplacian(scaled, cv2.CV_64F).var())
    best = 0.0
    for row in range(tiles):
        for column in range(tiles):
            patch = scaled[row * step_y:(row + 1) * step_y,
                           column * step_x:(column + 1) * step_x]
            best = max(best, float(cv2.Laplacian(patch, cv2.CV_64F).var()))
    return best


def normalised_laplacian_variance(image: np.ndarray) -> float:
    """Blur measure with exposure and contrast divided out.

    `laplacian_variance` scales with image contrast, so a darker or flatter
    exposure of the SAME scene at the SAME focus scores lower. That makes it
    unusable as an absolute floor across passes shot at different exposures.
    real27's two low passes measured 7.1 and 18.7 raw, a factor of 2.6, while
    the second is 5.5x sharper once contrast is removed: it was simply darker,
    mean grey 60.1 against 120.8.

    Standardising each frame to zero mean and unit standard deviation first
    gives a number comparable across exposures. Measured on real26/d's
    registered views below 15 cm, the only low pass known to have produced a
    usable reconstruction, the median is 0.0323.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float64)
    spread = gray.std()
    if spread < 1e-6:
        return 0.0
    return float(cv2.Laplacian((gray - gray.mean()) / spread, cv2.CV_64F).var())


def phash(image: np.ndarray, hash_size: int = 8) -> np.uint64:
    """64-bit perceptual hash by DCT, used to drop near-duplicate frames."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:hash_size, :hash_size].flatten()
    # Skip the DC term: it tracks brightness, not structure.
    median = np.median(low[1:])
    bits = low > median
    value = np.uint64(0)
    for bit in bits:
        value = np.uint64(value << np.uint64(1)) | np.uint64(bool(bit))
    return value


def hamming(a: np.uint64, b: np.uint64) -> int:
    return int(bin(int(a) ^ int(b)).count("1"))


def write_video(
    frames_dir: str | Path,
    out_path: str | Path,
    fps: float = 15.0,
    pattern: str = "%05d.png",
) -> Path | None:
    """Encode a frame directory to an mp4 preview. Returns None if it fails."""
    frames_dir = Path(frames_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary(), "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        # H.264 requires even dimensions.
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return out_path
    except subprocess.CalledProcessError as exc:
        log.warning("preview video failed: %s", exc.stderr.strip()[:400])
        return None


def audio_streams(path) -> list[dict]:
    """List the audio streams in a file. Return an empty list when there are none.

    Parse the JSON output. Do not read a line of text output.

    A text-mode check reported "no audio" for two files that carry audio. Those
    two files hold a third stream, a one-frame timecode track. ffprobe emits an
    empty record for it, so the csv output starts with a blank line. A `head -1`
    then returns that blank line. The third file has two streams, no blank line,
    and the same command worked. The failure was silent and it was wrong in the
    direction that loses data.
    """
    import json as _json
    import subprocess

    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    payload = _json.loads(result.stdout)
    return [
        stream for stream in payload.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]


def has_audio(path) -> bool:
    """State whether a file carries an audio stream."""
    return bool(audio_streams(path))

