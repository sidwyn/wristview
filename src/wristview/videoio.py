"""Video probing, frame extraction, and frame-quality measures.

ffmpeg does the decoding. It handles iPhone HEVC and rotation metadata, which
OpenCV's capture path gets wrong often enough to matter.
"""

from __future__ import annotations

import json
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
        }


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
    focal_35mm = None
    tags = {**payload.get("format", {}).get("tags", {}), **stream.get("tags", {})}
    for key in ("focal_length_in_35mm_film", "com.apple.quicktime.lens", "focal_length"):
        if key in tags:
            try:
                focal_35mm = float(str(tags[key]).split()[0])
                break
            except (ValueError, IndexError):
                continue

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

