"""Generate a printable ArUco marker for metric scale.

The marker is the cheapest way to give a reconstruction real units. Unlike the
known-object route it needs no detector, no segmentation, and no guess about
what is in the shot: the side length is whatever you printed, and Stage 1
triangulates it.

What matters is that the printed size is exactly what the config says. Printer
scaling is the usual way this goes wrong, so the sheet carries its own ruler
and the number to check.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .logging_setup import get

log = get(__name__)

# 300 dpi is a safe floor for a desktop printer and gives crisp module edges.
DPI = 300
MM_PER_INCH = 25.4


def generate(
    out_path: Path,
    side_m: float = 0.15,
    marker_id: int = 0,
    dictionary_name: str = "DICT_4X4_50",
    quiet_zone_fraction: float = 0.25,
) -> Path:
    """Write a printable marker sheet at an exact physical size.

    `side_m` is the black square's side, edge to edge, which is what
    `scene.scale.aruco_marker_length_m` must be set to.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    side_px = int(round(side_m * 1000 / MM_PER_INCH * DPI))
    quiet_px = int(round(side_px * quiet_zone_fraction))
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)

    # The quiet zone is part of the specification, not decoration: without it
    # detection is unreliable against a busy background.
    sheet_w = side_px + 2 * quiet_px
    sheet_h = side_px + 2 * quiet_px + int(round(0.9 * quiet_px))
    sheet = np.full((sheet_h, sheet_w), 255, np.uint8)
    sheet[quiet_px:quiet_px + side_px, quiet_px:quiet_px + side_px] = marker

    # Corner ticks and a printed length, so a ruler can verify the print
    # scaled correctly. A marker printed at 96 percent silently biases every
    # downstream distance by the same 4 percent.
    y = quiet_px + side_px + int(round(0.34 * quiet_px))
    cv2.line(sheet, (quiet_px, y), (quiet_px + side_px, y), 0, max(2, side_px // 300))
    for x in (quiet_px, quiet_px + side_px):
        cv2.line(sheet, (x, y - quiet_px // 6), (x, y + quiet_px // 6), 0,
                 max(2, side_px // 300))

    label = f"{side_m * 100:.1f} cm  -  measure this line before use"
    scale = side_px / 1400
    cv2.putText(
        sheet, label, (quiet_px, y + int(round(0.72 * quiet_px))),
        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, scale), 0, max(1, int(round(2 * scale))),
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet, f"{dictionary_name} id={marker_id}",
        (quiet_px, quiet_px - int(round(0.28 * quiet_px))),
        cv2.FONT_HERSHEY_SIMPLEX, max(0.3, scale * 0.7), 0, max(1, int(round(scale))),
        cv2.LINE_AA,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)

    log.info(
        "marker %d (%s) at %.1f cm, %d dpi, %dx%d px -> %s",
        marker_id, dictionary_name, side_m * 100, DPI, sheet_w, sheet_h, out_path,
    )
    return out_path


def verify(image_path: Path, dictionary_name: str = "DICT_4X4_50") -> list[int]:
    """Detect markers in an image. Returns the ids found."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read {image_path}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(image)
    return [] if ids is None else [int(v) for v in ids.ravel()]
