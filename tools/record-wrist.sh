#!/usr/bin/env bash
# Wrist camera capture — the ONLY way this camera should ever be recorded.
#
# The rig parameters below are frozen. Changing any of them invalidates the
# lens calibration, which is a silent failure: the recording looks fine and
# every downstream measurement is wrong.
#
#   Usage:  ./record-wrist.sh <session> <clip> [seconds]
#   Ex:     ./record-wrist.sh real25 demo_1 20

set -euo pipefail

# ─── FROZEN RIG PARAMETERS ────────────────────────────────────────────────
# Change these ONLY together with a fresh calibration, and record the change
# in CAPTURE-SOP.md with the date and the reason.
DEVICE_NAME="Arducam"      # matched against ffmpeg's device list
VIDEO_SIZE="800x600"       # 4:3 = full sensor width. 16:9 modes crop.
FRAMERATE="30"             # exact; 29.97 is rejected by the device
CALIBRATION="calib/wrist_800x600.json"
# ──────────────────────────────────────────────────────────────────────────

SESSION="${1:?usage: record-wrist.sh <session> <clip> [seconds]}"
CLIP="${2:?usage: record-wrist.sh <session> <clip> [seconds]}"
SECS="${3:-20}"

OUT_DIR="captures/${SESSION}"
OUT="${OUT_DIR}/wrist_${CLIP}.mov"
mkdir -p "$OUT_DIR"

# ─── Resolve the device index by name, never by a hardcoded number ────────
# Index order changes when other cameras are plugged in. Matching the name
# means the script cannot silently record from the built-in camera.
IDX=$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1 \
      | grep -i "$DEVICE_NAME" | head -1 | sed -E 's/.*\[([0-9]+)\].*/\1/')

if [[ -z "${IDX:-}" ]]; then
  echo "FAIL: no video device matching '${DEVICE_NAME}'." >&2
  echo "Devices seen:" >&2
  ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -A20 "video devices" >&2
  exit 1
fi

echo "device      ${DEVICE_NAME} at index ${IDX}"
echo "mode        ${VIDEO_SIZE} @ ${FRAMERATE} fps"
echo "calibration ${CALIBRATION}"
echo "output      ${OUT}"
echo
echo "Point BOTH cameras at the sync slate for 3 s, do the take, then show the"
echo "slate again for 3 s. Recording ${SECS}s — Ctrl-C stops early."
echo

ffmpeg -hide_banner -loglevel warning \
  -f avfoundation -framerate "$FRAMERATE" -video_size "$VIDEO_SIZE" \
  -i "${IDX}:0" -t "$SECS" \
  -c:v libx264 -preset ultrafast -crf 18 -c:a aac \
  "$OUT"

# ─── VERIFY. A recording is not accepted until it is measured. ────────────
echo
echo "── verification ──────────────────────────────────────────────────────"

read -r W H NF DUR < <(ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=width,height,nb_frames \
  -show_entries format=duration \
  -of csv=p=0:nk=1 "$OUT" | paste -sd' ' -)

FPS=$(python3 -c "print(f'{$NF/$DUR:.2f}')")
echo "size        ${W}x${H}   (expected ${VIDEO_SIZE})"
echo "frames      ${NF} over ${DUR}s = ${FPS} fps   (expected ${FRAMERATE})"

if [[ "${W}x${H}" != "$VIDEO_SIZE" ]]; then
  echo "FAIL: sensor mode is not the frozen one. Calibration does not apply." >&2
  exit 1
fi

python3 -c "
import sys
fps=$FPS; want=float($FRAMERATE)
if abs(fps-want) > 0.5:
    print(f'FAIL: dropped frames — {fps:.2f} fps against {want}.'); sys.exit(1)
"

# Exposure drift. Auto-exposure cannot currently be locked on this camera,
# so measure it every time rather than assume it behaved.
echo -n "brightness  "
ffmpeg -v error -i "$OUT" -vf "fps=5,scale=160:90" -f rawvideo -pix_fmt gray - \
| python3 -c "
import sys
d=sys.stdin.buffer.read(); n=160*90
v=[sum(d[i*n:(i+1)*n])/n for i in range(len(d)//n)]
lo,hi=min(v),max(v); spread=hi-lo
print(f'{lo:.1f} to {hi:.1f}, spread {spread:.1f}')
print('            WARN: exposure drifted. Add light, or flag this clip.' if spread>8
      else '            stable.')
"

echo "──────────────────────────────────────────────────────────────────────"
echo "OK  ${OUT}"
