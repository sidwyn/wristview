#!/usr/bin/env bash
#
# Run Stage 2 (demo localization) on a rented CUDA box.
#
# Stage 2 is 98.5 percent LightGlue matching, and LightGlue is 39 times faster
# on a 4090 than on this Mac's MPS: 7.3 ms a pair against 282.8 ms. A 30 clip
# session measured 15.1 minutes on the pod against 368.9 minutes locally.
#
# Stage 1 is deliberately NOT here. Its matching speeds up the same way, but
# COLMAP's incremental mapper is serial, ran 1.5x SLOWER on the pod's Ryzen
# 7950X than on the Mac, and is 83 percent of the stage once matching is quick.
# Measured end to end: 4.6x, which does not pay for the code. Run Stage 1
# locally and ship its output here.
#
#   tools/pod_stage2.sh linktest  --host H --port P
#   tools/pod_stage2.sh provision --host H --port P
#   tools/pod_stage2.sh push      --host H --port P --run runs/real31full
#   tools/pod_stage2.sh stage2    --host H --port P --run runs/real31full
#   tools/pod_stage2.sh pull      --host H --port P --run runs/real31full
#   tools/pod_stage2.sh all       --host H --port P --run runs/real31full
#
# Get the host and port from `runpodctl pod get <id> -o json`, field
# `.ssh.ssh_command`. Pods MUST be created with `--ports 22/tcp` or no SSH is
# ever published.
#
set -euo pipefail

KEY="${HOME}/.runpod/ssh/runpodctl-ssh-key"
HOST=""
PORT=""
RUN=""
REMOTE="/workspace"
VERB="${1:-}"
[ $# -gt 0 ] && shift

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --run) RUN="$2"; shift 2 ;;
    --remote) REMOTE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "need --host and --port" >&2; exit 2; }

# aes128-gcm, and compression off. The default cipher is roughly three times
# slower on these links and the payload is already-compressed JPEG and float32,
# so gzip only burns CPU.
SSH_OPTS=(-i "$KEY" -c aes128-gcm@openssh.com -o Compression=no
          -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
SSH=(ssh "${SSH_OPTS[@]}" -p "$PORT" "root@${HOST}")
SCP=(scp "${SSH_OPTS[@]}" -P "$PORT")

need_run() {
  [ -n "$RUN" ] || { echo "need --run <local run directory>" >&2; exit 2; }
  [ -d "$RUN" ] || { echo "no such run directory: $RUN" >&2; exit 2; }
}

RUN_ID=""
[ -n "$RUN" ] && RUN_ID="$(basename "$RUN")"

# ---------------------------------------------------------------------------
# linktest. Do this before pushing gigabytes.
#
# Pod network quality varies enormously and is not visible from the listing. A
# measured EUR-IS-2 pod ran at 0.31 MB/s while this Mac pushed 30 MB/s to
# Cloudflare at the same moment, so the pod was the bottleneck, not the uplink.
# At 0.31 MB/s a Stage 2 payload takes 75 minutes. Terminate and rent another.
# ---------------------------------------------------------------------------
# Below this a push costs more in pod time than re-renting. Measured: the
# same 2.9 GB payload moved at 14 MB/s on a good pod and 3 MB/s on a poor one.
MIN_LINK_MB_S="${MIN_LINK_MB_S:-4}"

cmd_linktest() {
  echo "== pod =="
  "${SSH[@]}" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
               echo cores=\$(nproc); python3 -c 'import sys; print(\"python\", sys.version.split()[0])'"
  echo "== uplink, 100 MB =="
  dd if=/dev/zero bs=1m count=100 2>/dev/null > /tmp/pod_linktest.bin
  local start end secs
  start=$(date +%s)
  "${SSH[@]}" "cat > /dev/null" < /tmp/pod_linktest.bin
  end=$(date +%s)
  rm -f /tmp/pod_linktest.bin
  secs=$(( end - start )); [ "$secs" -lt 1 ] && secs=1
  local rate=$(( 100 / secs ))
  echo "100 MB in ${secs}s = ${rate} MB/s"

  # RETURN the verdict, do not merely print it. This printed "terminate this
  # pod and rent another" and returned 0, so every caller sailed past it. That
  # is the defect this codebase repeats: compute the number, then never read
  # it. A slow pod is paid for at $0.74/hr while it crawls -- one 2.9 GB push
  # at 3 MB/s costs about 16 minutes against 4 on a good link.
  if [ "$rate" -lt "$MIN_LINK_MB_S" ]; then
    echo "LINK TOO SLOW: ${rate} MB/s is under the ${MIN_LINK_MB_S} MB/s floor." >&2
    echo "Terminate this pod and rent another; a push at this rate wastes more" >&2
    echo "than a fresh pod costs." >&2
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# provision. Four traps, all of them cost an hour the first time.
# ---------------------------------------------------------------------------
cmd_provision() {
  "${SSH[@]}" "mkdir -p ${REMOTE}/code" >/dev/null
  "${SSH[@]}" 'bash -s' <<'PROVISION'
set -euxo pipefail
export PIP_ROOT_USER_ACTION=ignore
T0=$(date +%s)

# TRAP 1: the image's python is PEP 668 externally-managed, so pip refuses to
# install anything into it and every command "succeeds" while doing nothing.
# --system-site-packages reuses the image's torch+cu128 rather than pulling
# 2.5 GB of wheels again.
python3 -m venv --system-site-packages /workspace/venv
PY=/workspace/venv/bin/python
PIP=/workspace/venv/bin/pip
$PIP install --no-cache-dir -q --upgrade pip

# opencv-python-headless, matching the manifest. ArUco has lived in the main
# objdetect module since OpenCV 4.7, so the contrib build buys nothing here;
# verified byte-identical marker detections between the two on real31scan.
# Headless because a server has no GUI libraries for the plain wheel to link.
# numpy is pinned to the Mac's version so a pose difference can never be
# blamed on a numpy major.
$PIP install --no-cache-dir "numpy==1.26.4" scipy h5py pyyaml tqdm \
    opencv-python-headless kornia

# pycolmap has a manylinux wheel. This was expected to be the hard part of the
# port and takes 4 seconds.
$PIP install --no-cache-dir pycolmap==4.1.1

# TRAP 3: pip lowercases a git+https URL, github answers the lowercased path
# with an auth prompt, and the clone dies on "could not read Username". The
# codeload archive works. hloc is Stage 1 only and is not installed here.
# TRAP 4: LightGlue does not declare kornia but imports it, so --no-deps alone
# yields a package that cannot be imported. kornia is installed above.
$PIP install --no-cache-dir --no-deps \
    "https://github.com/cvg/LightGlue/archive/refs/heads/main.zip"

$PY -c "import torch, pycolmap, cv2, h5py, numpy, scipy; from lightglue import SuperPoint, LightGlue
assert torch.cuda.is_available(), 'no CUDA visible'
assert hasattr(cv2, 'aruco'), 'opencv has no aruco: wrong opencv distribution'
print('OK  torch', torch.__version__, torch.cuda.get_device_name(0))
print('OK  pycolmap', pycolmap.__version__, '| cv2', cv2.__version__, '| numpy', numpy.__version__)"
echo "provision took $(( $(date +%s) - T0 ))s"
PROVISION

  # The package itself, installed from source with --no-deps: Stage 2 needs
  # only what provision installed above. Pulling the full dependency set would
  # drag in mediapipe, trimesh and yourdfpy, none of which Stage 2 imports.
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  tar -cf - -C "$here" src pyproject.toml README.md configs \
    | "${SSH[@]}" "tar -xf - -C ${REMOTE}/code"
  "${SSH[@]}" "/workspace/venv/bin/pip install --no-cache-dir -q --no-deps -e ${REMOTE}/code \
               && /workspace/venv/bin/wristview --help >/dev/null && echo 'OK  wristview CLI'"
}

# ---------------------------------------------------------------------------
# push. Only what Stage 2 reads.
#
# Notably NOT 01_scene/colmap/database.db: it is 245 MB and Stage 2 never opens
# it. 01_scene/features.h5 is 1.1 GB and is not optional, because the COLMAP
# model's point2D indices refer to exactly those keypoints. Regenerating it on
# the pod would renumber them and every 2D-3D correspondence would be garbage.
# ---------------------------------------------------------------------------
cmd_push() {
  need_run
  local dest="${REMOTE}/data/${RUN_ID}"
  "${SSH[@]}" "mkdir -p ${dest}/01_scene/colmap" >/dev/null

  local start end
  start=$(date +%s)
  tar -chf - -C "$RUN" \
      00_ingest/manifest.json 00_ingest/intrinsics.json \
      $( [ -f "$RUN/00_ingest/meta.json" ] && echo 00_ingest/meta.json ) \
      config.yaml \
    | "${SSH[@]}" "tar -xf - -C ${dest}"

  # Frames: the scan (retrieval descriptors and the marker's world pose) and
  # every demo clip.
  tar -chf - -C "$RUN/00_ingest" scan $(cd "$RUN/00_ingest" && ls -d demo_* 2>/dev/null) \
    | "${SSH[@]}" "tar -xf - -C ${dest}/00_ingest"

  tar -chf - -C "$RUN/01_scene" cameras.json scale.json meta.json features.h5 matches.h5 \
    | "${SSH[@]}" "tar -xf - -C ${dest}/01_scene"
  tar -chf - -C "$RUN/01_scene/colmap" sparse \
    | "${SSH[@]}" "tar -xf - -C ${dest}/01_scene/colmap"
  end=$(date +%s)

  echo "pushed $(du -shL "$RUN/00_ingest" "$RUN/01_scene" | awk '{s+=$1} END {print s}' 2>/dev/null || echo '?') in $(( end - start ))s"
  "${SSH[@]}" "du -sh ${dest}"
}

# Run one stage remotely and WAIT for it, reporting progress.
#
# Two defects this replaces, both found the hard way on a 60-clip run.
#
# 1. It was fire-and-forget: launch `nohup ... &`, print "launched", return.
#    A caller that ran `stage2` then `pull` would pull an empty directory. It
#    appeared to work only because of defect 2.
# 2. The launching ssh did not exit. It held the session open for minutes
#    after the remote job finished, with the pod billing at $0.74/hr. `ssh -n`
#    closes stdin, which is what keeps the channel open.
#
# So: launch detached with stdin closed, then poll for the process to leave
# the process table. Polling the PROCESS, not the log, because a stage that
# dies silently stops writing and would otherwise look like it was thinking.
remote_stage() {
  local stage="$1"
  need_run
  local log="${REMOTE}/stage${stage}.log"
  local done="${REMOTE}/stage${stage}.done"

  # A MARKER FILE, not pgrep. `pgrep -f "wristview stage 2"` run over ssh
  # matches the ssh command string that ASKS the question:
  #
  #   891 bash -c pgrep -af 'wristview stage 2' | head -3
  #
  # so it answers "still running" forever. Measured on this run: the remote
  # job finished at 12:37 and the wait loop would never have exited, billing
  # $0.74/hr indefinitely. The marker cannot lie: the job wrote it or it did
  # not. Same fix as `remote_job` in the e2e driver, which had the same bug.
  "${SSH[@]}" -n "cd ${REMOTE} && rm -f ${log} ${done} && \
      nohup sh -c '/workspace/venv/bin/wristview stage ${stage} \
        --run ${REMOTE}/data/${RUN_ID} > ${log} 2>&1; echo \$? > ${REMOTE}/stage${stage}.rc; \
        touch ${done}' > /dev/null 2>&1 < /dev/null & echo launched stage ${stage}"

  local waited=0
  while true; do
    sleep 30
    waited=$((waited + 30))
    local finished
    finished=$("${SSH[@]}" -n "test -f ${done} && echo done" 2>/dev/null)
    local last
    last=$("${SSH[@]}" -n "tail -1 ${log} 2>/dev/null | cut -c1-110" 2>/dev/null)
    printf '  [%4ds] %s\n' "$waited" "${last:-no output yet}"
    if [ "$finished" = "done" ]; then
      local rc
      rc=$("${SSH[@]}" -n "cat ${REMOTE}/stage${stage}.rc 2>/dev/null")
      echo "  remote stage ${stage} finished rc=${rc:-?} after ${waited}s"
      [ "${rc:-1}" = "0" ] || { echo "REMOTE STAGE ${stage} EXITED ${rc}" >&2; return 1; }
      break
    fi
  done

  # A stage that vanished without writing its meta is a failure, not a pass.
  if "${SSH[@]}" -n "grep -qE 'FAILED|Traceback' ${log}" 2>/dev/null; then
    echo "REMOTE STAGE ${stage} FAILED:" >&2
    "${SSH[@]}" -n "tail -30 ${log}" >&2
    return 1
  fi
  return 0
}

cmd_stage2() { remote_stage 2; }
cmd_stage4() { remote_stage 4; }
cmd_stage5() { remote_stage 5; }

# ---------------------------------------------------------------------------
# pull. Poses only.
#
# 02_localize also holds a per-clip features.h5 and matches.h5, 7.9 GB for 30
# clips, which are intermediates. Dragging those home would cost more than the
# stage saved.
# ---------------------------------------------------------------------------
cmd_pull() {
  need_run
  mkdir -p "$RUN/02_localize"
  "${SSH[@]}" "cd ${REMOTE}/data/${RUN_ID} && tar -cf - \
      \$(find 02_localize -name camera_poses.npy -o -name pose_valid.npy \
         -o -name status.json -o -name summary.json -o -name meta.json)" \
    | tar -xf - -C "$RUN"
  echo "pulled poses into $RUN/02_localize"
  find "$RUN/02_localize" -name camera_poses.npy | wc -l | xargs echo "clips:"
}

case "$VERB" in
  linktest) cmd_linktest ;;
  provision) cmd_provision ;;
  push) cmd_push ;;
  stage2) cmd_stage2 ;;
  stage4) cmd_stage4 ;;
  stage5) cmd_stage5 ;;
  pull) cmd_pull ;;
  all) cmd_linktest; cmd_provision; cmd_push; cmd_stage2 ;;
  *) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
