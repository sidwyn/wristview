# real26 splat training on RunPod

The payload is built and hashed. Nothing here creates a pod. Start the pod when
you are ready, then work down the list.

## What is being sent

| Item | Size |
|---|---|
| 298 scan frames, 1920x1080 JPEG, exactly as the camera wrote them | 140 MB |
| COLMAP sparse model: cameras.bin, images.bin, points3D.bin | 9.7 MB |
| `train_gsplat.py` and `undistort_export.py` | 28 KB |
| config.yaml, export.json | 8 KB |
| **Archive total** | **150.4 MB** |

```
/private/tmp/claude-501/-Users-sidwyn-Documents-Documents-Personal-Projects-atlas/4c7d9149-910a-457c-9850-0c4d7292cfb6/scratchpad/exports/real26-splat.tar.gz
150,405,158 bytes
sha256 af4ce0e78cb77e0e0d7d2d023e1d8b19b8118497452c3a9627347fa4ea6fff5e
```

No video is in the archive. Checked, count zero.
SfM is not re-run on the pod. The reconstruction ships as binaries and is read,
not rebuilt.

The images ship distorted and the pod undistorts them to PNG. That avoids a
JPEG re-encode. Measured on this capture, the undistortion changes the image by
40.65 dB and a quality-95 re-encode damages it by 45.29 dB, so shipping
undistorted JPEGs would give back a third of what the correction buys, and add
94 MB.

## Which GPU

**RTX 4090, 24 GB.** Do not take more.

Memory at full 1920x1080, 298 images:

| Item | Memory |
|---|---|
| Image cache on GPU | 7.4 GiB |
| 2 M Gaussians, parameters plus Adam state | 1.3 GiB |
| Rasteriser working set, `packed=True` | 2 to 4 GiB |
| **Total** | **11 to 13 GiB** |

24 GB leaves headroom. A 16 GB card also fits, with less room; add
`--resolution 1600` there and the cache drops to 5.1 GiB. An 80 GB A100 is
over-provisioned for this and costs several times more.

**Wall clock:** 35 to 60 minutes for 30,000 steps. Add about 10 minutes for
`pip install` and the gsplat CUDA kernel build, 2 minutes for undistortion, and
a few minutes each way for transfer. Budget **1.5 hours of pod time**.

**Cost:** the real06 run recorded 4090 pricing at $0.40 to $0.80 an hour, so
**about $1.00 to $1.20**. That is from the earlier run's record, not live
pricing. Check the rate the console quotes you before you accept it.

---

## 1. Local: the archive already exists

It was built and hashed above. Confirm it is intact before you start paying:

```bash
cd /private/tmp/claude-501/-Users-sidwyn-Documents-Documents-Personal-Projects-atlas/4c7d9149-910a-457c-9850-0c4d7292cfb6/scratchpad/exports
shasum -a 256 real26-splat.tar.gz
# expect af4ce0e78cb77e0e0d7d2d023e1d8b19b8118497452c3a9627347fa4ea6fff5e
```

## 2. Start the pod

RTX 4090, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 60 GB
container disk, no network volume. Open the web terminal.

## 3. Send the archive

`runpodctl` avoids SSH keys entirely. Run this on the laptop:

```bash
cd /private/tmp/claude-501/-Users-sidwyn-Documents-Documents-Personal-Projects-atlas/4c7d9149-910a-457c-9850-0c4d7292cfb6/scratchpad/exports
runpodctl send real26-splat.tar.gz
```

It prints a one-time code, for example `8243-galileo-cobra-nixon-4`.
Type this in the pod's web terminal, with that code:

```bash
cd /root
runpodctl receive 8243-galileo-cobra-nixon-4
sha256sum real26-splat.tar.gz
# must read af4ce0e78cb77e0e0d7d2d023e1d8b19b8118497452c3a9627347fa4ea6fff5e
tar -xzf real26-splat.tar.gz
```

Stop if the hash differs. Send it again rather than training on a damaged file.

## 4. Install, on the pod

```bash
pip install gsplat opencv-python-headless numpy
python -c "import gsplat, torch; print(gsplat.__version__, torch.cuda.get_device_name(0))"
```

The first `import gsplat` builds CUDA kernels and takes a few minutes.

## 5. Undistort, on the pod

```bash
cd /root/real26
python job/undistort_export.py --export /root/real26 --format png
```

Expect `largest correction 0.90 px at the frame border` and 298 PNGs.

## 6. Probe before the long run

Densification silently failed to run on an earlier GPU job and 28,000 steps did
nothing. Confirm growth for a few cents first:

```bash
cd /root/real26
python job/train_gsplat.py --data /root/real26 --out /tmp/probe \
    --iterations 2500 --resolution 1920
```

Look for two lines:

- `using undistorted images and pinhole intrinsics from cameras_pinhole.json`
- `densification confirmed: 8810 -> N Gaussians by step 2000`

If either is missing, stop and say so. Do not start step 7.

## 7. Train

30,000 steps. The densification schedule is real06's, and it is the trainer's
default, so it needs no flags: `refine_start_iter` 500, `refine_stop_iter`
15,000 (half of iterations), `refine_every` 100, `reset_every` 3,000,
`prune_opa` 0.005, `grow_grad2d` 2e-4.

```bash
cd /root/real26
nohup python job/train_gsplat.py --data /root/real26 --out /root/result \
    --iterations 30000 --resolution 1920 > /root/train.log 2>&1 &
tail -f /root/train.log
```

Watch the Gaussian count. real06 finished near 2.7 M. A count still in the tens
of thousands at step 5,000 means something is wrong.

## 8. Bring the splat back

On the pod:

```bash
ls -l /root/result/splat.pt
sha256sum /root/result/splat.pt
runpodctl send /root/result/splat.pt
```

On the laptop, with the code it prints:

```bash
cd /private/tmp/claude-501/-Users-sidwyn-Documents-Documents-Personal-Projects-atlas/4c7d9149-910a-457c-9850-0c4d7292cfb6/scratchpad
runpodctl receive <code>
shasum -a 256 splat.pt      # must match what the pod printed
```

Also pull the log, it is small and it is the training record:

```bash
# on the pod
runpodctl send /root/train.log
```

Verify the hash before you stop the pod, not after.

## 9. Convert and gate, on the laptop

```bash
cd "/Users/sidwyn/Documents/Documents/Personal Projects/atlas/wristview"
source .venv/bin/activate
S=/private/tmp/claude-501/-Users-sidwyn-Documents-Documents-Personal-Projects-atlas/4c7d9149-910a-457c-9850-0c4d7292cfb6/scratchpad

PYTHONPATH=src python -m tools.convert_gsplat_checkpoint \
    $S/splat.pt $S/runs/real26/01_scene/splat_gpu.pt

PYTHONPATH=src python -m tools.check_splat \
    --splat $S/runs/real26/01_scene/splat_gpu.pt \
    --run $S/runs/real26 --views 12 \
    --stills $S/gate_gpu --report $S/gate_gpu.json
```

The gate exits 1 and prints `GATE FAILED` if the splat cannot redraw its own
training views. It needs 25 dB median and 200 Gaussians per view. The current
local splat scores 17.42 dB and 35 per view, and fails on both.

**If the gate fails, stop. Do not render.**
