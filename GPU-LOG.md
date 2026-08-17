# GPU pod usage log

Every start and stop, with a timestamp and a reason, so the bill can be
reconstructed. Times are local (PDT).

Pods: `yvll07diewht6x` is the live one, a migration of `xaxa1v6y92ryv2`, which
is EXITED and kept only for its disk. **Stop, never remove or terminate.**

| When | Action | Reason | Notes |
|---|---|---|---|
| 2026-08-16 08:34 | **stop** | found RUNNING with no job on it | Inherited running from the previous session. real04 Stage 1 is still matching locally, so the export is many minutes away. Nothing to compute. |
| 2026-08-16 08:36 | note | pod inventory | `yvll07diewht6x` VOLUME DISK 0, so no network volume: `/workspace` does not carry over from `xaxa1v6y92ryv2`. All scripts and exports are pushed fresh from this machine, so nothing depends on the old disk. |
| 2026-08-16 08:58 | **TERMINATE + volume delete** | user instruction; EXITED pod still billing 50 GB volume + 30 GB container disk | Pod `xaxa1v6y92ryv2` terminated and network volume `i7geqra7ab` (50 GB, EU-RO-1) removed. Verified first that `artifacts/splat_30k.pt` is intact locally: 361,101,827 bytes, sha256 d0af10d4… matches. Lost with it: `result2_distorted/splat.pt` (the intermediate comparison splat) and the raw training logs; their numbers are already recorded in BUILD-PLAN.md. |
| 2026-08-16 13:20 | **stop** | idle ~4.4 h (~$3.25) with no job and SSH never reachable | Pod `hp12k8b3du571v`. Started 15:57 UTC, never accepted SSH on either route (proxy "container not found", direct TCP refused the key) despite `runtime=running`. Export not built, so nothing to compute. Stopping per the lifecycle rule; the new ID/port will be re-read with `runpodctl pod list` before the next job. |
| 2026-08-16 13:40 | **TERMINATE** | user confirmed; needed to detach the 50 GB volume | Pod `yvll07diewht6x` (dead intermediate). Stopping does not detach a network volume; only deleting the pod does. |
| 2026-08-16 13:40 | note | volume NOT released | `yvll07diewht6x` deleted, but volume `i7geqra7ab` (50 GB) still refuses: the remaining holder is `hp12k8b3du571v`, the active pod. So `/workspace` on it IS this network volume; the `VOLUME DISK 0` column does not show network volumes. Releasing the 50 GB requires terminating `hp12k8b3du571v`, and re-creating a pod is a user decision. Blocked pending that call. |

## POD CREATION

| 2026-08-16 16:13 | **CREATE POD** | explicitly authorised by the user for the real04 splat run only | RTX 4090, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 60 GB container disk, **no network volume attached** so nothing new starts billing storage. Pod id `avhwcfugzxysm3`. Export was built and verified locally first (170 MB payload, sha256 500d68e6...), so the machine computes from the first minute. |
| 2026-08-16 16:40 | **stop+start** | SSH key rejected after working; container lost | Pod `avhwcfugzxysm3`. scp of the 170 MB payload succeeded and the hash verified, then SSH began returning "Permission denied (publickey)" and the proxy "container not found", while the API still reported `runtime=running`. Same failure as `hp12k8b3du571v`. Cycling once to force key re-injection; container disk survives a stop, so the payload should remain. |
| 2026-08-16 17:01 | **stop** | SSH auth failed twice on the same pod; rule says stop rather than think while billing | Pod `avhwcfugzxysm3`. Pattern across three pods: SSH accepts the key for a few minutes after start, then returns "Permission denied (publickey)" while the API still reports `runtime=running`. Also confirmed: with no network volume, a stop wipes `/root` entirely, so the 170 MB payload and the gsplat install were lost on the first cycle. No training run was completed. |
| 2026-08-16 17:03 | **start** | user added the public key to the RunPod account; retrying the real04 splat | Pod `avhwcfugzxysm3`. Testing whether SSH auth now survives past the first few minutes, which is what failed three times before. |
