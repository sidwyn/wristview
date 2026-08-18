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
| 2026-08-17 12:32 | **CREATE POD** | authorised for the real06 splat run | Export built and verified locally first, sha f7c621ee1e5d, so the machine computes from the first minute. |
| 2026-08-17 12:45 | **stop** | SSH never came up; nothing was trained | Pod `974hngj5oitlf3`. |
| 2026-08-17 13:01 | **stop** | pod exposed no public TCP port, so SSH was never possible | Pod `974hngj5oitlf3`, datacenter IS. The PORTS column stayed empty for the whole 13 minute wait, where the one pod that worked (`avhwcfugzxysm3`, datacenter US) showed `209.170.80.132:14047->22 (pub,tcp)` within a minute. This is placement, not authentication: `--ports "22/tcp"` is a request, and a host without a free public port silently does not honour it. Stopped rather than left billing. Four pods have now failed to become reachable; not creating a fifth without a decision. |
| 2026-08-17 13:11 | **CREATE POD** | authorised real06 splat, datacenter US-CA-2 | Pod `992frbsvsko0mf`. Earlier pods failed because they got no public TCP port; this run pins the datacenter and gives up fast. |
| 2026-08-17 13:13 | **stop** | no public TCP port appeared | Pod `992frbsvsko0mf` in US-CA-2. Same failure as 974hngj5oitlf3. |
| 2026-08-17 13:13 | **CREATE POD** | authorised real06 splat, datacenter US-IL-1 | Pod `087h1slnbh3pe2`. Earlier pods failed because they got no public TCP port; this run pins the datacenter and gives up fast. |
| 2026-08-17 13:16 | **stop** | no public TCP port appeared | Pod `087h1slnbh3pe2` in US-IL-1. Same failure as 974hngj5oitlf3. |
| 2026-08-17 13:16 | **CREATE POD** | authorised real06 splat, datacenter US-NC-1 | Pod `2b401fw65oxmzq`. Earlier pods failed because they got no public TCP port; this run pins the datacenter and gives up fast. |
| 2026-08-17 13:18 | **stop** | no public TCP port appeared | Pod `2b401fw65oxmzq` in US-NC-1. Same failure as 974hngj5oitlf3. |
| 2026-08-17 13:20 | **start** | user supplied the SSH proxy address; public TCP ports were unavailable in every datacenter tried | Pod `2b401fw65oxmzq`. Proxy route `2b401fw65oxmzq-64410b4a@ssh.runpod.io` needs no public port. |
| 2026-08-17 14:28 | **stop** | run complete, splat pulled and hash-verified first | Pod `2b401fw65oxmzq`. 2,734,543 Gaussians, 645,354,499 bytes, sha 358da1d6, verified on both ends before stopping. Pod had no network volume, so nothing was left behind. |
| 2026-08-17 14:49 | **start** | all-views PSNR eval, so real03 and real06 can be compared by the same method | Pod `2b401fw65oxmzq`. Stop wiped /root, so the payload and splat must be re-sent: about 15 minutes, not 5. |
| 2026-08-17 14:52 | note | new pod supplied by user; pods now stay up | Pod `rssn54xdtrcybs`. User has budget for ~5 hours, so the stop-after-each-job rule is lifted for now. That also removes the container-wipe problem: nothing needs re-sending between jobs. |
| 2026-08-17 16:20 | **SSH BLOCKED** | cannot reach pod `rssn54xdtrcybs` by either route | Proxy `rssn54xdtrcybs-64410bcd@ssh.runpod.io` rejects `~/.ssh/id_ed25519` with `Permission denied (publickey)`; accepts `~/.ssh/runpod` then hangs indefinitely on `Error: Your SSH client doesn't support PTY`, with and without `-tt` and closed stdin. Direct public port `213.173.103.155:34125` rejects both keys. Pod is RUNNING, 5512s uptime, $0.74/hr. Left running because the user lifted the stop rule and the render still needs it. Asked the user for help rather than trying a sixth variation. |
