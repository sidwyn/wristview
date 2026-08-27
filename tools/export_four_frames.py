import json
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

S=Path("/tmp/claude-501/-Users-sidwyn-Documents-Documents-Personal-Projects-atlas/4c7d9149-910a-457c-9850-0c4d7292cfb6/scratchpad")
run=S/"runs/real26bm"
ds=LeRobotDataset(repo_id="wristview/real26bm", root=S/"lrds26bm")
st=json.loads((run/"05_render/demo_0/status.json").read_text())
src=np.array(st["source_frame_index"])
tr=np.load(run/"04_retarget/demo_0/ee_trajectory.npz")
closed=tr["closed"].astype(bool)[src]
poses=tr["poses"][src]
plane=json.loads((run/"01_scene/scale.json").read_text())["diagnostics"]["desk_plane"]
n=np.array(plane["normal"])
d=float(plane["offset_m"])
op=np.load(run/"03_estimate/demo_0/object_pose.npy")
times=tr["timestamps_s"][src]
fps=float(tr["source_fps"])
demo_idx=np.clip(np.round(times*fps).astype(int),0,len(op)-1)
obj_h=(op[demo_idx][:,:3,3]@n)-d
k_contact=int(np.argmax(closed))
k_release=int(len(closed)-1-np.argmax(closed[::-1]))
k_lift=int(np.argmax(np.where(closed, obj_h, -np.inf)))
picks=[("first frame",0),("contact begins",k_contact),
       ("maximum lift",k_lift),("release",k_release)]
print("STAGE 3/4 SAY:")
for lab,k in picks:
    print(f"  {lab:16s} dataset index {k:3d}  object height {obj_h[k]*100:6.2f} cm  "
          f"gripper {'CLOSED' if closed[k] else 'open'}")
out=S/"fourframes_ego"
out.mkdir(exist_ok=True)
tiles=[]
for lab,k in picks:
    w=(ds[k]["observation.images.ego"].numpy().transpose(1,2,0)*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.putText(w,f"{lab}  idx {k}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2)
    cv2.putText(w,f"obj {obj_h[k]*100:.1f}cm  {'CLOSED' if closed[k] else 'open'}",
                (8,46),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),1)
    tiles.append(w)
cv2.imwrite(str(out/"four.png"), np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
print(f"\nwrote {out/'four.png'}")
