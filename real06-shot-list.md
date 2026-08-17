# real06 · Bimanual shot list

**Goal:** two synchronised wrist-camera views from one head camera. Both hands doing real work.

Everything below is one session. Nothing on the desk moves from the first frame of the scan to the last frame of the final demo.

---

## Before you press record

### The surface

- Patterned mat, newspaper page, or book cover under the working area. **No bare wood.**
- Four to six small static props around the edge, out of the hand path. They never move.
- Bright, diffuse light. No hard shadows, no glare.

### The marker

- ArUco `DICT_4X4_50`, id 0, 100 mm black square.
- Flat, **inside the working area**, 5 to 10 cm from where the objects sit.
- It must stay visible in the close scan passes AND the demo frames. Last session it was in 21% of scan frames. Aim for most of them.

### The objects

Matte, printed, rigid, 5 to 8 cm. No gloss, no dark plastic, no deep concave shapes.

You need **three**:

- Object A — the block you pick with the right hand.
- Object B — the block you pick with the left hand.
- Object C — a rigid container or box you can hold and place things into.

### One tip that costs nothing

**Put a coloured band on your left wrist.** It gives the tracker a hard cue for hand identity, which is the thing most likely to break when two hands are in frame.

### Camera

Blackmagic Camera. Main wide lens only, no zoom. Stabilisation off. 1080p60. Shutter 1/120. Lock ISO, white balance, focus. Head mount on.

---

## The scan · 90 seconds, one continuous take

**Change from last time: start close and work outward.** You have under-shot the close pass twice, because it came last. Now it comes first, while you are paying attention.

**Distance gauge:** 0.15 m is about **one phone length** from the surface. Check it once with your hand before you record.

| Order | Time | Pass | What to do |
|---|---|---|---|
| 1 | **40 s** | **Wrist rehearsal, 0.15 m** | Trace the path your hands will take. Hover over each object. Circle each one at close range, dipping low enough to see its sides. Visit both pick spots and both place spots. |
| 2 | 30 s | Close orbit, 0.20 to 0.25 m | Circle the whole working area, looking down at 45 to 60 degrees. |
| 3 | 20 s | Wide orbit, 0.5 m | Context and background only. |

**Rules:** never stop recording. Move slowly and smoothly. Keep the working area in frame throughout. Move outward gradually so consecutive frames always overlap.

**Count the 40 seconds out loud, or set a timer.** It always feels longer than it is.

---

## The demos · five clips, 15 seconds each

Ordered easiest to hardest **for the pipeline**. Shoot them in this order, so if something breaks you still have the early ones.

Every clip starts with all objects fully in frame and **both hands out of shot**.

### Clip 1 · Parallel pick and place · easiest

Both hands reach in at the same time. Right hand takes object A, left hand takes object B. Both lift about 10 cm, move to two separate spots, place, hold one second, withdraw.

**Why first:** the hands never touch, never cross, never occlude each other. This is the cleanest possible bimanual case, and it is the one most likely to work.

### Clip 2 · Hold and insert

Left hand picks up container C and holds it steady in the air. Right hand picks up object A and places it inside. Both hands withdraw.

**Why:** one hand stabilises, one hand acts. This is the classic bimanual pattern and it is what buyers mean by bimanual.

### Clip 3 · Handoff

Right hand picks up object A. Both hands meet in the middle. Object A passes from right hand to left. Left hand places it down.

**Why:** tests whether the grasp detector can handle release by one hand and acquisition by the other, in the same moment. Expect this one to be hard.

### Clip 4 · Hold and open

Left hand holds container C on the desk. Right hand lifts the lid off and sets it aside.

**Why:** a rigid two-part object, and a genuine force interaction. If your container has no lid, use a book and open the cover instead.

### Clip 5 · Single-hand control

Right hand only. Pick up object A, move, place. Left hand **completely out of frame**.

**Why:** this is your control. It is the same task as `real04`, so it lets you compare the bimanual result against a known-good single-hand result from the same scan.

---

## Rules for every demo

- **Both hands out of frame at the start.** Objects fully visible. Then reach in.
- **Grab by the body**, thumb opposed to fingers, so finger separation visibly closes. Not by a handle. Not a full wrap.
- **Hold still for one second after contact**, and again after placing. This gives the grasp detector a stable window.
- **Move slowly.** 15 seconds, not 8. Slow reach, deliberate grasp, slow transport, deliberate release.
- **Do not let your hands cross or overlap** except in clip 3, where crossing is the point.
- Nothing on the desk moves between clips. Return objects to their exact start positions between takes.

---

## Before you clear the desk

**Do not pack up until the pipeline has passed end to end.** This cost you the option of a supplementary close pass last session.

Run the pre-flight on the new scan with clip 1 before you shoot the rest. If it fails, you have lost 15 seconds instead of five clips.

---

## What to tell Claude Code

> `real06`, bimanual. One scan, five demos. Scan is ~90 s, one continuous take, ordered close to far: 40 s wrist rehearsal at 0.15 m, 30 s close orbit at 0.20 to 0.25 m, 20 s wide orbit at 0.5 m.
>
> Five demo clips, 15 s each: 1 parallel pick and place, 2 hold and insert, 3 handoff, 4 hold and open, 5 single-hand control. I wear a coloured band on my left wrist as a hand-identity cue.
>
> Run the pre-flight on clip 1 first and stop if it fails.
>
> Then track **both** hands, retarget each to its own end effector, and render two wrist views from the same splat — left and right, sharing scene, pose and metric scale. Report separately for each hand: tracking rate, grasp detection, roll flips, and marker-versus-pose agreement.
>
> Tell me what breaks: hand identity across frames, left-versus-right assignment, occlusion when the hands cross in clip 3, and whether the grasp detector handles two objects at once. Clip 5 is the single-hand control — its result should match `real04`, and if it does not, the bimanual change broke something.
