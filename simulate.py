"""
Minimum viable event-camera-on-a-pan-tilt simulation using MuJoCo.

Pipeline:
  1. Drive a 2-DOF pan-tilt rig on a sinusoidal trajectory.
  2. Render RGB frames from the mounted camera at a fixed rate.
  3. Convert frames to log intensity and generate events with a simple
     ESIM-style per-pixel threshold model (no frame interpolation --
     good enough to see the concept working; swap in vid2e later for
     the full accurate model with adaptive upsampling).
  4. Save: an RGB preview video, an accumulated event image, and the
     raw event stream as a .npy structured array (t, x, y, p).

Run:
    MUJOCO_GL=osmesa python3 simulate.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless rendering, no GPU needed

import numpy as np
import mujoco
import imageio

# ---------------------------------------------------------------- settings
XML_PATH = "pan_tilt.xml"
WIDTH, HEIGHT = 320, 240
SIM_DT = 0.002          # physics timestep
RENDER_HZ = 100          # frame render rate fed into the event generator
DURATION = 4.0           # seconds
CONTRAST_THRESHOLD = 0.35  # log-intensity change needed to fire an event
PAN_AMPLITUDE_DEG = 60
TILT_AMPLITUDE_DEG = 15
PAN_FREQ_HZ = 0.25
TILT_FREQ_HZ = 0.4

# ---------------------------------------------------------------- setup
model = mujoco.MjModel.from_xml_path(XML_PATH)
model.opt.timestep = SIM_DT
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

pan_act = model.actuator("pan_act").id
tilt_act = model.actuator("tilt_act").id

render_dt = 1.0 / RENDER_HZ
n_frames = int(DURATION * RENDER_HZ)
steps_per_frame = max(1, int(round(render_dt / SIM_DT)))

# ---------------------------------------------------------------- helpers
def to_log_intensity(rgb_uint8, eps=1e-3):
    gray = rgb_uint8.astype(np.float32).mean(axis=-1) / 255.0
    return np.log(gray + eps)

# ---------------------------------------------------------------- main loop
frames_preview = []
event_frames = []  # per-frame accumulated event image, for a preview video
all_events = []     # list of (t, x, y, p)

ref_log = None  # per-pixel reference log-intensity since last event

for i in range(n_frames):
    t = i * render_dt

    # drive pan/tilt with a smooth trajectory
    pan_deg = PAN_AMPLITUDE_DEG * np.sin(2 * np.pi * PAN_FREQ_HZ * t)
    tilt_deg = TILT_AMPLITUDE_DEG * np.sin(2 * np.pi * TILT_FREQ_HZ * t)
    data.ctrl[pan_act] = np.deg2rad(pan_deg)
    data.ctrl[tilt_act] = np.deg2rad(tilt_deg)

    for _ in range(steps_per_frame):
        mujoco.mj_step(model, data)

    renderer.update_scene(data, camera="event_cam")
    rgb = renderer.render()
    frames_preview.append(rgb.copy())

    log_img = to_log_intensity(rgb)

    if ref_log is None:
        ref_log = log_img.copy()
        event_frames.append(np.zeros((HEIGHT, WIDTH), dtype=np.float32))
        continue

    diff = log_img - ref_log
    ev_frame = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    # positive events (brightness increased)
    n_pos = np.floor(diff / CONTRAST_THRESHOLD)
    pos_mask = n_pos >= 1
    # negative events (brightness decreased)
    n_neg = np.floor(-diff / CONTRAST_THRESHOLD)
    neg_mask = n_neg >= 1

    ys, xs = np.nonzero(pos_mask)
    for y, x in zip(ys, xs):
        k = int(n_pos[y, x])
        for _ in range(k):
            all_events.append((t, x, y, 1))
        ref_log[y, x] += k * CONTRAST_THRESHOLD
        ev_frame[y, x] = min(k, 3)

    ys, xs = np.nonzero(neg_mask)
    for y, x in zip(ys, xs):
        k = int(n_neg[y, x])
        for _ in range(k):
            all_events.append((t, x, y, -1))
        ref_log[y, x] -= k * CONTRAST_THRESHOLD
        ev_frame[y, x] = -min(k, 3)

    event_frames.append(ev_frame)

    if i % 20 == 0:
        print(f"frame {i}/{n_frames}  t={t:.2f}s  events so far={len(all_events)}")

print(f"Total events generated: {len(all_events)}")

# ---------------------------------------------------------------- save outputs
os.makedirs("out", exist_ok=True)

# 1. RGB preview video (what the camera actually saw)
imageio.mimsave("out/rgb_preview.mp4", frames_preview, fps=RENDER_HZ)

# 2. Event visualization video: red = positive events, blue = negative
event_vis = []
for ev in event_frames:
    vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    vis[ev > 0, 0] = 255       # positive -> red
    vis[ev < 0, 2] = 255       # negative -> blue
    event_vis.append(vis)
imageio.mimsave("out/events_preview.mp4", event_vis, fps=RENDER_HZ)

# 3. Single accumulated event image over the whole run (like a classic event-camera plot)
accum = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
acc_pos = np.zeros((HEIGHT, WIDTH))
acc_neg = np.zeros((HEIGHT, WIDTH))
for (t, x, y, p) in all_events:
    if p > 0:
        acc_pos[y, x] += 1
    else:
        acc_neg[y, x] += 1
accum[..., 0] = np.clip(acc_pos * 40, 0, 255)
accum[..., 2] = np.clip(acc_neg * 40, 0, 255)
imageio.imwrite("out/accumulated_events.png", accum)

# 4. Raw event stream as a structured numpy array (t, x, y, polarity)
events_arr = np.array(
    all_events,
    dtype=[("t", "f8"), ("x", "i4"), ("y", "i4"), ("p", "i1")],
)
np.save("out/events.npy", events_arr)

print("Saved: out/rgb_preview.mp4, out/events_preview.mp4, "
      "out/accumulated_events.png, out/events.npy")
