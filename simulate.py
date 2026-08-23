"""
Minimum viable event-camera-on-a-pan-tilt simulation using MuJoCo + IEBCS.

Pipeline:
  1. Drive a 2-DOF pan-tilt rig on a sinusoidal trajectory.
  2. Render RGB frames from the mounted camera at a fixed rate.
  3. Feed each frame into IEBCS's DvsSensor.update(frame, dt), one frame
     at a time -- a real per-pixel sensor model (latency, jitter,
     refractory period, threshold mismatch, measured sensor noise),
     called live in the same loop as the MuJoCo render, rather than a
     batch conversion of a pre-recorded video.
  4. Save: an RGB preview video, an accumulated event image, and the
     raw event stream as a .npy structured array (t, x, y, p).

Run:
    MUJOCO_GL=osmesa python3 simulate.py       # Linux headless
    python3 simulate.py                         # macOS / has a display
"""
import os
# os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless rendering; unset/cgl on macOS

import sys
import numpy as np
import mujoco
import imageio
import cv2

# Vendored IEBCS source (see external/IEBCS_src, external/IEBCS_data)
sys.path.append(os.path.join(os.path.dirname(__file__), "external", "IEBCS", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "external", "SpeckEgoMotion"))

from dvs_sensor import DvsSensor  # noqa: E402

from OMS_helpers import *
from attention_simple_helpers import *

# ---------------------------------------------------------------- settings
XML_PATH = "pan_tilt.xml"
WIDTH, HEIGHT = 320, 240
SIM_DT = 0.002          # physics timestep
RENDER_HZ = 100          # frame render rate fed into the sensor
DURATION = 4.0           # seconds
PAN_AMPLITUDE_DEG = 60
TILT_AMPLITUDE_DEG = 15
PAN_FREQ_HZ = 0.25
TILT_FREQ_HZ = 0.4

# IEBCS sensor parameters (values from the repo's own example)
TH_POS = 0.9          # ON threshold (log change)
TH_NEG = 0.9          # OFF threshold (log change)
TH_NOISE = 0.01       # threshold mismatch std dev
LATENCY_US = 100      # asymptotic latency (us)
TAU_US = 40           # log front-end time constant at 1 klux (us)
JITTER_US = 10        # temporal jitter std dev (us)
BGN_POS_HZ = 0.1      # ON background noise rate (events/pixel/s)
BGN_NEG_HZ = 0.01     # OFF background noise rate (events/pixel/s)
REFRACTORY_US = 100   # refractory period (us)
DATA_DIR = os.path.join(os.path.dirname(__file__), "external", "IEBCS", "data")

# ---------------------------------------------------------------- MuJoCo setup
model = mujoco.MjModel.from_xml_path(XML_PATH)
model.opt.timestep = SIM_DT
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

pan_act = model.actuator("pan_act").id
tilt_act = model.actuator("tilt_act").id

render_dt = 1.0 / RENDER_HZ
n_frames = int(DURATION * RENDER_HZ)
steps_per_frame = max(1, int(round(render_dt / SIM_DT)))
frame_dt_us = int(round(render_dt * 1e6))  # IEBCS wants dt in microseconds

# ---------------------------------------------------------------- IEBCS sensor setup
def to_flux(rgb_uint8):
    """Convert an RGB frame to the radiometric-ish flux units IEBCS expects,
    the same way the IEBCS example does: luma channel scaled so 255 -> 1e4
    (nominally '10 klux')."""
    luma = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LUV)[:, :, 0]
    return luma.astype(np.float64) / 255.0 * 1e4

dvs = DvsSensor("pan_tilt_event_cam")
dvs.initCamera(
    WIDTH, HEIGHT,
    lat=LATENCY_US, jit=JITTER_US, ref=REFRACTORY_US, tau=TAU_US,
    th_pos=TH_POS, th_neg=TH_NEG, th_noise=TH_NOISE,
    bgnp=BGN_POS_HZ, bgnn=BGN_NEG_HZ,
)
# Use measured real-sensor noise distributions instead of the simpler
# fixed-frequency noise model (comment out to use the simpler model instead)
dvs.init_bgn_hist(
    os.path.join(DATA_DIR, "noise_pos_161lux.npy"),
    os.path.join(DATA_DIR, "noise_neg_161lux.npy"),
)

# ---------------------------------------------------------------- main loop
frames_preview = []
event_frames = []  # per-frame accumulated event image, for a preview video
all_events = []     # list of (t, x, y, p)

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

    flux = to_flux(rgb)

    if i == 0:
        # IEBCS needs an initial condition frame before the first update() call
        dvs.init_image(flux)
        event_frames.append(np.zeros((HEIGHT, WIDTH), dtype=np.float32))
        continue

    ev = dvs.update(flux, frame_dt_us)  # <-- the live, per-frame call

    xs, ys, ps = ev.get_x(), ev.get_y(), ev.get_p()
    ts = ev.get_ts() * 1e-6  # us -> s

    ev_frame = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for xi, yi, pi, ti in zip(xs, ys, ps, ts):
        polarity = 1 if pi else -1
        all_events.append((float(ti), int(xi), int(yi), polarity))
        ev_frame[yi, xi] = polarity
    event_frames.append(ev_frame)

    if i % 20 == 0:
        print(f"frame {i}/{n_frames}  t={t:.2f}s  events so far={len(all_events)}")

print(f"Total events generated: {len(all_events)}")

# ---------------------------------------------------------------- save outputs
os.makedirs("out", exist_ok=True)

# 1. RGB preview video (what the camera actually saw)
imageio.mimsave("out/rgb_preview.mp4", frames_preview, fps=RENDER_HZ)

# 2. Event visualization video: red = ON events, blue = OFF events
event_vis = []
for ev_frame in event_frames:
    vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    vis[ev_frame > 0, 0] = 255
    vis[ev_frame < 0, 2] = 255
    event_vis.append(vis)
imageio.mimsave("out/events_preview.mp4", event_vis, fps=RENDER_HZ)

# 3. Single accumulated event image over the whole run
accum = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
acc_pos = np.zeros((HEIGHT, WIDTH))
acc_neg = np.zeros((HEIGHT, WIDTH))
for (ti, xi, yi, pi) in all_events:
    if pi > 0:
        acc_pos[yi, xi] += 1
    else:
        acc_neg[yi, xi] += 1
accum[..., 0] = np.clip(acc_pos * 4, 0, 255)
accum[..., 2] = np.clip(acc_neg * 4, 0, 255)
imageio.imwrite("out/accumulated_events.png", accum)

# 4. Raw event stream as a structured numpy array (t, x, y, polarity)
events_arr = np.array(
    all_events,
    dtype=[("t", "f8"), ("x", "i4"), ("y", "i4"), ("p", "i1")],
)
np.save("out/events.npy", events_arr)

print("Saved: out/rgb_preview.mp4, out/events_preview.mp4, "
      "out/accumulated_events.png, out/events.npy")
