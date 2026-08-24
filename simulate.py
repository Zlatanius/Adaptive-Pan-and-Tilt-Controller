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
import sys
import numpy as np
import mujoco
import imageio
import cv2
import math
from loguru import logger

# Vendored IEBCS source (see external/IEBCS_src, external/IEBCS_data)
sys.path.append(os.path.join(os.path.dirname(__file__), "external", "IEBCS", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "external", "SpeckEgoMotion", "functions"))

from dvs_sensor import DvsSensor  # noqa: E402

from OMS_helpers import *
from attention_simple_helpers import *

# ---------------------------------------------------------------- settings
XML_PATH = "pan_tilt.xml"
WIDTH, HEIGHT = 320, 240
SIM_DT = 0.002          # physics timestep
RENDER_HZ = 300         # frame render rate fed into the sensor
SALIENCY_HZ = 30
DURATION = 8.0          # seconds
PAN_AMPLITUDE_DEG = 0.2
TILT_AMPLITUDE_DEG = 0.2
PAN_FREQ_HZ = 20
TILT_FREQ_HZ = 20
OUT_PATH = './out/saliency.mp4'

# IEBCS sensor parameters (values from the repo's own example)
TH_POS = 0.2          # ON threshold (log change)
TH_NEG = 0.2          # OFF threshold (log change)
TH_NOISE = 0.01       # threshold mismatch std dev
LATENCY_US = 100      # asymptotic latency (us)
TAU_US = 40           # log front-end time constant at 1 klux (us)
JITTER_US = 10        # temporal jitter std dev (us)
BGN_POS_HZ = 0.1      # ON background noise rate (events/pixel/s)
BGN_NEG_HZ = 0.01     # OFF background noise rate (events/pixel/s)
REFRACTORY_US = 100   # refractory period (us)
DATA_DIR = os.path.join(os.path.dirname(__file__), "external", "IEBCS", "data")


# MAX_X / MAX_Y are set dynamically in main() to match the resolution
# of the loaded .npy file (they only matched 128x128 by coincidence
# when using the physical 128x128 Speck sensor).
MAX_X, MAX_Y = WIDTH, HEIGHT

# TODO: Change OMS_PARAMS and ATTENTION_PARAMS to be an object or sepparate params

# OMS Parameters (unchanged from demo.py)
OMS_PARAMS = {
    'size_krn_center': 8,
    'sigma_center': 1,
    'size_krn_surround': 8,
    'size_krn_after_oms': 121,
    'sigma_surround': 4,
    'threshold': 0.46,
    'tau_memOMS': 0.3,
    'sc': 1,
    'ss': 1
}

# Attention Parameters (unchanged from demo.py)
ATTENTION_PARAMS = {
    'size_krn': 10,
    'r0': 4,
    'rho': 0.1,
    'theta': np.pi * 3 / 2,
    'thetas': np.arange(0, 2 * np.pi, np.pi / 4),
    'thick': 3,
    'offsetpxs': 0,
    'offset': (0, 0),
    'fltr_resize_perc': [2, 2],
    'num_pyr': 3,
    'tau_mem': 0.3,
    'stride': 1,
    'out_ch': 1
}

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

# ---------------------------------------------------------------- Attention / OMS setup

def visualiser(window, OMS, saliency_map, salmax_coords, panel_size=200):
    """
    Trimmed-down version of demo.py's visualiser(): drops the control-loop
    panel (cmd / pan-tilt target dot) since there is no controller here.
    Shows: raw events | saliency map | OMS map, with the saliency peak marked.

    Each input array (window, saliency_map, OMS) may come out at a *different*
    resolution depending on the source data (unlike demo.py's fixed 128x128
    sensor, here they depend on whatever resolution the .npy file has and how
    egomotion()/run_attentiondemo() reshape it). So every panel is explicitly
    resized to `panel_size` x `panel_size` before being combined -- this
    normalises the display and also fixes the salmax_coords marker position,
    which is rescaled into the same panel coordinates.
    """
    h = w = panel_size
    combined = np.full((h + 20, w * 3 + 4, 3), fill_value=255, dtype=np.uint8)

    def to_rgb(img, orig_shape=None):
        img = np.asarray(img, dtype=np.float32)

        img_min, img_max = img.min(), img.max()

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255
        else:
            img = np.zeros_like(img)

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img[:, :, None].repeat(3, axis=2)

    combined[10:10 + h, 1:1 + w, :] = to_rgb(window)
    combined[10:10 + h, w + 2:w + 2 + w, :] = to_rgb(saliency_map)
    combined[10:10 + h, 2 * w + 3:2 * w + 3 + w, :] = to_rgb(OMS)

    # salmax_coords are in saliency_map's original coordinate space --
    # rescale them into the resized panel before drawing
    sal_h, sal_w = np.asarray(saliency_map).shape
    scaled_y = int(salmax_coords[0] * h / sal_h)
    scaled_x = int(salmax_coords[1] * w / sal_w)
    cv2.circle(
        combined,
        (w + 2 + scaled_x, 10 + scaled_y),
        4, (0, 255, 0), -1
    )

    cv2.putText(combined, 'Events', (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(combined, 'OMS', (2 * w + 8, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(combined, 'Saliency', (w + 7, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return combined

# ---------------------------------------------------------------- main loop
frames_preview = []
event_frames = []  # per-frame accumulated event image, for a preview video
all_events = []     # list of (t, x, y, p)

# ----- OMS setup -----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# TODO: check if the order of WDITH and HEIGHT is correct
resolution = WIDTH, HEIGHT

# Same model initialisation as demo.py
net_center, net_surround = initialize_oms(device, OMS_PARAMS)
net_attention = initialise_attention(device, ATTENTION_PARAMS)

# TODO: replace size_krn_after_oms with a global variable
size_krn_after_oms = OMS_PARAMS['size_krn_after_oms']
saliency_map = np.zeros((size_krn_after_oms, size_krn_after_oms), dtype=np.float32)
salmax_coords = np.zeros((2,), dtype=np.int32)

# Video writer so results are inspectable without a live display.
# panel_size must match the value used in visualiser() below.
panel_size = 400
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_w = panel_size * 3 + 4
out_h = panel_size + 20
writer = cv2.VideoWriter(OUT_PATH, fourcc, SALIENCY_HZ, (out_w, out_h))

latency = []
n_windows = 0
control = []
salmax = []

for i in range(n_frames):
    t = i * render_dt

    # drive pan/tilt with a smooth trajectory
    pan_deg = PAN_AMPLITUDE_DEG * np.sin(2 * np.pi * PAN_FREQ_HZ * t)
    tilt_deg = -TILT_AMPLITUDE_DEG * np.sin(2 * np.pi * TILT_FREQ_HZ * t)
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

    frame_img_generater = False

    # ----- OMS and Atention -----
    if i % SALIENCY_HZ == 0:
        n_frames_in_window = math.ceil(RENDER_HZ / SALIENCY_HZ)

        window_frame = np.sum(event_frames[-n_frames_in_window:], axis=0)
        window_frame = abs(window_frame)
        window_frame *= 255

        #frame_vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        #window_vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        #frame_vis[ev_frame > 0] = (0, 0, 255)
        #frame_vis[ev_frame < 0] = (255, 0, 0)

        #window_vis[window_frame > 0] = (0, 0, 255)
        #window_vis[window_frame < 0] = (255, 0, 0)

        #if not frame_img_generater:
        #    cv2.imwrite("out/event_window_ss.png", window_vis)
        #    cv2.imwrite("out/event_frame_ss.png", frame_vis)

        wOMS = torch.tensor(window_frame, dtype=torch.float32).unsqueeze(0).to(device)

        # ----- OMS (unchanged from demo.py) -----
        OMS, indexes = egomotion(
            wOMS, net_center, net_surround, device,
            MAX_Y, MAX_X, OMS_PARAMS['threshold']
        )
        vSliceOMS = OMS.squeeze(0)
        OMS_np = OMS.squeeze(0).squeeze(0).cpu().detach().numpy()

        # ----- Attention / saliency (unchanged from demo.py) -----
        sal, sal_max = run_attentiondemo(
            vSliceOMS, net_attention, device, resolution,
            size_krn_after_oms, num_pyr=ATTENTION_PARAMS['num_pyr']
        )
        saliency_map[:] = sal
        salmax_coords[:] = sal_max

        frame = visualiser(window_frame, OMS_np, saliency_map, salmax_coords, panel_size=panel_size)
        writer.write(frame)
        n_windows += 1

        if n_windows % 20 == 0:
            logger.info(f"Processed {n_windows} windows | last salmax: {tuple(salmax_coords)}")


    # ----- Control logging and plotting -----
    #cmd = controller(np.array(salmax_coords))
    #control.append(cmd)
    #salmax.append(np.array(salmax_coords))

    # ----- Latency logging -----
    #latency.append(time_module.time() - start)
    if i % 20 == 0:
        print(f"frame {i}/{n_frames}  t={t:.2f}s  events so far={len(all_events)}")

writer.release()
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

# 3. Raw event stream as a structured numpy array (t, x, y, polarity)
events_arr = np.array(
    all_events,
    dtype=[("t", "f8"), ("x", "i4"), ("y", "i4"), ("p", "i1")],
)
np.save("out/events.npy", events_arr)

print("Saved: out/rgb_preview.mp4, out/events_preview.mp4, "
      "out/accumulated_events.png, out/events.npy")
