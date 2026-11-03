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

from dvs_sensor import DvsSensor
from functions.OMS_helpers import *
from functions.attention_simple_helpers import *
from controller.nengo_controller import NengoController
from controller.helper import plot_controller_response
from helpers import visualiser

# ---------------------------------------------------------------- settings
XML_PATH = "pan_tilt.xml"
WIDTH, HEIGHT = 320, 240
SIM_DT = 0.002          # physics timestep
RENDER_HZ = 300         # frame render rate fed into the sensor
SALIENCY_HZ = 30
DURATION = 4.0          # seconds
OUT_PATH = './out'

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

# OMS Parameters (unchanged from demo.py)
OMS_PARAMS = {
    'size_krn_center': 8,
    'sigma_center': 1,
    'size_krn_surround': 8,
    'size_krn_after_oms': 121,
    'sigma_surround': 4,
    'threshold': 0.6,
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

# Controller and pan/tilt settings
SIGN_TILT   = 1 # Positive tilt direction is down
SIGN_PAN    = -1 # Positive pan direction is left
FOVY = np.deg2rad(55) # matches fovy in pan_tilt.xml
# saliency map spans the image height
RAD_PER_PX_TILT = FOVY / OMS_PARAMS['size_krn_after_oms'] 
# Approcximation of horizontal FOV
RAD_PER_PX_PAN  = FOVY * (WIDTH / HEIGHT) / OMS_PARAMS['size_krn_after_oms'] 
MICRO_FREQ_HZ = 10
MICRO_AMP_DEG = 0.5
MICRO_PHASE = np.pi / 2

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
resolution = WIDTH, HEIGHT

frames_preview = []
event_frames = []  # per-frame accumulated event image, for a preview video
all_events = []     # list of (t, x, y, p)
sal_frames = []

# Same model initialisation as demo.py
net_center, net_surround = initialize_oms(device, OMS_PARAMS)
net_attention = initialise_attention(device, ATTENTION_PARAMS)

size_krn_after_oms = OMS_PARAMS['size_krn_after_oms']
saliency_map = np.zeros((size_krn_after_oms, size_krn_after_oms), dtype=np.float32)
salmax_coords = np.zeros((2,), dtype=np.int32)

# Panel size for visualizer
panel_size = 400

# ------------------------------- Controller Setup -------------------------------
kp = 0.5
k_pan = np.array([kp, 0, 0])
k_tilt = np.array([kp, 0, 0])
target_coords = np.array([size_krn_after_oms // 2, size_krn_after_oms // 2])
pan_target = 0.0     # rad, integrated state
tilt_target = 0.0

# print("Target coords: ", target_coords)

controller = NengoController(target_tilt=target_coords[0], target_pan=target_coords[1], k_pan=k_pan, k_tilt=k_tilt)

# ------------------------------- Diagnostics Varibles -------------------------------
latency = []
n_windows = 0
control = []
salmax = []

# ---------------------------------------------------------------- main loop
for i in range(n_frames):
    t = i * render_dt

    # Step the simulation
    for _ in range(steps_per_frame):
        mujoco.mj_step(model, data)

    # Render the scene
    renderer.update_scene(data, camera="event_cam")
    rgb = renderer.render()
    frames_preview.append(rgb.copy())

    # IEBCS rgb to events pipeline
    flux = to_flux(rgb)

    if i == 0:
        # IEBCS needs an initial condition frame before the first update() call
        dvs.init_image(flux)
        event_frames.append(np.zeros((HEIGHT, WIDTH), dtype=np.float32))
        continue


    ev_frame = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    ev = dvs.update(flux, frame_dt_us)  # <-- the live, per-frame call

    if ev:
        xs, ys, ps = ev.get_x(), ev.get_y(), ev.get_p()
        ts = ev.get_ts() * 1e-6  # us -> s

        for xi, yi, pi, ti in zip(xs, ys, ps, ts):
            polarity = 1 if pi else -1
            all_events.append((float(ti), int(xi), int(yi), polarity))
            ev_frame[yi, xi] = polarity

    event_frames.append(ev_frame)

    # frame_img_generater = False

    # ----- OMS and Atention -----
    if i % (RENDER_HZ // SALIENCY_HZ) == 0:
        n_frames_in_window = RENDER_HZ // SALIENCY_HZ

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
        # for _ in range(round(RENDER_HZ / SALIENCY_HZ)):
        #     sal_frames.append(frame)
        sal_frames.append(frame)
        n_windows += 1

        if n_windows % 20 == 0:
            logger.info(f"Processed {n_windows} windows | last salmax: {tuple(salmax_coords)}")

    # Save salmax_coords on every sim frame
    salmax.append(np.array(salmax_coords))

    # -------------------- Control and pan and tilt drive --------------------
    cmd = controller(np.array(salmax_coords))
    control.append(cmd)

    # Microsaccades
    wt = 2 * np.pi * MICRO_FREQ_HZ * t
    micro_pan  = np.deg2rad(MICRO_AMP_DEG) * np.cos(wt)
    micro_tilt = np.deg2rad(MICRO_AMP_DEG) * np.sin(wt + MICRO_PHASE)

    # Control
    ctrl_tilt = SIGN_TILT * cmd[0] * RAD_PER_PX_TILT
    ctrl_pan  = SIGN_PAN  * cmd[1] * RAD_PER_PX_PAN

    pan_target  = micro_pan
    tilt_target = micro_tilt

    if i % 50 == 0:
    # if False:
        pan_target  += ctrl_pan
        tilt_target += ctrl_tilt

    # if i == 200:
    #     data.ctrl[pan_act] += np.pi / 6

    data.ctrl[pan_act]  = np.clip(data.ctrl[pan_act] + pan_target,  np.deg2rad(-65), np.deg2rad(65))
    data.ctrl[tilt_act] = np.clip(data.ctrl[tilt_act] + tilt_target, np.deg2rad(-25), np.deg2rad(25))

    # ----- Latency logging -----
    #latency.append(time_module.time() - start)
    if i % 20 == 0:
        print(f"frame {i}/{n_frames}  t={t:.2f}s  events so far={len(all_events)}")

print(f"Total events generated: {len(all_events)}")

# ------------------- Save Outputs -------------------
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

# 3. Event map, OMS, Max Saliency
imageio.mimsave(f'{OUT_PATH}/saliency.mp4', sal_frames, fps=SALIENCY_HZ)

# 4. Raw event stream as a structured numpy array (t, x, y, polarity)
events_arr = np.array(
    all_events,
    dtype=[("t", "f8"), ("x", "i4"), ("y", "i4"), ("p", "i1")],
)
np.save("out/events.npy", events_arr)

print("Saved: out/rgb_preview.mp4, out/events_preview.mp4, "
      "out/accumulated_events.png, out/events.npy")

# -------------- Plot Controller Outputs --------------

print("Saliency Array:", np.array(salmax).shape)

plot_controller_response(
    salmax_arr=np.array(salmax), 
    control_arr=np.array([c[:2] for c in control]), 
    target_tilt=target_coords[0],
    target_pan=target_coords[1],
    controller_gain=kp,
    out_path=OUT_PATH
)
