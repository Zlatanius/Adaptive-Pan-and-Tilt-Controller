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
import torch
from tqdm import tqdm
from loguru import logger

from controller.nengo_controller import NengoController
from controller.helper import plot_controller_response
from helpers import visualiser

# ---------------------------------------------------------------- settings
XML_PATH = "pan_tilt.xml"
WIDTH, HEIGHT = 320, 320
FRAME_SIZE = HEIGHT     # assuming equalt height and width for now
SIM_DT = 0.002          # physics timestep
RENDER_HZ = 300         # frame render rate fed into the sensor
SALIENCY_HZ = 30
DURATION = 4.0          # seconds
OUT_PATH = './out'

# MAX_X / MAX_Y are set dynamically in main() to match the resolution
# of the loaded .npy file (they only matched 128x128 by coincidence
# when using the physical 128x128 Speck sensor).
MAX_X, MAX_Y = WIDTH, HEIGHT

# Controller and pan/tilt settings
SIGN_TILT   = 1 # Positive tilt direction is down
SIGN_PAN    = -1 # Positive pan direction is left
FOVY = np.deg2rad(55) # matches fovy in pan_tilt.xml
# saliency map spans the image height
RAD_PER_PX_TILT = FOVY / FRAME_SIZE
# Approcximation of horizontal FOV
CONTROL_HZ = 100
RAD_PER_PX_PAN  = FOVY * (WIDTH / HEIGHT) / FRAME_SIZE 
USE_MICRO_SACC = 0      # 1 use microsaccades; 0 don't use mircro saccades
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
frames_preview = []

# ------------------------------- Controller Setup -------------------------------
kp = 0.5
k_pan = np.array([kp, 0, 0])
k_tilt = np.array([kp, 0, 0])
center_coords = np.array([FRAME_SIZE // 2, FRAME_SIZE // 2])
pan_cmd = 0.0     # rad, integrated state
tilt_cmd = 0.0

controller = NengoController(target_tilt=center_coords[0], target_pan=center_coords[1], k_pan=k_pan, k_tilt=k_tilt)

# ------------------------------- Diagnostics Varibles -------------------------------
latency = []
n_windows = 0
control = []
measured_positions = []

# ---------------------------------------------------------------- main loop
for i in tqdm(range(n_frames), unit="frames", ncols=100):
    t = i * render_dt

    # Step the simulation
    for _ in range(steps_per_frame):
        mujoco.mj_step(model, data)

    # Render the scene
    renderer.update_scene(data, camera="event_cam")
    rgb = renderer.render()
    frames_preview.append(rgb.copy())

    # -------------------- Control and pan and tilt drive --------------------
    cmd = controller(center_coords)
    control.append(cmd)

    # Microsaccades
    wt = 2 * np.pi * MICRO_FREQ_HZ * t
    micro_pan  = np.deg2rad(MICRO_AMP_DEG) * np.cos(wt)
    micro_tilt = np.deg2rad(MICRO_AMP_DEG) * np.sin(wt + MICRO_PHASE)

    # Control
    ctrl_tilt = SIGN_TILT * cmd[0] * RAD_PER_PX_TILT
    ctrl_pan  = SIGN_PAN  * cmd[1] * RAD_PER_PX_PAN

    pan_cmd  = micro_pan * USE_MICRO_SACC
    tilt_cmd = micro_tilt * USE_MICRO_SACC

    if i % RENDER_HZ // CONTROL_HZ == 0:
    # if False:
        pan_cmd  += ctrl_pan
        tilt_cmd += ctrl_tilt

    # if i == 200:
    #     data.ctrl[pan_act] += np.pi / 6

    data.ctrl[pan_act]  = np.clip(data.ctrl[pan_act] + pan_cmd,  np.deg2rad(-65), np.deg2rad(65))
    data.ctrl[tilt_act] = np.clip(data.ctrl[tilt_act] + tilt_cmd, np.deg2rad(-25), np.deg2rad(25))

    measured_positions.append(center_coords)

# ------------------- Save Outputs -------------------
os.makedirs("out", exist_ok=True)

# 1. RGB preview video (what the camera actually saw)
imageio.mimsave("out/rgb_preview.mp4", frames_preview, fps=RENDER_HZ)

print("Saved: out/rgb_preview.mp4, out/events_preview.mp4, "
      "out/accumulated_events.png, out/events.npy")

# -------------- Plot Controller Outputs --------------

plot_controller_response(
    measured_arr=np.array(measured_positions), 
    control_arr=np.array([c[:2] for c in control]), 
    target_tilt=center_coords[0],
    target_pan=center_coords[1],
    controller_gain=kp,
    out_path=OUT_PATH
)
