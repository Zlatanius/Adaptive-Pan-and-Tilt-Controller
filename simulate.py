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
from matplotlib.pyplot import flag
import numpy as np
import mujoco
import imageio
import cv2
import math
import torch
from tqdm import tqdm
from loguru import logger
from typing import NamedTuple

from controller.nengo_controller import NengoController
from controller.helper import plot_controller_response
from sim import PanTiltSim

# ------------------------------- Sim Setup ------------------------------
sim = PanTiltSim(
    xml_path="pan_tilt.xml",
    out_path="./out",
    width=320,
    height=320,
    sim_dt=0.002,
    render_hz=300,
    control_hz=30,
    duration=4.0,
    cam_name="event_cam",
    sign_pan=-1,
    sign_tilt=1,
    mark_targets=True
)



n_frames = int(sim.duration * sim.render_hz)
steps_per_frame = max(1, int(round(sim.render_dt / sim.sim_dt)))

# ------------------------------- Controller Setup -------------------------------
kp = 0.5
k_pan = np.array([kp, 0, 0])
k_tilt = np.array([kp, 0, 0])
center_coords = np.array([sim.width // 2, sim.height // 2])
pan_cmd = 0.0     # rad, integrated state
tilt_cmd = 0.0

controller = NengoController(target_tilt=center_coords[0], target_pan=center_coords[1], k_pan=k_pan, k_tilt=k_tilt)

# ------------------------------- Diagnostics Varibles -------------------------------
control = []
measured_positions = []

# ---------------------------------------------------------------- main loop
# Progress bar delayed by 1 it so that I can print debug in first it
for i in tqdm(range(n_frames), ncols=100, delay=1):

    # Step the simulation
    for _ in range(steps_per_frame):
        sim.step()

    # Render the scene
    sim.render()

    # -------------------- Control and pan and tilt drive --------------------
    target = sim.get_target_pos_px(0)

    if target == None:
        target = center_coords

    cmd = controller(target)[0:2]
    measured_positions.append(target)

    sim.set_cmd(cmd)
    control.append(cmd)


# ------------------- Debug -------------------
    # if i == 0:
    #     print(target_pos_px[0])

# ------------------- Save Outputs -------------------
os.makedirs("out", exist_ok=True)

# 1. RGB preview video (what the camera actually saw)
sim.save_rgb()

print("Saved: out/rgb_preview.mp4, out/events_preview.mp4, "
      "out/accumulated_events.png, out/events.npy")

# -------------- Plot Controller Outputs --------------

plot_controller_response(
    measured_arr=np.array(measured_positions), 
    control_arr=np.array([c[:2] for c in control]), 
    target_tilt=center_coords[0],
    target_pan=center_coords[1],
    controller_gain=kp,
    out_path="./out"
)
