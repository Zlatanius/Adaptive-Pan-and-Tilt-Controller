import os
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass

from controller.nengo_controller import NengoController
from controller.helper import plot_controller_response
from sim import PanTiltSim
from models.config import SimulatorConfig

# ------------------------------- Sim Setup ------------------------------
sim_config = SimulatorConfig(
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

sim = PanTiltSim(sim_config)


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

# -------------------------------- Main Loop --------------------------------
# Progress bar delayed by 1 it so that I can print debug in first it
for i in tqdm(range(n_frames), ncols=100, delay=1):

    # Step the simulation
    for _ in range(steps_per_frame):
        sim.step()

    # Render the scene
    sim.render()

    # -------------------- Control and Pan and Tilt Drive --------------------
    target = sim.get_target_pos_px(0)

    if target == None:
        target = center_coords

    cmd = controller(target)[0:2]
    measured_positions.append(target)

    sim.set_cmd(cmd)
    control.append(cmd)


# ------------------- Debug -------------------
    # if i == 0:
    #     print("Target: ", target)
    #     print("Target from sim: ", sim.get_target_pos_px(0))

# ------------------- Save Outputs -------------------
os.makedirs("out", exist_ok=True)

# 1. RGB preview video (what the camera actually saw)
sim.save_rgb()

print("Saved: out/rgb_preview.mp4")

# -------------- Plot Controller Outputs --------------

plot_controller_response(
    measured_arr=np.array(measured_positions), 
    control_arr=np.array([c[:2] for c in control]), 
    target_pan=center_coords[0],
    target_tilt=center_coords[1],
    controller_gain=kp,
    out_path="./out"
)
