import time
import numpy as np
from loguru import logger

import matplotlib.pyplot as plt

from controller.nengo_controller import NengoController

_nengo_controller_obj = None

def run_controller(measured_coords, target_coords, k_pan, k_tilt):
    global _nengo_controller_obj
    if _nengo_controller_obj is None:
        ## Create a singleton object for the nengo controller.
        ## Parameters are the target (tilt, pan) position of the tracked
        ## signal and the gains for the pan and tilt axes.
        _nengo_controller_obj = NengoController(
                target_tilt=target_coords[0],
                target_pan=target_coords[1],
                k_pan=k_pan,
                k_tilt=k_tilt,
        )
    # return (target_coords - measured_coords)
    return _nengo_controller_obj(measured_coords)

def plot_controller_response(measured_arr, control_arr, target_pan, target_tilt, controller_gain, out_path, padding = 0.2):
    """
    measured_arr: (N, 2) array, columns = (tilt, pan) -- measured position of the tracked signal
    control_arr:  (N, 2) array, columns = (tilt, pan) -- controller output
    target_tilt, target_pan: scalars -- the fixed setpoints
    """

    error_pan  = measured_arr[:, 0] - target_pan
    error_tilt = measured_arr[:, 1] - target_tilt

    fig, axes = plt.subplots(5, 2, figsize=(12, 8), sharex=True)

    # --- Row 1: measured position vs target ---
    y_min = measured_arr.min()
    y_max = measured_arr.max()

    pad = padding * (y_max - y_min) or 1.0
    y_min, y_max = y_min - pad, y_max + pad

    axes[0, 0].plot(measured_arr[:, 0], label='measured_pan', color='tab:orange')
    axes[0, 0].axhline(target_pan, color='gray', linestyle='--', label='target')
    axes[0, 0].set_ylim(y_min, y_max)
    axes[0, 0].set_title('Pan axis')
    axes[0, 0].set_ylabel('position')
    # axes[0, 1].legend()

    axes[0, 1].plot(measured_arr[:, 1], label='measured_tilt')
    axes[0, 1].axhline(target_tilt, color='gray', linestyle='--', label='target')
    axes[0, 1].set_ylim(y_min, y_max)
    axes[0, 1].set_title('Tilt axis')
    # axes[0, 0].legend()


    # --- Row 2: error (measured - target) ---
    y_min = min(error_tilt.min(), error_pan.min())
    y_max = max(error_tilt.max(), error_pan.max())

    pad = padding * (y_max - y_min) or 1.0
    y_min, y_max = y_min - pad, y_max + pad

    axes[1, 0].plot(error_pan, color='tab:red')
    axes[1, 0].axhline(0, color='black', linewidth=0.8)
    axes[1, 0].set_ylim(y_min, y_max)
    axes[1, 0].set_ylabel('error')

    axes[1, 1].plot(error_tilt, color='tab:red')
    axes[1, 1].axhline(0, color='black', linewidth=0.8)
    axes[1, 1].set_ylim(y_min, y_max)

    # --- Row 3: controller output ---
    y_min = control_arr.min()
    y_max = control_arr.max()

    pad = padding * (y_max - y_min) or 1.0
    y_min, y_max = y_min - pad, y_max + pad

    axes[2, 0].plot(control_arr[:, 0], color='tab:green')
    axes[2, 0].axhline(0, color='black', linewidth=0.8)
    axes[2, 0].set_ylim(y_min, y_max)
    axes[2, 0].set_ylabel('controller output')
    axes[2, 0].set_xlabel('window index')

    axes[2, 1].plot(control_arr[:, 1], color='tab:green')
    axes[2, 1].axhline(0, color='black', linewidth=0.8)
    axes[2, 1].set_ylim(y_min, y_max)
    axes[2, 1].set_xlabel('window index')

    # --- Row 4: error * gain - control output ---
    data_b = error_pan * controller_gain - control_arr[:, 0]
    data_a = error_tilt * controller_gain - control_arr[:, 1]

    y_min = min(data_a.min(), data_b.min())
    y_max = max(data_a.max(), data_b.max())

    pad = padding * (y_max - y_min) or 1.0
    y_min, y_max = y_min - pad, y_max + pad

    axes[3, 0].plot(error_pan * controller_gain - control_arr[:, 0], color='tab:purple')
    axes[3, 0].set_xlabel('window index')
    axes[3, 0].set_ylim(y_min, y_max)
    axes[3, 0].set_ylabel('error * kp - control')

    axes[3, 1].plot(error_tilt * controller_gain - control_arr[:, 1], color='tab:purple')
    axes[3, 1].set_xlabel('window index')
    axes[3, 1].set_ylim(y_min, y_max)

    # --- Row 5: error - control output ---
    data_a = error_tilt - control_arr[:, 1]
    data_b = error_pan - control_arr[:, 0]

    y_min = min(data_a.min(), data_b.min())
    y_max = max(data_a.max(), data_b.max())

    pad = padding * (y_max - y_min) or 1.0
    y_min, y_max = y_min - pad, y_max + pad

    axes[4, 0].plot(error_pan - control_arr[:, 0], color='tab:purple')
    axes[4, 0].set_ylim(y_min, y_max)
    axes[4, 0].set_xlabel('window index')
    axes[4, 0].set_ylabel('error - control')

    axes[4, 1].plot(error_tilt - control_arr[:, 1], color='tab:purple')
    axes[4, 1].set_ylim(y_min, y_max)
    axes[4, 1].set_xlabel('window index')

    fig.suptitle('Controller response: position, error, output')
    fig.tight_layout()
    fig.savefig(f'{out_path}/controller_vis.png')
    plt.close(fig)
    logger.info(f"Controller response plot saved to {out_path}")
