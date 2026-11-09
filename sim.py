import numpy as np
import cv2
import imageio
from typing import NamedTuple

import mujoco

from models.config import SimulatorConfig

class CameraPose(NamedTuple):
    position: np.ndarray
    orientation: np.ndarray

class PanTiltSim:
    def __init__(
        self,
        config: SimulatorConfig
    ):
        # Paths and timing
        self.xml_path       = config.xml_path
        self.out_path       = config.out_path
        self.sim_dt         = config.sim_dt
        self.render_hz      = config.render_hz
        self.control_hz     = config.control_hz
        self.duration       = config.duration
        self.current_step   = 0
        self.current_time   = 0

        # Image
        self.width  = config.width
        self.height = config.height
        self.max_x  = config.width
        self.max_y  = config.height  # can be overridden later to match the .npy resolution

        # MuJoCo
        self.model              = mujoco.MjModel.from_xml_path(config.xml_path)
        self.model.opt.timestep = config.sim_dt
        self.data               = mujoco.MjData(self.model)
        self.renderer           = mujoco.Renderer(self.model,
                                                  height=config.height,
                                                  width=config.width)
        self.render_dt  = 1 / config.render_hz
        self.cam_name   = config.cam_name
        self.cam_id     = self.model.camera(config.cam_name).id
        self.pan_act    = self.model.actuator("pan_act").id
        self.tilt_act   = self.model.actuator("tilt_act").id

        # Camera intrinsics, read from the model so they match the XML
        self.fovy = np.deg2rad(self.model.cam_fovy[self.cam_id])
        self.fovx = 2 * np.arctan(np.tan(self.fovy / 2) * config.width / config.height)
        self.f_px = (config.height / 2) / np.tan(self.fovy / 2) # focal length in pixels

        self.rad_per_px_tilt    = self.fovy / config.height
        self.rad_per_px_pan     = self.fovx / config.width

        # Controller
        self.sign_tilt  = config.sign_tilt
        self.sign_pan   = config.sign_pan

        # Microsaccades
        self.use_micro_sacc = config.use_micro_sacc
        self.micro_freq_hz  = config.micro_freq_hz
        self.micro_amp      = np.deg2rad(config.micro_amp_deg)
        self.micro_phase    = config.micro_phase

        # Derived step counts
        self.steps_per_render       = round(1 / (config.render_hz * config.sim_dt))
        self.renders_per_control    = config.render_hz // config.control_hz
        self.n_steps                = round(config.duration / config.sim_dt)

        # Targets
        self._num_of_targets    = 3
        self.mark_targets       = config.mark_targets
        self.target_positions   = [None, None, None]
        self.target_pos_px      = [None, None, None]

        # Output
        self.output_buffer = []

    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        self.current_time += self.sim_dt

    def read_target_pos(self, num_of_targets: int=3) -> list[np.ndarray]:
        return [self.data.geom(f'target{t + 1}').xpos.copy() for t in range(num_of_targets)]

    def get_camera_pose(self) -> CameraPose:
        cam_id = self.model.camera(self.cam_name).id
        cam_pos = self.data.cam_xpos[cam_id].copy()
        R = self.data.cam_xmat[cam_id].reshape(3, 3)

        return CameraPose(cam_pos, R)

    def world_to_pixel(self, target_pos: np.ndarray, cam_pose: CameraPose) -> tuple[float, float] | None:
        cam_pos, R = cam_pose
        x, y, z = R.T @ (target_pos - cam_pos) # target pos in camera coords

        if z >= 0:
            return None

        u = self.width / 2 + self.f_px * x / (-z)
        v = self.height / 2 - self.f_px * y / (-z)

        if not (0 <= u < self.width and 0 <= v < self.height):
            return None

        return float(u), float(v)

    def get_target_pos_px(self, target: int):
        if not (0 <= target <= self._num_of_targets - 1):
            print("Target index out of range.")
            return None

        return self.target_pos_px[target]

    def set_pan_cmd(self, cmd: float):
        self.data.ctrl[self.pan_act]  = np.clip(self.data.ctrl[self.pan_act] + cmd * self.sign_pan,  np.deg2rad(-65), np.deg2rad(65))

    def set_tilt_cmd(self, cmd: float):
        self.data.ctrl[self.tilt_act] = np.clip(self.data.ctrl[self.tilt_act] + cmd * self.sign_tilt, np.deg2rad(-25), np.deg2rad(25))

    def set_cmd(self, cmd: list[float]):
        self.set_pan_cmd(cmd[0])
        self.set_tilt_cmd(cmd[1])

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.cam_name)
        frame = self.renderer.render()

        self.target_positions = self.read_target_pos(3)
        cam_pose = self.get_camera_pose()
        self.target_pos_px = [self.world_to_pixel(target_pos=p, cam_pose=cam_pose) for p in self.target_positions]

        if self.mark_targets:
            for u, v in (p for p in self.target_pos_px if p is not None):
                cv2.circle(frame, (round(u), round(v)), radius=5, color=(0, 255, 255))

        self.output_buffer.append(frame)
        return frame

    def save_rgb(self, path: str | None = None):
        if path == None:
            path = self.out_path

        imageio.mimsave("out/rgb_preview.mp4", self.output_buffer, fps=self.render_hz)
