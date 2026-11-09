import math
from dataclasses import dataclass

@dataclass
class SimulatorConfig:
    xml_path: str = "pan_tilt.xml"
    out_path: str = "./out"
    width: int = 320
    height: int = 320
    sim_dt: float = 0.002           # physics timestep
    render_hz: int = 300            # frame render rate fed into the sensor
    control_hz: int = 30
    duration: float = 4.0           # seconds
    cam_name: str = "event_cam"
    sign_pan: int = 1               # positive pan direction is left
    sign_tilt: int = 1              # positive tilt direction is down
    use_micro_sacc: bool = False
    micro_freq_hz: float = 10.0
    micro_amp_deg: float = 0.5
    micro_phase: float = math.pi / 2
    mark_targets: bool = False
