# Event Camera on a Pan-Tilt — Minimum Viable Project

Simulates a 2-DOF pan-tilt rig in MuJoCo, moves it on a sinusoidal
trajectory in front of a textured wall with a few high-contrast objects,
renders RGB frames from the mounted camera, and converts those frames
into a synthetic event stream using a simple per-pixel log-intensity
threshold model (the same core idea ESIM/vid2e use, minus the adaptive
frame interpolation).

## Setup

```bash
pip install -r requirements.txt
# Linux headless rendering (no GPU required):
export MUJOCO_GL=osmesa
# If osmesa isn't available on your system:
#   sudo apt-get install libosmesa6 libosmesa6-dev
# On macOS/Windows or if you have a GPU, you can instead try:
#   export MUJOCO_GL=egl     (GPU, headless)
#   export MUJOCO_GL=glfw    (opens a window)
```

## Run

```bash
python3 simulate.py
```

Takes a few seconds. Outputs land in `out/`:

- `rgb_preview.mp4` — what the camera actually saw (ground truth video)
- `events_preview.mp4` — event stream rendered as a video (red = brightness
  increase, blue = brightness decrease), frame-aligned with the RGB preview
- `accumulated_events.png` — every event over the whole run collapsed into
  one image, the classic "event camera snapshot" visualization
- `events.npy` — the raw event stream as a numpy structured array with
  fields `(t, x, y, p)` — load with `np.load("out/events.npy")`

## Tuning knobs (top of `simulate.py`)

- `CONTRAST_THRESHOLD` — how much log-intensity change is needed to fire
  an event. Lower = more sensitive / noisier, higher = sparser / cleaner.
- `PAN_AMPLITUDE_DEG`, `TILT_AMPLITUDE_DEG`, `PAN_FREQ_HZ`, `TILT_FREQ_HZ` —
  the pan-tilt motion profile.
- `RENDER_HZ` — how often frames are rendered/compared. Higher gives more
  accurate event timing for fast motion, at the cost of runtime.

## Next steps

This event model is intentionally simplified (no frame interpolation,
no noise model, no refractory period) so the whole thing runs with just
`mujoco` + `numpy` + `imageio` — no ROS, no compiled C++ dependencies.

To get a more physically accurate event stream, swap the event-generation
block in `simulate.py` for **vid2e** (`uzh-rpg/rpg_vid2e`), which
implements the full ESIM contrast-threshold model with adaptive frame
upsampling — it takes the same input (a sequence of frames + timestamps)
that this script already produces.
