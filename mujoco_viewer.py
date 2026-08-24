import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path("./pan_tilt.xml")
d = mujoco.MjData(m)
mujoco.viewer.launch(m, d)
