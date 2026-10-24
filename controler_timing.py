import time
import numpy as np
from external.SpeckEgoMotion.controller.nengo_controller import NengoController
c = NengoController(target_tilt=60, target_pan=60,
                    k_pan=np.array([3.1, 0, 0]), k_tilt=np.array([3.1, 0, 0]))

for n in (1, 10, 100):
    t0 = time.perf_counter()
    c.sim.run_steps(n, progress_bar=False)
    print(f"run_steps({n:3d}): {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
_ = c.sim.data[c.p_tilt]
print(f"reading probe data: {time.perf_counter() - t0:.4f}s")

t0 = time.perf_counter()
c(np.array([25, 35]))
print(f"full __call__:      {time.perf_counter() - t0:.4f}s")
