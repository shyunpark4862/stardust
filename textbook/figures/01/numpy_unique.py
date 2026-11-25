"""Chapter 1: numpy's unique keeps -0.0 as the representative of {+0.0, -0.0} even when +0.0
comes first (checked with numpy 2.5 on this machine)."""
import numpy as np
for a in ([0.0, -0.0], [-0.0, 0.0], [0.0] * 5 + [-0.0] * 5):
    print(a[:4], "->", np.unique(np.array(a)), "signbit", np.signbit(np.unique(np.array(a))))
print("numpy", np.__version__)
