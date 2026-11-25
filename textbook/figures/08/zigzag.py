"""경사 하강의 지그재그: f = (x^2 + 10 y^2)/2, (10, 1)에서 정확한 선 탐색과 걸음 1/L."""
import numpy as np, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
H = np.diag([1.0, 10.0])
for name, exact in (("gd_exact", True), ("gd_fixed", False)):
    x = np.array([10.0, 1.0]); pts = [x.copy()]
    for k in range(12):
        g = H @ x
        a = (g @ g) / (g @ H @ g) if exact else 1.0 / 10.0
        x = x - a * g; pts.append(x.copy())
    with open(f"{OUT}/{name}.dat", "w") as o:
        o.write("x y f\n")
        for p in pts: o.write(f"{p[0]:.6f} {p[1]:.6f} {0.5*p@H@p:.6e}\n")
    print(name, [round(0.5 * p @ H @ p, 4) for p in pts[:6]])
