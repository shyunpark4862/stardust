"""NNLS 반복마다의 |P|와 비용(따로 빌드한 stardust; max_iter = k로 멈춘 반복점). nnls_kkt.py의 (200, 80, 1e8) 문제."""
import numpy as np, stardust as sd, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(4)
for (m, n, cond) in ((30, 10, 1e1), (60, 30, 1e3), (100, 50, 1e6), (200, 80, 1e8)):
    U, _ = np.linalg.qr(rng.normal(size=(m, n))); V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.logspace(0, -np.log10(cond), n)
    A = (U * s) @ V.T
    b = rng.normal(size=m)
    if m != 200: continue
    full = sd.optimize.nnls(sd.asarray(A), sd.asarray(b)); K = int(full.n_iter); cmin = float(full.cost)
    with open(f"{OUT}/nnls_traj.dat", "w") as o:
        o.write("k P cost\n")
        for k in range(1, K + 1):
            r = sd.optimize.nnls(sd.asarray(A), sd.asarray(b), max_iter=k)
            x = np.array(r.x.tolist())
            o.write(f"{k} {int((x > 0).sum())} {float(r.cost) - cmin:.6e}\n")
    print("done", K, cmin)
