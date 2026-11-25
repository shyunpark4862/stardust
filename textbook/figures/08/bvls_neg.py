import stardust as sd, numpy as np
A = np.eye(3); b = np.array([-1.0, -2.0, 5.0])
for meth in ("bvls", "trf"):
    r = sd.optimize.lsq_linear(sd.asarray(A), sd.asarray(b), bounds=(0.0, 1.0), method=meth)
    print(meth, "x", r.x.tolist(), "optimality", r.optimality, "active", r.active_mask.tolist(), "status", r.status, "n_iter", r.n_iter)
