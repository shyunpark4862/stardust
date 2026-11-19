"""O4: sd.optimize.minimize (Newton-CG, trust-ncg, trust-exact) against scipy 1.18.1 (docs/design/optimize.md
§5): time, the time outside the user's callbacks per iteration (the method's own cost), nfev, njev, nhev
and the iterations.

Problem: scipy's generalized Rosenbrock function (rosen, rosen_der, rosen_hess, rosen_hess_prod) at
n = 2, 10, 100, 1000 from x0 = (-1.2, 1, -1.2, 1, ...), written with each library's arrays (sd for sd,
numpy for scipy). Newton-CG with hess, hessp, hess='2-point' and neither (scipy's differences of the
gradient); trust-ncg with hess and hessp; trust-exact with hess. Large n runs are capped (max_iter, the
same for both); the steps are the same in both libraries (O9), so the iterations and counts are too.

Time is the best of up to 5 runs (fewer when one run takes over 2 s), in ms. "own" is the time outside
fun, jac, hess and hessp (summed with perf_counter around each call) divided by the iterations. Each
thread count runs in its own process (VECLIB_MAXIMUM_THREADS for Accelerate, sd.set_num_threads for sd).

Run: .venv/bin/python bench/optimize/o4/speed.py
"""

import json
import os
import pathlib
import subprocess
import sys
import time
import warnings

HERE = pathlib.Path(__file__).parent


class Timed:
    """A user function wrapped with the time spent inside it (of the last run)."""

    def __init__(self, f):
        self.f, self.t = f, 0.0

    def __call__(self, *a):
        t0 = time.perf_counter()
        v = self.f(*a)
        self.t += time.perf_counter() - t0
        return v


def run(solve, funcs):
    """(ms, own ms per iteration, result) of the fastest run; funcs: name -> function or None."""
    best_ms, best_own, res, total = None, None, None, 0.0
    for _ in range(5):
        timed = {k: (Timed(v) if v is not None else None) for k, v in funcs.items()}
        t0 = time.perf_counter()
        r = solve(timed)
        dt = time.perf_counter() - t0
        total += dt
        inside = sum(t.t for t in timed.values() if t is not None)
        iters = r.n_iter if hasattr(r, "n_iter") else r.nit
        if best_ms is None or dt * 1e3 < best_ms:
            best_ms, best_own, res = dt * 1e3, (dt - inside) * 1e3 / max(iters, 1), r
        if total > 2.0:
            break
    return best_ms, best_own, res


# (method, Hessian, n, max_iter or None)
PLAN = []
for _n in (2, 10, 100, 1000):
    PLAN += [
        ("newton-cg", "hess", _n, 200 if _n == 1000 else None),
        ("newton-cg", "hessp", _n, 200 if _n == 1000 else None),
        ("newton-cg", "2-point", _n, 200 if _n == 1000 else None),
        ("newton-cg", "none", _n, 200 if _n == 1000 else None),
        ("trust-ncg", "hess", _n, 300 if _n == 1000 else None),
        ("trust-ncg", "hessp", _n, 300 if _n == 1000 else None),
        ("trust-exact", "hess", _n, 100 if _n == 1000 else None),
    ]
SCIPY = {"newton-cg": "Newton-CG", "trust-ncg": "trust-ncg", "trust-exact": "trust-exact"}


def sd_rosen():
    import stardust as sd

    def f(x):
        a = x[1:] - x[:-1] * x[:-1]
        b = 1.0 - x[:-1]
        return float(sd.sum(100.0 * (a * a) + b * b))

    def g(x):
        xm, xm_m1, xm_p1 = x[1:-1], x[:-2], x[2:]
        out = sd.zeros(x.shape[0])
        out[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
        out[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        out[-1] = 200 * (x[-1] - x[-2] ** 2)
        return out

    def h(x):
        n = x.shape[0]
        H = sd.zeros((n, n))
        flat = sd.reshape(H, (-1,))
        off = -400 * x[:-1]
        flat[1 :: n + 1] = off
        flat[n :: n + 1] = off
        d = sd.zeros(n)
        d[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
        d[-1] = 200
        d[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
        flat[:: n + 1] = d
        return H

    def hp(x, p):
        out = sd.zeros(p.shape[0])
        out[0] = (1200 * x[0] ** 2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
        out[1:-1] = -400 * x[:-2] * p[:-2] + (202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]) * p[1:-1] - 400 * x[1:-1] * p[2:]
        out[-1] = -400 * x[-2] * p[-2] + 200 * p[-1]
        return out

    return f, g, h, hp


def cases():
    import numpy as np
    from scipy import optimize as so

    import stardust as sd

    fs, gs, hs, hps = sd_rosen()
    rows = []
    for method, hk, n, cap in PLAN:
        x0 = [-1.2 if i % 2 == 0 else 1.0 for i in range(n)]
        opts_sd = {} if cap is None else {"max_iter": cap}
        opts_sp = {} if cap is None else {"maxiter": cap}

        def kw(t, hk=hk):
            k = {"jac": t["g"]}
            if hk == "hess":
                k["hess"] = t["h"]
            elif hk == "hessp":
                k["hessp"] = t["hp"]
            elif hk == "2-point":
                k["hess"] = "2-point"
            return k

        def s_sd(t, method=method, x0=x0, opts=opts_sd, kw=kw):
            return sd.optimize.minimize(t["f"], x0, method=method, options=opts, **kw(t))

        def s_sp(t, method=method, x0=x0, opts=opts_sp, kw=kw):
            return so.minimize(t["f"], np.array(x0), method=SCIPY[method], options=opts, **kw(t))

        ms, own, r = run(s_sd, {"f": fs, "g": gs, "h": hs, "hp": hps})
        ms2, own2, r2 = run(s_sp, {"f": so.rosen, "g": so.rosen_der, "h": so.rosen_hess, "hp": so.rosen_hess_prod})
        rows.append(
            {
                "method": method,
                "hess": hk,
                "n": n,
                "cap": cap,
                "sd_ms": ms,
                "sd_own": own,
                "sd_nfev": r.nfev,
                "sd_njev": r.njev,
                "sd_nhev": r.nhev,
                "sd_iter": r.n_iter,
                "sd_status": r.status,
                "sp_ms": ms2,
                "sp_own": own2,
                "sp_nfev": int(r2.nfev),
                "sp_njev": int(r2.njev),
                "sp_nhev": int(r2.nhev),
                "sp_iter": int(r2.nit),
                "sp_status": int(r2.status),
                "fun_sd": r.fun,
                "fun_sp": float(r2.fun),
            }
        )
    return rows


def worker(threads):
    warnings.simplefilter("ignore")
    import numpy as np

    np.seterr(all="ignore")
    import stardust as sd

    sd.set_num_threads(threads)
    print(json.dumps(cases()))


def main():
    import numpy
    import scipy

    results = {}
    for t in (1, 10):
        env = dict(os.environ, VECLIB_MAXIMUM_THREADS=str(t), OMP_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t))
        out = subprocess.run(
            [sys.executable, __file__, "--worker", str(t)], env=env, capture_output=True, text=True, check=True
        )
        results[t] = json.loads(out.stdout.strip().splitlines()[-1])
    (HERE / "speed.json").write_text(
        json.dumps({"scipy": scipy.__version__, "numpy": numpy.__version__, "rows": results}, indent=1) + "\n"
    )
    lines = [
        f"scipy {scipy.__version__}, numpy {numpy.__version__}",
        "",
        "| method | Hessian | n | cap | sd ms (1/10 thr) | scipy ms (1/10 thr) | sd own/iter ms (1/10) | "
        "scipy own/iter ms (1/10) | iterations sd/scipy | nfev sd/scipy | njev sd/scipy | nhev sd/scipy | "
        "status sd/scipy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(results[1], results[10]):
        lines.append(
            f"| {a['method']} | {a['hess']} | {a['n']} | {a['cap'] or ''} | {a['sd_ms']:.3g} / {b['sd_ms']:.3g} | "
            f"{a['sp_ms']:.3g} / {b['sp_ms']:.3g} | {a['sd_own']:.3g} / {b['sd_own']:.3g} | "
            f"{a['sp_own']:.3g} / {b['sp_own']:.3g} | {a['sd_iter']}/{a['sp_iter']} | {a['sd_nfev']}/{a['sp_nfev']} | "
            f"{a['sd_njev']}/{a['sp_njev']} | {a['sd_nhev']}/{a['sp_nhev']} | {a['sd_status']}/{a['sp_status']} |"
        )
    text = "\n".join(lines) + "\n"
    (HERE / "speed.log").write_text(text)
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main()
