"""O3: sd.optimize.minimize (Nelder-Mead, BFGS, L-BFGS-B) against scipy 1.18.1 (docs/design/optimize.md
§5): time, the time outside the user's callbacks per iteration (the method's own cost), nfev, njev
and the iterations.

Problem: scipy's generalized Rosenbrock function (rosen, rosen_der) at n = 2, 10, 100, 1000 (L-BFGS-B
also 10^4) from x0 = (-1.2, 1, -1.2, 1, ...), written with each library's arrays (sd for sd, numpy
for scipy); BFGS and L-BFGS-B with the analytic gradient and with differences (jac=None: scipy's
'2-point' with the absolute step eps). Large n runs are capped (max_iter, the same for both); the
steps are the same in both libraries (O9), so the iterations and counts are too.

Time is the best of up to 5 runs (fewer when one run takes over 2 s), in ms. "own" is the time
outside fun and jac (summed with perf_counter around each call) divided by the iterations. Each
thread count runs in its own process (VECLIB_MAXIMUM_THREADS for Accelerate, sd.set_num_threads
for sd).

Run: .venv/bin/python bench/optimize/o3/speed.py
"""

import json
import os
import pathlib
import subprocess
import sys
import time
import warnings

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[2]


class Timed:
    """fun / jac wrapped with the time spent inside them (of the last run)."""

    def __init__(self, f):
        self.f, self.t = f, 0.0

    def __call__(self, x):
        t0 = time.perf_counter()
        v = self.f(x)
        self.t += time.perf_counter() - t0
        return v


def run(solve, F, G):
    """(ms, own ms per iteration, result) of the fastest run."""
    best_ms, best_own, res, total = None, None, None, 0.0
    for _ in range(5):
        tf, tg = Timed(F), Timed(G) if G is not None else None
        t0 = time.perf_counter()
        r = solve(tf, tg)
        dt = time.perf_counter() - t0
        total += dt
        inside = tf.t + (tg.t if tg is not None else 0.0)
        iters = r.n_iter if hasattr(r, "n_iter") else r.nit
        if best_ms is None or dt * 1e3 < best_ms:
            best_ms, best_own, res = dt * 1e3, (dt - inside) * 1e3 / max(iters, 1), r
        if total > 2.0:
            break
    return best_ms, best_own, res


# (method, gradient, n, max_iter or None)
PLAN = [
    ("nelder-mead", None, 2, None),
    ("nelder-mead", None, 10, None),
    ("nelder-mead", None, 100, 2000),
    ("nelder-mead", None, 1000, 300),
    ("bfgs", "analytic", 2, None),
    ("bfgs", "analytic", 10, None),
    ("bfgs", "analytic", 100, None),
    ("bfgs", "analytic", 1000, 40),
    ("bfgs", "diff", 2, None),
    ("bfgs", "diff", 10, None),
    ("bfgs", "diff", 100, 100),
    ("bfgs", "diff", 1000, 10),
    ("l-bfgs-b", "analytic", 2, None),
    ("l-bfgs-b", "analytic", 10, None),
    ("l-bfgs-b", "analytic", 100, None),
    ("l-bfgs-b", "analytic", 1000, 300),
    ("l-bfgs-b", "analytic", 10000, 300),
    ("l-bfgs-b", "diff", 2, None),
    ("l-bfgs-b", "diff", 10, None),
    ("l-bfgs-b", "diff", 100, 100),
    ("l-bfgs-b", "diff", 1000, 10),
]
SCIPY = {"nelder-mead": "Nelder-Mead", "bfgs": "BFGS", "l-bfgs-b": "L-BFGS-B"}


def cases():
    import numpy as np
    from scipy import optimize as so

    import stardust as sd

    def f_sd(x):
        a = x[1:] - x[:-1] * x[:-1]
        b = 1.0 - x[:-1]
        return float(sd.sum(100.0 * (a * a) + b * b))

    def g_sd(x):
        xm, xm_m1, xm_p1 = x[1:-1], x[:-2], x[2:]
        g = sd.zeros(x.shape[0])
        g[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
        g[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        g[-1] = 200 * (x[-1] - x[-2] ** 2)
        return g

    rows = []
    for method, grad, n, cap in PLAN:
        x0 = [-1.2 if i % 2 == 0 else 1.0 for i in range(n)]
        opts_sd = {} if cap is None else {"max_iter": cap}
        opts_sp = {} if cap is None else {"maxiter": cap}
        analytic = grad == "analytic"

        def s_sd(tf, tg, method=method, analytic=analytic, x0=x0, opts=opts_sd):
            kw = {"jac": tg} if analytic else {}
            return sd.optimize.minimize(tf, x0, method=method, options=opts, **kw)

        def s_sp(tf, tg, method=method, analytic=analytic, x0=x0, opts=opts_sp):
            kw = {"jac": tg} if analytic else {}
            return so.minimize(tf, np.array(x0), method=SCIPY[method], options=opts, **kw)

        ms, own, r = run(s_sd, f_sd, g_sd if analytic else None)
        ms2, own2, r2 = run(s_sp, so.rosen, so.rosen_der if analytic else None)
        rows.append(
            {
                "method": method,
                "grad": grad or "-",
                "n": n,
                "cap": cap,
                "sd_ms": ms,
                "sd_own": own,
                "sd_nfev": r.nfev,
                "sd_njev": getattr(r, "njev", None) if hasattr(r, "njev") else None,
                "sd_iter": r.n_iter,
                "sd_status": r.status,
                "sp_ms": ms2,
                "sp_own": own2,
                "sp_nfev": int(r2.nfev),
                "sp_njev": int(r2.njev) if "njev" in r2 else None,
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
        "| method | gradient | n | cap | sd ms (1/10 thr) | scipy ms (1/10 thr) | sd own/iter ms (1/10) | "
        "scipy own/iter ms (1/10) | iterations sd/scipy | nfev sd/scipy | njev sd/scipy | status sd/scipy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(results[1], results[10]):
        lines.append(
            f"| {a['method']} | {a['grad']} | {a['n']} | {a['cap'] or ''} | {a['sd_ms']:.3g} / {b['sd_ms']:.3g} | "
            f"{a['sp_ms']:.3g} / {b['sp_ms']:.3g} | {a['sd_own']:.3g} / {b['sd_own']:.3g} | "
            f"{a['sp_own']:.3g} / {b['sp_own']:.3g} | {a['sd_iter']}/{a['sp_iter']} | {a['sd_nfev']}/{a['sp_nfev']} | "
            f"{a['sd_njev']}/{a['sp_njev']} | {a['sd_status']}/{a['sp_status']} |"
        )
    text = "\n".join(lines) + "\n"
    (HERE / "speed.log").write_text(text)
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main()
