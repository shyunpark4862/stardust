"""O1: sd.optimize.least_squares and approx_fprime against scipy 1.18.1 (docs/design/optimize.md §5):
time, the time outside the user's callbacks per iteration (the method's own cost), nfev and njev.

Problems:
  - Moré-Garbow-Hillstrom cases (tests/optimize/mgh.py): Rosenbrock (n = 2), Osborne 2 (n = 11,
    m = 65), Brown almost-linear (n = 40), from x0; the residuals are Python lists, the same
    code for both libraries (wrapped as sd.ndarray / numpy array);
  - O0's y_i - exp(x_i . theta) (x_i ~ N(0, 1/n), theta* = 0.3, start 0) at (m, n) = (10^3, 10),
    (10^5, 10), (10^4, 100), written with each library's arrays (sd for sd, numpy for scipy);
  - methods 'trf' and 'lm' with the analytic Jacobian and '2-point' differences;
  - approx_fprime of scipy's rosen at n = 10, 100, 1000: scipy's forward differences
    (epsilon = sqrt(u) for both) and sd's accurate 5-point differences (epsilon=None).

Time is the best of up to 5 runs (fewer when one run takes over 2 s), in ms. "own" is the time
outside fun and jac (summed with perf_counter around each call) divided by the iterations
(sd: n_iter; scipy: njev for trf, nfev for lm, the counts that are one per iteration). nfev is
each library's count: sd's counts the difference calls (O18), scipy's 'trf' and 'lm' with
'2-point' do not. Each thread count runs in its own process (VECLIB_MAXIMUM_THREADS for
Accelerate, sd.set_num_threads for sd).

Run: .venv/bin/python bench/optimize/o1/speed.py
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


def best(fn, reps=5, budget=2.0):
    out, v = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        v = fn()
        out.append(time.perf_counter() - t0)
        if sum(out) > budget:
            break
    return min(out) * 1e3, v


class Timed:
    """fun / jac wrapped with the time spent inside them (of the last run)."""

    def __init__(self, f):
        self.f, self.t = f, 0.0

    def __call__(self, x):
        t0 = time.perf_counter()
        v = self.f(x)
        self.t += time.perf_counter() - t0
        return v


def run_pair(lib, solve, F, J, iters):
    """(ms, own ms per iteration, result) of the fastest run."""
    best_ms, best_own, res = None, None, None
    total = 0.0
    for _ in range(5):
        tf, tj = Timed(F), Timed(J) if J is not None else None
        t0 = time.perf_counter()
        r = solve(tf, tj)
        dt = time.perf_counter() - t0
        total += dt
        inside = tf.t + (tj.t if tj is not None else 0.0)
        if best_ms is None or dt < best_ms / 1e3:
            best_ms = dt * 1e3
            best_own = (dt - inside) * 1e3 / max(iters(r), 1)
            res = r
        if total > 2.0:
            break
    return best_ms, best_own, res


def cases():
    import numpy as np
    from scipy import optimize as so

    import stardust as sd

    sys.path.insert(0, str(ROOT))
    from tests.optimize import mgh

    rows = []

    def both(group, size, F_sd, J_sd, F_np, J_np, x0, n):
        for meth in ("trf", "lm"):
            for jac in ("analytic", "2-point"):

                def s_sd(tf, tj, meth=meth, jac=jac):
                    return sd.optimize.least_squares(tf, x0, jac=tj if jac == "analytic" else "2-point", method=meth)

                def s_np(tf, tj, meth=meth, jac=jac):
                    return so.least_squares(
                        tf, np.asarray(x0, dtype=float), jac=tj if jac == "analytic" else "2-point", method=meth
                    )

                ms, own, r = run_pair("sd", s_sd, F_sd, J_sd if jac == "analytic" else None, lambda r: r.n_iter)
                sp_iters = (lambda r: r.njev) if meth == "trf" else (lambda r: r.njev if r.njev else r.nfev)
                ms2, own2, r2 = run_pair("scipy", s_np, F_np, J_np if jac == "analytic" else None, sp_iters)
                rows.append(
                    {
                        "group": group,
                        "size": size,
                        "method": f"{meth}/{jac}",
                        "sd_ms": ms,
                        "sd_own": own,
                        "sd_nfev": r.nfev,
                        "sd_njev": r.njev,
                        "sd_iter": r.n_iter,
                        "sd_status": r.status,
                        "sp_ms": ms2,
                        "sp_own": own2,
                        "sp_nfev": int(r2.nfev),
                        "sp_njev": None if r2.njev is None else int(r2.njev),
                        "sp_status": int(r2.status),
                        "cost_sd": r.cost,
                        "cost_sp": float(r2.cost),
                    }
                )

    for cid in ("mgh01-n2-m2-x1", "mgh19-n11-m65-x1", "mgh27-n40-m40-x1"):
        c = mgh.BY_ID[cid]
        both(
            "mgh",
            cid,
            lambda x, c=c: sd.asarray(c.fun(x.tolist())),
            lambda x, c=c: sd.asarray(c.jac(x.tolist())),
            lambda x, c=c: np.array(c.fun(x.tolist())),
            lambda x, c=c: np.array(c.jac(x.tolist())),
            c.x0(),
            c.n,
        )

    rng = np.random.default_rng(0)
    for m_, n in ((1000, 10), (100000, 10), (10000, 100)):
        X = rng.standard_normal((m_, n)) / np.sqrt(n)
        y = np.exp(X @ np.full(n, 0.3)) + 0.05 * rng.standard_normal(m_)
        Xs, ys = sd.asarray(X), sd.asarray(y)
        both(
            "exp",
            f"({m_}, {n})",
            lambda t, X=Xs, y=ys: y - sd.exp(sd.matmul(X, t)),
            lambda t, X=Xs: -sd.reshape(sd.exp(sd.matmul(X, t)), (X.shape[0], 1)) * X,
            lambda t, X=X, y=y: y - np.exp(X @ t),
            lambda t, X=X: -np.exp(X @ t)[:, None] * X,
            [0.0] * n,
            n,
        )

    for n in (10, 100, 1000):
        x = np.tile([-1.2, 1.0], n // 2)
        xs = sd.asarray(x)
        ms_sp, _ = best(lambda: so.approx_fprime(x, so.rosen))
        rs = lambda t: float(so.rosen(np.asarray(t)))  # noqa: E731
        ms_f, _ = best(lambda: sd.optimize.approx_fprime(xs, rs, epsilon=2.0**-26))
        ms_a, _ = best(lambda: sd.optimize.approx_fprime(xs, rs))
        calls = [0]

        def counted(t):
            calls[0] += 1
            return rs(t)

        sd.optimize.approx_fprime(xs, counted)
        rows.append(
            {
                "group": "approx_fprime",
                "size": f"n={n}",
                "method": "rosen",
                "sp_ms": ms_sp,
                "sd_fwd_ms": ms_f,
                "sd_ms": ms_a,
                "sd_calls": calls[0],
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
        "| group | size | method | sd ms (1/10 thr) | scipy ms (1/10 thr) | sd own/iter | scipy own/iter | "
        "sd nfev/njev | scipy nfev/njev | status sd/scipy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(results[1], results[10]):
        if a["group"] == "approx_fprime":
            continue
        lines.append(
            f"| {a['group']} | {a['size']} | {a['method']} | {a['sd_ms']:.3g} / {b['sd_ms']:.3g} | "
            f"{a['sp_ms']:.3g} / {b['sp_ms']:.3g} | {a['sd_own']:.3g} | {a['sp_own']:.3g} | "
            f"{a['sd_nfev']}/{a['sd_njev']} | {a['sp_nfev']}/{a['sp_njev']} | "
            f"{a['sd_status']}/{a['sp_status']} |"
        )
    lines += [
        "",
        "| approx_fprime | scipy ms | sd forward ms | sd accurate ms | sd accurate calls |",
        "|---|---:|---:|---:|---:|",
    ]
    for a in results[1]:
        if a["group"] == "approx_fprime":
            lines.append(
                f"| {a['size']} | {a['sp_ms']:.3g} | {a['sd_fwd_ms']:.3g} | {a['sd_ms']:.3g} | {a['sd_calls']} |"
            )
    text = "\n".join(lines) + "\n"
    (HERE / "speed.log").write_text(text)
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main()
