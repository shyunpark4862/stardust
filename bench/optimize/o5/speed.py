"""O5: sd.optimize.root, lsq_linear and nnls against scipy 1.18.1 (docs/design/optimize.md §5).

root: Broyden's tridiagonal system f_i = (3 - 2 x_i) x_i - x_{i-1} - 2 x_{i+1} + 1 at n = 2, 10, 100 from
x0 = (-10, ..., -10), written with each library's arrays (sd for sd, numpy for scipy), 'hybr' and 'lm'
with the Jacobian and without (differences). Time, and the time outside fun and jac per iteration (the
method's own cost): an iteration is a step tried for 'hybr' (sd's n_iter: the steps are scipy's, O9) and
a Jacobian for 'lm' with the Jacobian (each library's njev); 'lm' without it shows the time only.

lsq_linear ('trf', 'bvls') and nnls: a random A (m, n) (standard normal, seed 9), b = A z + e, x >= 0, at
(m, n) = (100, 10), (1000, 100), (10^4, 100). Time, iterations, status, and whether x has scipy's bits.

Time is the best of up to 5 runs (fewer when one run takes over 2 s), in ms. Each thread count runs in its
own process (VECLIB_MAXIMUM_THREADS for Accelerate, sd.set_num_threads for sd).

Run: .venv/bin/python bench/optimize/o5/speed.py
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
        self.f, self.t, self.calls = f, 0.0, 0

    def __call__(self, *a):
        t0 = time.perf_counter()
        v = self.f(*a)
        self.t += time.perf_counter() - t0
        self.calls += 1
        return v


def best(solve, funcs=None):
    """(ms, ms outside the callbacks, result, the timed callbacks) of the fastest of up to 5 runs."""
    out, total = None, 0.0
    for _ in range(5):
        timed = {k: Timed(v) for k, v in (funcs or {}).items()}
        t0 = time.perf_counter()
        r = solve(timed)
        dt = time.perf_counter() - t0
        total += dt
        own = dt - sum(t.t for t in timed.values())
        if out is None or dt * 1e3 < out[0]:
            out = (dt * 1e3, own * 1e3, r, timed)
        if total > 2.0:
            break
    return out


def sd_system():
    import stardust as sd

    def f(x):
        out = (3.0 - 2.0 * x) * x + 1.0
        out[1:] = out[1:] - x[:-1]
        out[:-1] = out[:-1] - 2.0 * x[1:]
        return out

    def j(x):
        n = x.shape[0]
        J = sd.zeros((n, n))
        flat = sd.reshape(J, (-1,))
        flat[:: n + 1] = 3.0 - 4.0 * x
        flat[n :: n + 1] = -1.0
        flat[1 :: n + 1] = -2.0
        return J

    return f, j


def np_system():
    import numpy as np

    def f(x):
        out = (3.0 - 2.0 * x) * x + 1.0
        out[1:] -= x[:-1]
        out[:-1] -= 2.0 * x[1:]
        return out

    def j(x):
        n = x.shape[0]
        J = np.zeros((n, n))
        flat = J.reshape(-1)
        flat[:: n + 1] = 3.0 - 4.0 * x
        flat[n :: n + 1] = -1.0
        flat[1 :: n + 1] = -2.0
        return J

    return f, j


def root_rows():
    import numpy as np
    from scipy import optimize as so

    import stardust as sd

    fs, js = sd_system()
    fn, jn = np_system()
    rows = []
    for n in (2, 10, 100):
        x0 = [-10.0] * n
        for method in ("hybr", "lm"):
            for jac in (True, False):

                def s_sd(t, method=method, jac=jac, x0=x0):
                    return sd.optimize.root(t["f"], x0, method=method, jac=t["j"] if jac else None)

                def s_sp(t, method=method, jac=jac, x0=x0):
                    return so.root(t["f"], np.array(x0), method=method, jac=t["j"] if jac else None)

                ms, own, r, _ = best(s_sd, {"f": fs, "j": js})
                ms2, own2, r2, t2 = best(s_sp, {"f": fn, "j": jn})
                if method == "hybr":
                    it, it2 = r.n_iter, r.n_iter  # the same steps
                elif jac:
                    it, it2 = r.njev, int(r2.njev)
                else:
                    it = it2 = None
                rows.append(
                    {
                        "method": method,
                        "jac": jac,
                        "n": n,
                        "sd_ms": ms,
                        "sp_ms": ms2,
                        "sd_own": own / it if it else None,
                        "sp_own": own2 / it2 if it2 else None,
                        "iter": it,
                        "sd_nfev": r.nfev,
                        "sp_calls": t2["f"].calls,
                        "sd_status": r.status,
                        "sp_status": int(r2.status),
                        "same_x": r.x.tolist() == r2.x.tolist(),
                    }
                )
    return rows


def linear_rows():
    import numpy as np
    from scipy import optimize as so

    import stardust as sd

    rows = []
    rng = np.random.default_rng(9)
    for m, n in ((100, 10), (1000, 100), (10000, 100)):
        A = rng.standard_normal((m, n))
        b = A @ rng.standard_normal(n) + rng.standard_normal(m)
        Asd, bsd = sd.asarray(A), sd.asarray(b)
        bounds = (np.zeros(n), np.full(n, np.inf))
        for what in ("trf", "bvls", "nnls"):
            if what == "nnls":
                ms, _, r, _ = best(lambda t: sd.optimize.nnls(Asd, bsd))
                ms2, _, r2, _ = best(lambda t: so.nnls(A, b))
                x2, it, it2, st, st2 = r2[0], r.n_iter, None, r.status, None
            else:
                ms, _, r, _ = best(lambda t, w=what: sd.optimize.lsq_linear(Asd, bsd, bounds=(0.0, np.inf), method=w))
                ms2, _, r2, _ = best(lambda t, w=what: so.lsq_linear(A, b, bounds=bounds, method=w))
                x2, it, it2, st, st2 = r2.x, r.n_iter, int(r2.nit), r.status, int(r2.status)
            rows.append(
                {
                    "what": what,
                    "m": m,
                    "n": n,
                    "sd_ms": ms,
                    "sp_ms": ms2,
                    "iter": it,
                    "sp_iter": it2,
                    "status": st,
                    "sp_status": st2,
                    "same_x": r.x.tolist() == x2.tolist(),
                }
            )
    return rows


def worker(threads):
    warnings.simplefilter("ignore")
    import numpy as np

    np.seterr(all="ignore")
    import stardust as sd

    sd.set_num_threads(threads)
    print(json.dumps({"root": root_rows(), "linear": linear_rows()}))


def g(v):
    return "" if v is None else f"{v:.3g}"


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
        "| method | jac | n | sd ms (1/10 thr) | scipy ms (1/10 thr) | sd own/iter ms (1/10) | scipy own/iter ms (1/10) "
        "| iterations | nfev sd / scipy's calls | status sd/scipy | same x |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for a, b in zip(results[1]["root"], results[10]["root"]):
        lines.append(
            f"| {a['method']} | {'jac' if a['jac'] else 'differences'} | {a['n']} | {g(a['sd_ms'])} / {g(b['sd_ms'])} | "
            f"{g(a['sp_ms'])} / {g(b['sp_ms'])} | {g(a['sd_own'])} / {g(b['sd_own'])} | "
            f"{g(a['sp_own'])} / {g(b['sp_own'])} | {a['iter'] or ''} | {a['sd_nfev']} / {a['sp_calls']} | "
            f"{a['sd_status']}/{a['sp_status']} | {a['same_x']} |"
        )
    lines += [
        "",
        "| function | m | n | sd ms (1/10 thr) | scipy ms (1/10 thr) | iterations sd/scipy | status sd/scipy | same x |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for a, b in zip(results[1]["linear"], results[10]["linear"]):
        lines.append(
            f"| {a['what']} | {a['m']} | {a['n']} | {g(a['sd_ms'])} / {g(b['sd_ms'])} | {g(a['sp_ms'])} / {g(b['sp_ms'])} "
            f"| {a['iter']}/{a['sp_iter'] if a['sp_iter'] is not None else '-'} | "
            f"{a['status']}/{a['sp_status'] if a['sp_status'] is not None else '-'} | {a['same_x']} |"
        )
    text = "\n".join(lines) + "\n"
    (HERE / "speed.log").write_text(text)
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main()
