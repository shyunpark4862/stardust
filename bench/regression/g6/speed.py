"""G6: the speed of nls against scipy's curve_fit (method 'lm', MINPACK lmdif) and R's nls
(Gauss-Newton, forward differences) (docs/design/regression.md §3.7). Results:
bench/regression/g6/speed.md (the raw output in speed.log).

Models (x uniform on [0, 5]; Gaussian noise of 2% of the signal's range; p0 the true values
times 1.1, plus 0.05 for the Gaussians' centres):
  - p = 2: a exp(-b x)
  - p = 4: d + (a - d) / (1 + (x / c)^b)  (four-parameter logistic)
  - p = 10: sum over k of a_k exp(-((x - c_k) / 0.4)^2), five Gaussians
n = 100, 10^4, 10^6. sd: nls(f, x, y, p0=...) with 5-point differences and with the analytic
jac, 1 and 10 threads (the threads only enter the model's sd operations and the QR). scipy:
curve_fit(f, x, y, p0) with numpy (2-point differences; with jac=). R: nls(y ~ ..., start)
in one session, system.time (100 calls at n = 100), best of several. nfev: calls of f (sd
and scipy's lmdif count the differences' calls; R's numericDeriv makes p calls per Jacobian,
which nls does not count, so R's column gives its iterations).

Run: uv run --no-sync python bench/regression/g6/speed.py
"""

import pathlib
import subprocess
import tempfile
import time
import warnings

import numpy as np

import stardust as sd

NS = (100, 10_000, 1_000_000)
CENTRES = (0.7, 1.6, 2.5, 3.4, 4.3)


def best(f, reps):
    out, v = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        v = f()
        out.append(time.perf_counter() - t0)
    return min(out), v


def ms(t):
    return f"{t * 1e3:,.3g}" if t < 10 else f"{t * 1e3:,.0f}"


def models(xp):
    exp = xp.exp
    m = {}
    m[2] = (
        lambda x, a, b: a * exp(-b * x),
        lambda x, a, b: xp.stack([exp(-b * x), -a * x * exp(-b * x)], axis=1),
        [2.0, 0.8],
    )

    def f4(x, a, b, c, d):
        return d + (a - d) / (1 + (x / c) ** b)

    def j4(x, a, b, c, d):
        u = (x / c) ** b
        den = 1 + u
        lg = xp.log(x / c)
        return xp.stack([1 / den, -(a - d) * u * lg / den**2, (a - d) * u * b / c / den**2, 1 - 1 / den], axis=1)

    m[4] = (f4, j4, [0.2, 2.5, 1.8, 3.0])

    def f10(x, *t):
        return sum(t[2 * k] * exp(-(((x - t[2 * k + 1]) / 0.4) ** 2)) for k in range(5))

    def j10(x, *t):
        cols = []
        for k in range(5):
            a, c = t[2 * k], t[2 * k + 1]
            g = exp(-(((x - c) / 0.4) ** 2))
            cols += [g, a * g * 2 * (x - c) / 0.16]
        return xp.stack(cols, axis=1)

    m[10] = (f10, j10, [1.0, CENTRES[0], 1.5, CENTRES[1], 0.8, CENTRES[2], 1.2, CENTRES[3], 0.6, CENTRES[4]])
    return m


R_FORMULA = {
    2: ("y ~ a*exp(-b*x)", ["a", "b"]),
    4: ("y ~ d + (a - d)/(1 + (x/c)^b)", ["a", "b", "c", "d"]),
    10: (
        "y ~ " + " + ".join(f"a{k}*exp(-((x - c{k})/0.4)^2)" for k in range(5)),
        [v for k in range(5) for v in (f"a{k}", f"c{k}")],
    ),
}


def p0_of(p, truth):
    out = [v * 1.1 for v in truth]
    if p == 10:
        out = [v * 1.1 if j % 2 == 0 else v + 0.05 for j, v in enumerate(truth)]
    return out


def data(n, p, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.01, 5.0, n)
    f, _, truth = models(np)[p]
    s = f(x, *truth)
    y = s + 0.02 * (s.max() - s.min()) * rng.standard_normal(n)
    return x, y, truth


def run_r(cases):
    lines = ["options(digits = 17)"]
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        for (n, p), (x, y, truth) in cases.items():
            path = d / f"d{n}_{p}.csv"
            np.savetxt(path, np.column_stack([y, x]), delimiter=",", fmt="%.17g")
            form, names = R_FORMULA[p]
            st = ", ".join(f"{a} = {v:.17g}" for a, v in zip(names, p0_of(p, truth)))
            reps = 100 if n <= 100 else (10 if n <= 10_000 else 1)
            lines.append(f"A <- read.csv('{path}', header = FALSE); dat <- data.frame(y = A[, 1], x = A[, 2])")
            lines.append(f"f <- tryCatch(nls({form}, data = dat, start = list({st})), error = function(e) NULL)")
            lines.append(
                f"if (!is.null(f)) {{ tt <- min(sapply(1:3, function(i) system.time(for (j in 1:{reps}) "
                f"nls({form}, data = dat, start = list({st})))[['elapsed']])) / {reps}; "
                f"cat('RES', '{n}', '{p}', tt, f$convInfo$finIter, sprintf('%.17g', coef(f)), '\\n') }}"
            )
        script = d / "run.R"
        script.write_text("\n".join(lines) + "\n")
        out = subprocess.run(
            ["/opt/homebrew/bin/Rscript", str(script)], capture_output=True, text=True, check=True
        ).stdout
    res = {}
    for ln in out.splitlines():
        if ln.startswith("RES"):
            parts = ln.split()
            res[(int(parts[1]), int(parts[2]))] = (float(parts[3]), int(parts[4]), [float(v) for v in parts[5:]])
    return res


def main():
    from scipy.optimize import curve_fit

    cases = {(n, p): data(n, p) for n in NS for p in (2, 4, 10)}
    r_res = run_r(cases)
    msd = models(sd)
    mnp = models(np)
    print(
        "| n | p | sd 1 thread | sd 10 threads | sd jac (10) | sd nfev (iterations, GN steps) | sd jac nfev | scipy | "
        "scipy nfev | scipy jac | scipy jac nfev | R nls | R iterations | max rel. diff sd - scipy |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (n, p), (x, y, truth) in cases.items():
        p0 = p0_of(p, truth)
        reps = 5 if n <= 10_000 else 2
        f, J, _ = msd[p]
        fn, Jn, _ = mnp[p]
        xs, ys = sd.asarray(x), sd.asarray(y)
        times = {}
        for th in (1, 10):
            sd.set_num_threads(th)
            times[th], r = best(lambda: sd.regression.nls(f, xs, ys, p0=p0), reps)
        tj, rj = best(lambda: sd.regression.nls(f, xs, ys, p0=p0, jac=J), reps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts, (popt, _, info, _, _) = best(lambda: curve_fit(fn, x, y, p0=p0, full_output=True), reps)
            tsj, (_, _, infoj, _, _) = best(lambda: curve_fit(fn, x, y, p0=p0, jac=Jn, full_output=True), reps)
        diff = max(abs(a - b) / abs(b) for a, b in zip(r.params.tolist(), popt))
        rt, rit, _ = r_res.get((n, p), (float("nan"), 0, None))
        it_nfev = r.nfev
        print(
            f"| {n:,} | {p} | {ms(times[1])} | {ms(times[10])} | {ms(tj)} | {it_nfev} ({r.n_iter} it., "
            f"{int(r._raw[6].tolist()[10])} GN) | {rj.nfev} | {ms(ts)} | {info['nfev']} | {ms(tsj)} | "
            f"{infoj['nfev']} | {ms(rt)} | {rit} | {diff:.1e} |"
        )
    sd.set_num_threads(10)


if __name__ == "__main__":
    main()
