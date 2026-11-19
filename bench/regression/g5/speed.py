"""G5: the speed of quantreg against R's quantreg 6.1 (rq.fit with method "br" and "fn") and
statsmodels' QuantReg (docs/design/regression.md §3.6). Results: bench/regression/g5/speed.md
(the raw output in speed.log).

Data: X standard normal (n, k), y = X beta + 0.5 + Student t(3) noise. sd: quantreg(y, X, q=...)
with cov_type='iid' (the fit and a small median regression for the sparsity: the closest to
R's rq.fit, which computes no standard errors) and with the default 'nid' (two more fits at
q +- h); 1 and 10 threads. R: the same data (written as doubles), rq.fit(cbind(1, X), y, tau,
method) timed by system.time inside one R session (no start-up; 100 calls per timing at n =
1000), best of several runs;
summary.rq(se = "nid") on the "fn" fit for the default standard errors. "br" is skipped at
n = 10^6 (its simplex from scratch takes minutes there). statsmodels: QuantReg(y,
add_constant(X)).fit(q) at its defaults (IRLS, p_tol 1e-6, max_iter 1000; its standard errors
are computed in fit), its BLAS threads by threadpoolctl. "objective excess" is statsmodels'
sum rho_q(r) over sd's, minus 1.

Run: uv run --no-sync python bench/regression/g5/speed.py
"""

import pathlib
import shutil
import subprocess
import tempfile
import time
import warnings

import numpy as np

import stardust as sd

QS9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CASES = ((1_000, 5), (100_000, 10), (1_000_000, 10))


def best(f, reps):
    out, v = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        v = f()
        out.append(time.perf_counter() - t0)
    return min(out), v


def ms(t):
    return f"{t * 1e3:,.3g}" if t < 10 else f"{t * 1e3:,.0f}"


def data(n, k, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    y = X @ np.linspace(1.0, 2.0, k) + 0.5 + rng.standard_t(3, n)
    return y, X


R_SCRIPT = r"""
suppressMessages(library(quantreg))
args <- commandArgs(TRUE)
n <- as.integer(args[1]); k <- as.integer(args[2]); reps <- as.integer(args[5])
X <- matrix(readBin(args[3], "double", n * k), n, k, byrow=TRUE); y <- readBin(args[4], "double", n)
x <- cbind(1, X)
inner <- max(1L, as.integer(1e5 %/% n))  # calls per timing: system.time resolves 1 ms
tm <- function(f) min(sapply(seq_len(reps), function(i) system.time(for (j in seq_len(inner)) f())[["elapsed"]])) / inner
taus <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
cat("fn", tm(function() rq.fit(x, y, tau=0.5, method="fn")), "\n")
f <- rq(y ~ X, tau=0.5, method="fn")
cat("fn_nid", tm(function() { g <- rq(y ~ X, tau=0.5, method="fn"); suppressWarnings(summary(g, se="nid")) }), "\n")
if (n <= 100000) cat("br", tm(function() rq.fit(x, y, tau=0.5, method="br")), "\n")
if (n == 100000) {
  cat("fn9", tm(function() for (t in taus) rq.fit(x, y, tau=t, method="fn")), "\n")
  cat("br9", system.time(for (t in taus) rq.fit(x, y, tau=t, method="br"))[["elapsed"]], "\n")
}
cat("nit", rq.fit.fnb(x, y, tau=0.5)$nit[1], "\n")
"""


def r_times(y, X, reps):
    if shutil.which("Rscript") is None:
        return {}
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        X.astype(np.float64).tofile(d / "X.bin")
        y.astype(np.float64).tofile(d / "y.bin")
        (d / "t.R").write_text(R_SCRIPT)
        out = subprocess.run(
            ["Rscript", str(d / "t.R"), str(len(y)), str(X.shape[1]), str(d / "X.bin"), str(d / "y.bin"), str(reps)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return {ln.split()[0]: float(ln.split()[1]) for ln in out.splitlines() if ln.strip()}


def rho(r, q):
    return float(np.sum(r * (q - (r < 0))))


def main():
    import statsmodels.api as sm
    from threadpoolctl import threadpool_limits

    print("## one quantile (q = 0.5), ms")
    print(
        "| (n, k) | threads | sd iid | sd nid | sd n_iter / pivots | R fn | R fn + nid | R br | R fn n_iter | "
        "statsmodels | sm iterations | sm objective excess |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n, k in CASES:
        y, X = data(n, k)
        ys, Xs = sd.asarray(y), sd.asarray(X)
        reps = 20 if n <= 1_000 else (5 if n <= 100_000 else 3)
        rt = r_times(y, X, reps)
        Xc = sm.add_constant(X)
        for th in (1, 10):
            sd.set_num_threads(th)
            ti, r = best(lambda: sd.regression.quantreg(ys, Xs, q=0.5, cov_type="iid"), reps)
            tn, _ = best(lambda: sd.regression.quantreg(ys, Xs, q=0.5), reps)
            with threadpool_limits(th), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tm, s = best(lambda: sm.QuantReg(y, Xc).fit(q=0.5), 1 if n >= 1_000_000 else reps)
            ex = rho(y - Xc @ s.params, 0.5) / r.objective - 1
            rbr = ms(rt["br"]) if "br" in rt else ""
            print(
                f"| ({n:,}, {k}) | {th} | {ms(ti)} | {ms(tn)} | {r.n_iter} / {r.n_pivots} | {ms(rt.get('fn', np.nan))} | "
                f"{ms(rt.get('fn_nid', np.nan))} | {rbr} | {int(rt.get('nit', 0))} | {ms(tm)} | {s.iterations} | "
                f"{ex:.1e} |",
                flush=True,
            )
    print()
    print("## nine quantiles 0.1, ..., 0.9 at (100,000, 10), ms")
    print(
        "| threads | sd q array (warm) | sd one call per q | pivots per q (warm) | R fn, 9 calls | R br, 9 calls | "
        "statsmodels, 9 calls |"
    )
    print("|---:|---:|---:|---|---:|---:|---:|")
    y, X = data(100_000, 10)
    ys, Xs = sd.asarray(y), sd.asarray(X)
    rt = r_times(y, X, 3)
    Xc = sm.add_constant(X)
    for th in (1, 10):
        sd.set_num_threads(th)
        tw, r = best(lambda: sd.regression.quantreg(ys, Xs, q=QS9, cov_type="iid"), 3)
        tc, _ = best(lambda: [sd.regression.quantreg(ys, Xs, q=q, cov_type="iid") for q in QS9], 3)
        with threadpool_limits(th), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tm, _ = best(lambda: [sm.QuantReg(y, Xc).fit(q=q) for q in QS9], 1)
        piv = "/".join(str(v) for v in r.n_pivots.tolist())
        print(
            f"| {th} | {ms(tw)} | {ms(tc)} | {piv} | {ms(rt.get('fn9', np.nan))} | {ms(rt.get('br9', np.nan))} | "
            f"{ms(tm)} |",
            flush=True,
        )
    sd.set_num_threads(10)


if __name__ == "__main__":
    main()
