"""R's robust regressions on fixed problems, for sd.regression.rlm's tests
(tests/regression/ref/robust_r.json; docs/design/regression.md §3.6). R is not a test
dependency: run this once where Rscript (with MASS and robustbase) is installed,

    uv run --no-sync python tools/gen_robust_r.py

What it stores, per problem (the inputs too):
  - MASS::rlm (7.3) with psi.huber, psi.bisquare and psi.hampel at their defaults, from least
    squares, scale.est = "proposal 2" with k2 = 2.5 (the equation of statsmodels' HuberScale(),
    sd's scale_est='huber') and scale.est = "MAD" (median |r| / 0.6745), iterated to
    acc = 1e-15: coefficients, scale, summary()'s standard errors.
  - robustbase::lmrob (0.99.7) at lmrob.control()'s defaults: coefficients, scale, the
    standard errors of vcov() (.vcov.avar1), the S start (a problem with n = 5000 takes its
    large-n fast-S).
  - From sd's own S start (rlm(method='MM', rng=default_rng(seed)); its params and scale are
    stored): robustbase's M step lmrob..M..fit at rel.tol = 1e-15 and .vcov.avar1, so the
    final step is compared without the subsamples' randomness.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "regression"))
OUT = ROOT / "tests" / "regression" / "ref" / "robust_r.json"

from refdata_robust import STACKLOSS_X, STACKLOSS_Y  # noqa: E402

R_SCRIPT = r'''
suppressMessages({library(MASS); library(robustbase)})
args <- commandArgs(TRUE)
A <- as.matrix(read.csv(args[1], header=FALSE)); st <- scan(args[2], quiet=TRUE)
y <- A[,1]; X <- A[,-1, drop=FALSE]; x <- cbind(1, X); p <- ncol(x)
bS <- st[-length(st)]; sS <- st[length(st)]
out <- function(name, v) cat(name, sprintf("%.17g", v), "\n")
psis <- list(huber=psi.huber, bisquare=psi.bisquare, hampel=psi.hampel)
for (nm in names(psis)) for (se in c("proposal 2", "MAD")) {
  tag <- paste0("mass_", nm, "_", if (se == "MAD") "mad" else "huber")
  f <- tryCatch(rlm(x, y, psi=psis[[nm]], scale.est=se, k2=2.5, maxit=1000, acc=1e-15), error=function(e) NULL)
  if (is.null(f)) next
  out(paste0(tag, "_coef"), coef(f)); out(paste0(tag, "_scale"), f$s)
  out(paste0(tag, "_se"), summary(f)$coefficients[, 2]); out(paste0(tag, "_conv"), as.numeric(f$converged))
}
f <- lmrob(y ~ X)
out("lmrob_coef", coef(f)); out("lmrob_scale", f$scale); out("lmrob_se", sqrt(diag(vcov(f))))
out("lmrob_S_coef", f$init.S$coefficients); out("lmrob_S_scale", f$init.S$scale)
ctrl <- lmrob.control(rel.tol=1e-15, max.it=1000)
m <- robustbase:::lmrob..M..fit(x=x, y=y, beta.initial=bS, scale=sS, control=ctrl)
obj <- list(control=ctrl, init=list(resid=drop(y - x %*% bS)), residuals=m$residuals, scale=sS, x=x,
            coefficients=m$coefficients, rank=p)
out("mstep_coef", m$coefficients); out("mstep_se", sqrt(diag(robustbase:::.vcov.avar1(obj, x=x))))
out("mstep_conv", as.numeric(m$converged))
'''


def problems():
    rng = np.random.default_rng(2026)
    out = {}
    out["stackloss"] = dict(y=[float(v) for v in STACKLOSS_Y], X=[[float(v) for v in r] for r in STACKLOSS_X], seed=3)
    n, p = 200, 3
    X = rng.standard_normal((n, p)) + 1.0
    y = X @ np.array([1.0, -0.5, 2.0]) + 0.3 + rng.standard_normal(n)
    y[:20] += rng.uniform(6.0, 12.0, 20)
    out["vertical"] = dict(y=y.tolist(), X=X.tolist(), seed=5)
    for name, frac, n in (("leverage20", 0.2, 300), ("leverage30", 0.3, 500)):
        X = rng.standard_normal((n, 2))
        y = X @ np.array([1.0, 2.0]) + 0.5 + 0.5 * rng.standard_normal(n)
        m = int(frac * n)
        X[:m] = rng.normal(5.0, 0.5, (m, 2))
        y[:m] = rng.normal(-10.0, 1.0, m)
        out[name] = dict(y=y.tolist(), X=X.tolist(), seed=7)
    # beyond 2000 rows: robustbase's large-n fast-S (G58)
    n = 5000
    X = rng.standard_normal((n, 3))
    y = X @ np.array([1.0, 2.0, -1.0]) + 0.5 + 0.5 * rng.standard_normal(n)
    m = n // 4
    X[:m] = rng.normal(5.0, 0.5, (m, 3))
    y[:m] = rng.normal(-10.0, 1.0, m)
    out["large25"] = dict(y=y.tolist(), X=X.tolist(), seed=13)
    return out


def run(prob):
    import stardust as sd

    r = sd.regression.rlm(prob["y"], prob["X"], method="MM", rng=sd.random.default_rng(prob["seed"]))
    bS = [float(v) for v in r._raw[8].tolist()]
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        np.savetxt(d / "data.csv", np.column_stack([prob["y"], prob["X"]]), delimiter=",", fmt="%.17g")
        np.savetxt(d / "start.txt", np.array(bS + [float(r.scale)]), fmt="%.17g")
        (d / "run.R").write_text(R_SCRIPT)
        res = subprocess.run(["Rscript", str(d / "run.R"), str(d / "data.csv"), str(d / "start.txt")],
                             capture_output=True, text=True, check=True)
    vals = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if parts:
            vals[parts[0]] = [float(v) for v in parts[1:]]
    return dict(prob, S_params=bS, S_scale=float(r.scale), R=vals)


def main():
    ver = subprocess.run(["Rscript", "-e", 'cat(R.version.string, "|", "MASS", format(packageVersion("MASS")), '
                          '"| robustbase", format(packageVersion("robustbase")))'], capture_output=True, text=True,
                         check=True).stdout
    data = {"_versions": ver.strip()}
    for name, prob in problems().items():
        data[name] = run(prob)
    OUT.write_text(json.dumps(data))
    print(OUT, OUT.stat().st_size, "bytes;", ver)


if __name__ == "__main__":
    main()
