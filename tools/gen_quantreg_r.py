"""R's quantile regressions on fixed problems, for sd.regression.quantreg's tests
(tests/regression/data/quantreg_r.json; docs/design/regression.md §3.6). R is not a test
dependency: run this once where Rscript (with quantreg) is installed,

    .venv/bin/python tools/gen_quantreg_r.py

What it stores, per problem (the inputs too) and quantile:
  - quantreg::rq (6.1) with method = "br" (Barrodale-Roberts simplex: an exact vertex) and
    method = "fn" (Frisch-Newton interior point, its default tolerance): coefficients, the
    objective sum rho_q(resid), and whether rq.fit.br warned "Solution may be nonunique".
  - summary.rq of the "br" fit with se = "nid", "iid" and "ker" (hs = TRUE: Hall-Sheather,
    and hs = FALSE: Bofinger): standard errors (NA where R fails). summary.rq's eps
    (.Machine$double.eps^(1/2)) is absolute, sd's is that times s = max |y_i| (G67): the
    standard errors are summary.rq's with its eps multiplied by s (the same as summary.rq on
    y / s times s, without rounding y / s).
  - summary.rq's "iid" and "ker" (the same eps) at sd.regression.quantreg's own vertex (its
    coefficients and residuals put into the "br" fit object; they differ where the solution is not
    unique and R's "br" picks another vertex).
The problems: Engel's food expenditure data (quantreg's engel), Brownlee's stack loss,
normal and heteroscedastic random designs, n = 5000, extreme quantiles, a model without an
intercept, and non-unique cases (a median of an even sample, integer data on dummies and on
a grid).
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "regression"))
OUT = ROOT / "tests" / "regression" / "data" / "quantreg_r.json"

from refdata_robust import STACKLOSS_X, STACKLOSS_Y  # noqa: E402

R_SCRIPT = r"""
suppressMessages(library(quantreg))
args <- commandArgs(TRUE)
A <- as.matrix(read.csv(args[1], header=FALSE)); taus <- scan(args[2], quiet=TRUE); icpt <- as.integer(args[3])
SD <- as.matrix(read.csv(args[4], header=FALSE))
y <- A[,1]; X <- A[,-1, drop=FALSE]
out <- function(name, v) cat(name, sprintf("%.17g", v), "\n")
# summary.rq's eps is absolute; sd's is 2^-26 s, s = max|y| (G67): summary.rq with its eps
# times s (the same as summary.rq on y / s with the standard errors times s, without
# rounding y / s)
s <- max(abs(y)); if (s == 0) s <- 1
src <- deparse(quantreg:::summary.rq)
at <- grep("eps <- .Machine$double.eps^(1/2)", src, fixed=TRUE)
stopifnot(length(at) == 1)
src[at] <- sub(".Machine$double.eps^(1/2)", "(.Machine$double.eps^(1/2) * .sd_scale)", src[at], fixed=TRUE)
summary_rel <- eval(parse(text=src)); environment(summary_rel) <- asNamespace("quantreg")
assign(".sd_scale", s, envir=globalenv())
fitq <- function(tau, method) {
  nu <- FALSE
  f <- withCallingHandlers(
    if (ncol(X) == 0) rq(y ~ 1, tau=tau, method=method)
    else if (icpt == 1) rq(y ~ X, tau=tau, method=method) else rq(y ~ X - 1, tau=tau, method=method),
    warning=function(w) { if (grepl("nonunique", conditionMessage(w))) nu <<- TRUE; invokeRestart("muffleWarning") })
  list(f=f, nu=nu)
}
obj <- function(r, tau) sum(r * (tau - (r < 0)))
se <- function(f, type, hs) tryCatch(suppressWarnings(summary_rel(f, se=type, hs=hs)$coefficients[, 2]),
                                    error=function(e) rep(NA, length(coef(f))))
for (j in seq_along(taus)) {
  tau <- taus[j]; tag <- paste0("t", j - 1, "_")
  b <- fitq(tau, "br"); fn <- fitq(tau, "fn")
  out(paste0(tag, "br"), coef(b$f)); out(paste0(tag, "brnu"), as.numeric(b$nu))
  out(paste0(tag, "obj_br"), obj(resid(b$f), tau))
  out(paste0(tag, "fn"), coef(fn$f)); out(paste0(tag, "obj_fn"), obj(resid(fn$f), tau))
  for (type in c("nid", "iid", "ker")) {
    out(paste0(tag, "se_", type), se(b$f, type, TRUE))
    out(paste0(tag, "se_", type, "_bof"), se(b$f, type, FALSE))
  }
  v <- b$f; v$coefficients[] <- SD[j, ]; v$residuals[] <- y - model.matrix(v) %*% SD[j, ]
  for (type in c("iid", "ker")) out(paste0(tag, "sdv_se_", type), se(v, type, TRUE))
}
"""


def engel():
    res = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "suppressMessages(library(quantreg)); data(engel); "
            'cat(sprintf("%.17g %.17g", engel$income, engel$foodexp), sep="\\n")',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [[float(v) for v in line.split()] for line in res.stdout.splitlines() if line.strip()]
    return [r[1] for r in rows], [[r[0]] for r in rows]


def problems():
    rng = np.random.default_rng(2027)
    out = {}
    y, X = engel()
    out["engel"] = dict(y=y, X=X, q=[0.1, 0.25, 0.5, 0.75, 0.9], intercept=True)
    out["stackloss"] = dict(
        y=[float(v) for v in STACKLOSS_Y],
        X=[[float(v) for v in r] for r in STACKLOSS_X],
        q=[0.25, 0.5, 0.75],
        intercept=True,
    )
    n = 200
    X = rng.standard_normal((n, 2))
    y = X @ np.array([2.0, 3.0]) + 1.0 + rng.standard_normal(n)
    out["normal200"] = dict(y=y.tolist(), X=X.tolist(), q=[0.05, 0.5, 0.95], intercept=True)
    n = 1000
    X = rng.uniform(0.0, 4.0, (n, 3))
    y = X @ np.array([1.0, -1.0, 0.5]) + 2.0 + (0.5 + X[:, 0]) * rng.standard_normal(n)
    out["hetero1000"] = dict(y=y.tolist(), X=X.tolist(), q=[0.1, 0.5, 0.9], intercept=True)
    n = 5000
    X = rng.standard_normal((n, 5))
    y = X @ np.array([1.0, 2.0, -1.0, 0.5, 0.0]) + 0.5 + rng.standard_t(3, n)
    out["t5000"] = dict(y=y.tolist(), X=X.tolist(), q=[0.25, 0.5], intercept=True)
    n = 500
    X = rng.standard_normal((n, 2))
    y = X @ np.array([1.0, 1.0]) + rng.standard_normal(n)
    out["tails500"] = dict(y=y.tolist(), X=X.tolist(), q=[0.01, 0.99], intercept=True)
    n = 300
    X = rng.uniform(1.0, 3.0, (n, 2))
    y = X @ np.array([1.5, -0.5]) + rng.exponential(1.0, n)
    out["noint300"] = dict(y=y.tolist(), X=X.tolist(), q=[0.3, 0.5], intercept=False)
    # non-unique: the median of an even sample, integer outcomes on dummies, a grid
    out["median10"] = dict(
        y=[float(v) for v in rng.permutation(np.arange(1, 11))], X=[[] for _ in range(10)], q=[0.5, 0.3], intercept=True
    )
    g = np.repeat([0.0, 1.0, 2.0], 20)
    D = np.column_stack([(g == 1).astype(float), (g == 2).astype(float)])
    y = rng.poisson(3.0 + g, g.size).astype(float)
    out["dummies60"] = dict(y=y.tolist(), X=D.tolist(), q=[0.5, 0.25, 0.75], intercept=True)
    x = np.tile(np.arange(1.0, 9.0), 5)
    y = np.round(x * 0.5 + rng.normal(0.0, 1.0, x.size))
    out["grid40"] = dict(y=y.tolist(), X=x[:, None].tolist(), q=[0.5, 0.2], intercept=True)
    return out


def run(prob):
    import stardust as sd

    X = np.asarray(prob["X"], dtype=float).reshape(len(prob["y"]), -1)
    sdv = [sd.regression.quantreg(prob["y"], X, q=q, intercept=prob["intercept"]).params.tolist() for q in prob["q"]]
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        np.savetxt(d / "data.csv", np.column_stack([prob["y"], X]), delimiter=",", fmt="%.17g")
        np.savetxt(d / "taus.txt", np.array(prob["q"]), fmt="%.17g")
        np.savetxt(d / "sd.csv", np.array(sdv).reshape(len(prob["q"]), -1), delimiter=",", fmt="%.17g")
        (d / "run.R").write_text(R_SCRIPT)
        res = subprocess.run(
            [
                "Rscript",
                "--vanilla",
                str(d / "run.R"),
                str(d / "data.csv"),
                str(d / "taus.txt"),
                "1" if prob["intercept"] else "0",
                str(d / "sd.csv"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    vals = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if parts:
            vals[parts[0]] = [None if v == "NA" else float(v) for v in parts[1:]]
    return dict(prob, sd_params=sdv, R=vals)


def main():
    ver = subprocess.run(
        ["Rscript", "--vanilla", "-e", 'cat(R.version.string, "|", "quantreg", format(packageVersion("quantreg")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = {"_versions": ver.strip()}
    for name, prob in problems().items():
        data[name] = run(prob)
    OUT.write_text(json.dumps(data))
    print(OUT, OUT.stat().st_size, "bytes;", ver)


if __name__ == "__main__":
    main()
