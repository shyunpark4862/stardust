"""R's variable selection, for the comparison of sd.regression's stepwise and best_subset
(tests/regression/data/selection_r.json, tests/regression/test_selection_against_r.py;
docs/design/regression.md §3.8). R is not a test dependency: run this once where Rscript with MASS
and leaps is installed,

    .venv/bin/python tools/gen_selection_r.py

What it stores, per problem (the inputs too):
  - lm (OLS, WLS with zero weights, no intercept, a forced column, a factor's dummies as one term,
    ten correlated columns with weak effects): step from the
    full model (both, backward; k = 2 and log(n)), from the intercept-only model (forward, both,
    with the full scope), MASS::stepAIC from the full model, the step path (R's anova component:
    the term moved and the AIC after it), add1 and drop1 with test = "F" of a middle model, and
    leaps::regsubsets(method = "exhaustive", nvmax = all) with summary's which, rss, cp, adjr2
    and bic (force.in where the problem has `include`).
  - glm (binomial logit, poisson log, gamma log): step from the full model (k = 2 and log(n)),
    its path, add1 and drop1 with test = "Chisq" of a middle model.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "regression" / "data" / "selection_r.json"

PRELUDE = r"""
suppressMessages({library(MASS); library(leaps)})
D <- commandArgs(TRUE)[1]
rd <- function(nm) { A <- as.matrix(read.csv(file.path(D, paste0(nm, ".csv")), header=FALSE))
dimnames(A) <- NULL
A }
rv <- function(nm) drop(rd(nm))
out <- function(name, v) cat(name, ifelse(is.na(v), "NA", sprintf("%.17g", as.numeric(v))), "\n")
outs <- function(name, v) cat(name, if (length(v)) v else "-", "\n")
# the terms of a fit (sorted by position) and the path's moves ("- x2", "+ x3") and AIC values
terms_of <- function(f) { tl <- attr(terms(f), "term.labels"); tl[order(match(tl, all_terms))] }
path <- function(name, f) { a <- f$anova; outs(paste0(name, "_moves"), gsub(" ", "", a$Step[-1]))
  out(paste0(name, "_aic"), a$AIC); outs(paste0(name, "_terms"), terms_of(f)) }
"""

R_LM = r"""
y <- rv("y"); X <- rd("X"); w <- rv("w"); g <- rv("g"); meta <- rv("meta")
icpt <- meta[1] == 1; has_factor <- meta[2] == 1; nforce <- meta[3]
p <- ncol(X); n <- length(y)
dat <- data.frame(y = y, w = w)
nm <- paste0("x", seq_len(p))
for (j in seq_len(p)) dat[[nm[j]]] <- X[, j]
all_terms <- nm
if (has_factor) {  # the last two columns are the treatment dummies of the factor g
  dat$f <- factor(g)
  all_terms <- c(nm[seq_len(p - 2)], "f")
}
lhs <- if (icpt) "y ~ " else "y ~ 0 + "
fb <- as.formula(paste(lhs, paste(all_terms, collapse = " + ")))
# the smallest model: the intercept and the forced terms (without the intercept, the first term)
lower_terms <- if (nforce > 0) all_terms[seq_len(nforce)] else if (icpt) character(0) else all_terms[1]
f0 <- as.formula(paste(lhs, if (length(lower_terms)) paste(lower_terms, collapse = " + ") else "1"))
sc <- list(lower = f0, upper = fb)
big <- lm(fb, data = dat, weights = w)
nobs <- sum(w > 0)
path("both", step(big, scope = sc, trace = 0))
path("both_bic", step(big, scope = sc, trace = 0, k = log(nobs)))
path("backward", step(big, scope = sc, direction = "backward", trace = 0))
path("stepaic", stepAIC(big, scope = sc, trace = 0))
small <- lm(f0, data = dat, weights = w)
path("forward", step(small, scope = sc, direction = "forward", trace = 0))
path("forward_bic", step(small, scope = sc, direction = "forward", trace = 0, k = log(nobs)))
path("both_empty", step(small, scope = sc, direction = "both", trace = 0))
mid <- lm(as.formula(paste(lhs, paste(all_terms[1:2], collapse = " + "))), data = dat, weights = w)
a <- add1(mid, scope = fb, test = "F"); out("add1_F", a$`F value`[-1]); out("add1_p", a$`Pr(>F)`[-1])
outs("add1_terms", rownames(a)[-1])
d <- drop1(big, test = "F"); out("drop1_F", d$`F value`[-1]); out("drop1_p", d$`Pr(>F)`[-1])
if (!has_factor) {
  Xr <- X[w > 0, , drop = FALSE]; yr <- y[w > 0]; wr <- w[w > 0]
  fi <- if (nforce > 0) seq_len(nforce) else NULL
  r <- regsubsets(Xr, yr, weights = wr, nvmax = p, intercept = icpt, force.in = fi, method = "exhaustive")
  s <- summary(r)
  W <- s$which
  if (icpt) W <- W[, -1, drop = FALSE]
  out("leaps_which", t(W) * 1)
  out("leaps_rss", s$rss); out("leaps_cp", s$cp); out("leaps_adjr2", s$adjr2); out("leaps_bic", s$bic)
}
"""

R_GLM = r"""
y <- rv("y"); X <- rd("X"); meta <- rv("meta")
fams <- c("gaussian", "binomial", "poisson", "Gamma")
links <- c("identity", "log", "logit", "probit", "cloglog", "sqrt", "inverse")
fam <- get(fams[meta[1]])(link = links[meta[2]])
p <- ncol(X); n <- length(y)
dat <- data.frame(y = y)
nm <- paste0("x", seq_len(p))
all_terms <- nm
for (j in seq_len(p)) dat[[nm[j]]] <- X[, j]
ctl <- glm.control(epsilon = 1e-14, maxit = 200)
fb <- as.formula(paste("y ~", paste(nm, collapse = " + ")))
big <- glm(fb, family = fam, data = dat, control = ctl)
path("both", step(big, trace = 0))
path("both_bic", step(big, trace = 0, k = log(n)))
small <- glm(y ~ 1, family = fam, data = dat, control = ctl)
path("forward", step(small, scope = fb, direction = "forward", trace = 0))
mid <- glm(as.formula(paste("y ~", paste(nm[1:2], collapse = " + "))), family = fam, data = dat, control = ctl)
a <- add1(mid, scope = fb, test = "Chisq"); out("add1_lrt", a[-1, ncol(a) - 1]); out("add1_p", a[-1, ncol(a)])
d <- drop1(big, test = "Chisq"); out("drop1_lrt", d[-1, ncol(d) - 1]); out("drop1_p", d[-1, ncol(d)])
"""


def run_r(script, arrays):
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        for k, v in arrays.items():
            v = np.asarray(v, dtype=float)
            np.savetxt(d / f"{k}.csv", v[:, None] if v.ndim == 1 else v, delimiter=",", fmt="%.17g")
        (d / "run.R").write_text(PRELUDE + script)
        res = subprocess.run(["Rscript", str(d / "run.R"), str(d)], capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(res.stderr)
    vals = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        key, rest = parts[0], parts[1:]
        if key.endswith(("_moves", "_terms")):
            vals[key] = [] if rest == ["-"] else rest
        else:
            vals[key] = [None if v == "NA" else float(v) for v in rest]
    return vals


def short(v):
    return np.round(np.asarray(v, dtype=float), 6)


def lm_problems():
    rng = np.random.default_rng(20261121)
    out = {}
    n, p = 120, 6
    X = short(rng.standard_normal((n, p)))
    X[:, 3] = short(X[:, 3] + 0.6 * X[:, 0])  # some correlation
    y = short(1.0 + X @ [0.9, 0.0, -0.5, 0.0, 0.25, 0.12] + rng.standard_normal(n))
    out["ols"] = dict(y=y, X=X, w=np.ones(n), g=np.zeros(n), meta=[1, 0, 0])
    n = 90
    Xw = short(rng.standard_normal((n, 5)))
    ww = short(rng.uniform(0.3, 3.0, n))
    ww[[4, 40, 77]] = 0.0
    yw = short(2.0 + Xw @ [1.0, 0.0, -0.6, 0.2, 0.0] + rng.standard_normal(n) / np.sqrt(np.maximum(ww, 0.3)))
    out["wls"] = dict(y=yw, X=Xw, w=ww, g=np.zeros(n), meta=[1, 0, 0])
    n = 70
    Xn = short(rng.standard_normal((n, 5)) + 0.5)
    yn = short(Xn @ [0.7, 0.0, 0.4, -0.3, 0.0] + rng.standard_normal(n))
    out["noint"] = dict(y=yn, X=Xn, w=np.ones(n), g=np.zeros(n), meta=[0, 0, 0])
    # include: the first column forced in (leaps' force.in)
    out["include"] = dict(y=y, X=X, w=np.ones(len(y)), g=np.zeros(len(y)), meta=[1, 0, 1])
    n = 100
    g = rng.integers(0, 3, n)
    Xf = np.column_stack([short(rng.standard_normal((n, 3))), (g == 1).astype(float), (g == 2).astype(float)])
    yf = short(0.5 + Xf @ [0.8, 0.0, 0.3, 0.6, -0.4] + rng.standard_normal(n))
    out["factor"] = dict(y=yf, X=Xf, w=np.ones(n), g=g.astype(float), meta=[1, 1, 0])
    # ten correlated columns and weak effects: the directions and the criteria part ways
    n, p = 60, 10
    Z = rng.standard_normal((n, p))
    X10 = short(Z + 0.5 * Z[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
    y10 = short(X10 @ [0.5, -0.3, 0.0, 0.25, 0.0, 0.0, 0.2, -0.15, 0.0, 0.1] + rng.standard_normal(n))
    out["ols10"] = dict(y=y10, X=X10, w=np.ones(n), g=np.zeros(n), meta=[1, 0, 0])
    return out


def glm_problems():
    rng = np.random.default_rng(1121)
    out = {}
    n, p = 200, 5
    X = short(rng.standard_normal((n, p)))
    eta = 0.2 + X @ [0.9, 0.0, -0.6, 0.0, 0.25]
    out["binomial_logit"] = dict(
        y=(rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float), X=X, family="binomial", link="logit"
    )
    out["poisson_log"] = dict(y=rng.poisson(np.exp(0.5 * eta)).astype(float), X=X, family="poisson", link="log")
    out["gamma_log"] = dict(y=short(rng.gamma(3.0, np.exp(0.5 * eta) / 3.0)), X=X, family="gamma", link="log")
    return out


FAMS = ["gaussian", "binomial", "poisson", "gamma"]
LINKS = ["identity", "log", "logit", "probit", "cloglog", "sqrt", "inverse"]


def tolist(v):
    return np.asarray(v).tolist()


def main():
    ver = subprocess.run(
        [
            "Rscript",
            "-e",
            'cat(R.version.string, "| MASS", format(packageVersion("MASS")), '
            '"| leaps", format(packageVersion("leaps")))',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = {"_versions": ver.strip(), "lm": {}, "glm": {}}
    for name, p in lm_problems().items():
        data["lm"][name] = {k: tolist(v) for k, v in p.items()} | {"R": run_r(R_LM, p)}
    for name, p in glm_problems().items():
        meta = [FAMS.index(p["family"]) + 1, LINKS.index(p["link"]) + 1]
        data["glm"][name] = {k: (tolist(v) if isinstance(v, np.ndarray) else v) for k, v in p.items()} | {
            "R": run_r(R_GLM, dict(y=p["y"], X=p["X"], meta=meta))
        }
    OUT.write_text(json.dumps(data))
    print(OUT, OUT.stat().st_size, "bytes;", ver)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
