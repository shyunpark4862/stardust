"""R's lm, sandwich, lmtest, glm and glmnet on fixed problems, for sd.regression's G1-G3
tests (tests/regression/ref/regression_r.json, tests/regression/test_against_r.py;
docs/design/regression.md §3.1-§3.5). R is not a test dependency: run this once where Rscript
(with sandwich, lmtest and glmnet) is installed,

    uv run --no-sync python tools/gen_regression_r.py

What it stores, per problem (the inputs too):
  - lm (G1): lm() with tol = 1e-7 (R's default; the problem says which sd rtol matches it):
    coefficients, vcov(complete=TRUE)'s standard errors, sigma, df.residual, R^2, adjusted
    R^2, F, residuals, fitted values, logLik, AIC, hatvalues, rstandard, rstudent,
    cooks.distance; sandwich 3.1 vcovHC HC0-HC3, NeweyWest(lag, prewhite=FALSE, adjust=FALSE),
    vcovCL(type="HC1") and lmtest 0.9 coeftest p-values (df.residual, and G - 1 for clusters);
    dwtest, bptest (studentized and not). With zero weights, sandwich counts the zero-weight
    rows in n (and its HC2/HC3 recycle hatvalues of the wrong length), so the sandwich values
    are R's fit with those rows removed (sd drops them, G20).
  - glm (G2): R's glm.fit evaluated at sd's solution (start = sd's params, one Fisher step:
    maxit = 1), so the comparison has no convergence-tolerance part: coefficients, summary's
    standard errors, statistics and p-values, deviance, Pearson chi^2, dispersion, fitted
    values; for the canonical links without zero weights the sandwich and
    coeftest values; for the non-canonical links vcovHC HC0 (expected-information bread). From
    a separate fit at epsilon = 1e-13: the null deviance, logLik and AIC.
  - glmnet (G3, glmnet 5.2): coefficients at single lambdas with thresh = 1e-20, the df and
    dev.ratio, and the first lambda of the default paths (alpha = 1 and alpha = 0).
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "regression" / "ref" / "regression_r.json"

PRELUDE = r"""
suppressMessages({library(sandwich); library(lmtest); library(glmnet)})
D <- commandArgs(TRUE)[1]
rd <- function(nm) { A <- as.matrix(read.csv(file.path(D, paste0(nm, ".csv")), header=FALSE)); dimnames(A) <- NULL; A }
rv <- function(nm) drop(rd(nm))
out <- function(name, v) cat(name, ifelse(is.na(v), "NA", sprintf("%.17g", as.numeric(v))), "\n")
"""

R_LM = r"""
y <- rv("y"); X <- rd("X"); w <- rv("w"); g <- rv("g"); meta <- rv("meta")
icpt <- meta[1] == 1; L <- meta[2]; sandwich_ok <- meta[3] == 1
f <- if (icpt) lm(y ~ X, weights=w) else lm(y ~ 0 + X, weights=w)
s <- summary(f)
out("coef", coef(f)); out("se", sqrt(diag(vcov(f, complete=TRUE)))); out("p", coef(s)[, 4])
out("sigma", s$sigma); out("df_resid", df.residual(f)); out("nobs", nobs(f))
out("r2", s$r.squared); out("r2adj", s$adj.r.squared); out("fstat", s$fstatistic[1])
out("resid", residuals(f)); out("fitted", fitted(f)); out("llf", logLik(f)); out("aic", AIC(f))
out("hat", hatvalues(f)); out("rstandard", rstandard(f)); out("rstudent", rstudent(f)); out("cooks", cooks.distance(f))
if (sandwich_ok) {
  for (ty in c("HC0", "HC1", "HC2", "HC3")) {
    V <- vcovHC(f, type=ty); out(paste0("se_", ty), sqrt(diag(V))); out(paste0("p_", ty), coeftest(f, vcov.=V)[, 4])
  }
  V <- NeweyWest(f, lag=L, prewhite=FALSE, adjust=FALSE); out("se_HAC", sqrt(diag(V))); out("p_HAC", coeftest(f, vcov.=V)[, 4])
  V <- vcovCL(f, cluster=g, type="HC1"); out("se_CL", sqrt(diag(V)))
  out("p_CL", coeftest(f, vcov.=V, df=length(unique(g)) - 1)[, 4])
}
if (all(w == 1)) {
  out("dw", dwtest(f)$statistic)
  Xc <- data.frame(X=I(X))  # the auxiliary regression on a constant and X, as sd (G30)
  out("bp", bptest(f, varformula=~X, data=Xc)$statistic); out("bp_df", bptest(f, varformula=~X, data=Xc)$parameter)
  out("bp0", bptest(f, varformula=~X, data=Xc, studentize=FALSE)$statistic)
}
"""

R_GLM = r"""
y <- rv("y"); X <- rd("X"); w <- rv("w"); off <- rv("off"); g <- rv("g"); st <- rv("start"); meta <- rv("meta")
fams <- c("gaussian", "binomial", "poisson", "Gamma", "quasibinomial", "quasipoisson")
links <- c("identity", "log", "logit", "probit", "cloglog", "sqrt", "inverse")
fam <- get(fams[meta[1]])(link=links[meta[2]]); icpt <- meta[3] == 1; L <- meta[4]; robust <- meta[5]
fit <- function(ctl, start=NULL) suppressWarnings(if (icpt) glm(y ~ X, family=fam, weights=w, offset=off, control=ctl, start=start)
                                                  else glm(y ~ 0 + X, family=fam, weights=w, offset=off, control=ctl, start=start))
f <- fit(glm.control(maxit=1), st)
s <- suppressWarnings(summary(f))
out("coef", coef(f)); out("se", sqrt(diag(vcov(f)))); out("stat", coef(s)[, 3]); out("p", coef(s)[, 4])
out("deviance", f$deviance); out("pearson", sum(residuals(f, "pearson")^2)); out("dispersion", s$dispersion)
out("df_resid", f$df.residual); out("fitted", fitted(f))
if (robust == 1) {
  for (ty in c("HC0", "HC1", "HC2", "HC3")) {
    V <- vcovHC(f, type=ty); out(paste0("se_", ty), sqrt(diag(V))); out(paste0("p_", ty), coeftest(f, vcov.=V)[, 4])
  }
  V <- NeweyWest(f, lag=L, prewhite=FALSE, adjust=FALSE); out("se_HAC", sqrt(diag(V))); out("p_HAC", coeftest(f, vcov.=V)[, 4])
  V <- vcovCL(f, cluster=g, type="HC1"); out("se_CL", sqrt(diag(V)))
  out("p_CL", coeftest(f, vcov.=V, df=length(unique(g)) - 1)[, 4])
}
if (robust == 2) out("se_HC0_expected", sqrt(diag(vcovHC(f, type="HC0"))))
t <- fit(glm.control(epsilon=1e-13, maxit=500))
out("null_deviance", t$null.deviance); out("aic", t$aic)
out("llf", if (is.na(t$aic)) NA else as.numeric(logLik(t)))
"""

R_GLMNET = r"""
y <- rv("y"); X <- rd("X"); w <- rv("w"); lam <- rv("lam"); meta <- rv("meta")
fam <- c("gaussian", "binomial", "poisson")[meta[1]]; alpha <- meta[2]; std <- meta[3] == 1; icpt <- meta[4] == 1
for (i in seq_along(lam)) {
  f <- glmnet(X, y, family=fam, weights=w, alpha=alpha, lambda=lam[i], standardize=std, intercept=icpt, thresh=1e-20, maxit=1e8)
  out(paste0("coef", i), as.numeric(coef(f))); out(paste0("df", i), f$df); out(paste0("dev_ratio", i), f$dev.ratio)
}
out("lambda_max", glmnet(X, y, family=fam, weights=w, alpha=alpha, standardize=std, intercept=icpt)$lambda[1])
out("lambda_max_ridge", glmnet(X, y, family=fam, weights=w, alpha=0, standardize=std, intercept=icpt)$lambda[1])
"""

GLM_FAMS = ["gaussian", "binomial", "poisson", "Gamma", "quasibinomial", "quasipoisson"]
GLM_LINKS = ["identity", "log", "logit", "probit", "cloglog", "sqrt", "inverse"]
CANONICAL = {("gaussian", "identity"), ("binomial", "logit"), ("poisson", "log"), ("gamma", "inverse")}


def run_r(script, arrays):
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        for k, v in arrays.items():
            v = np.asarray(v, dtype=float)
            np.savetxt(d / f"{k}.csv", v[:, None] if v.ndim == 1 else v, delimiter=",", fmt="%.17g")
        (d / "run.R").write_text(PRELUDE + script)
        res = subprocess.run(["Rscript", str(d / "run.R"), str(d)], capture_output=True, text=True, check=True)
    vals = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if parts:
            vals[parts[0]] = [None if p == "NA" else float(p) for p in parts[1:]]
    return vals


def short(v):
    """Inputs rounded to 6 decimals (still exact doubles; a smaller file)."""
    return np.round(np.asarray(v, dtype=float), 6)


def lags(n):
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


# ---- lm


def lm_problems():
    rng = np.random.default_rng(11)
    P = {}
    n = 60
    X = rng.standard_normal((n, 3)) * [1.0, 3.0, 0.5] + [0, 2, -1]
    y = 1 + X @ [0.5, -1.0, 2.0] + rng.standard_normal(n) * (1 + 0.5 * np.abs(X[:, 0]))
    P["hetero"] = dict(y=y, X=X, intercept=True)
    n = 50
    X = rng.standard_normal((n, 2)) + [1.0, 0.5]
    P["noint"] = dict(y=X @ [1.0, 2.0] + rng.standard_normal(n) * (1 + np.abs(X[:, 0])), X=X, intercept=False)
    n = 25
    x1 = rng.standard_normal(n)
    P["noint_const"] = dict(y=3 + 2 * x1 + rng.standard_normal(n), X=np.column_stack([np.ones(n), x1]), intercept=False)
    n = 30
    Z = short(rng.standard_normal((n, 3)))
    X = np.column_stack([Z[:, 0], Z[:, 1], Z[:, 0] + Z[:, 1], Z[:, 2]])
    y = short(1 + X[:, 0] - X[:, 3] + rng.standard_normal(n))
    P["alias"] = dict(y=y, X=X, intercept=True)
    Xn = X.copy()
    Xn[:, 2] = X[:, 0] + X[:, 1] + 1e-10 * rng.standard_normal(n)
    P["near_alias"] = dict(y=y, X=Xn, intercept=True, rtol=1e-7)
    P["const_col"] = dict(y=y, X=np.column_stack([X[:, 0], np.ones(n), X[:, 3]]), intercept=True)
    n = 40
    X = rng.standard_normal((n, 2))
    w = rng.uniform(0.5, 2.0, n)
    w[[3, 17, 30]] = 0.0
    P["zero_weights"] = dict(y=2 + X @ [1.0, -1.0] + rng.standard_normal(n), X=X, w=w, intercept=True)
    for name, p in P.items():
        if name not in ("alias", "near_alias", "const_col"):  # built from rounded columns, exactly aliased
            for k in ("y", "X", "w"):
                if k in p:
                    p[k] = short(p[k])
    return P


def lm_ref(p):
    n = len(p["y"])
    w = p.get("w", np.ones(n))
    g = np.arange(n) % 7
    arrs = dict(y=p["y"], X=p["X"], w=w, g=g, meta=[1 if p["intercept"] else 0, lags(n), 1 if np.all(w > 0) else 0])
    r = run_r(R_LM, arrs)
    if np.any(w == 0):
        keep = w > 0
        r0 = run_r(R_LM, dict(arrs, y=p["y"][keep], X=p["X"][keep], w=w[keep], g=g[keep], meta=[1, lags(n), 1]))
        r.update({k: v for k, v in r0.items() if k.startswith(("se_", "p_"))})
    return r


# ---- glm


def _inv(link, eta):
    from scipy.stats import norm

    return dict(
        identity=lambda: eta,
        log=lambda: np.exp(eta),
        logit=lambda: 1 / (1 + np.exp(-eta)),
        probit=lambda: norm.cdf(eta),
        cloglog=lambda: -np.expm1(-np.exp(eta)),
        sqrt=lambda: eta**2,
        inverse=lambda: 1 / eta,
    )[link]()


def glm_problems():
    rng = np.random.default_rng(23)
    P = {}
    n = 60

    def design():
        return rng.standard_normal((n, 2)) * [1.0, 0.7]

    for fam, link, beta in [
        ("gaussian", "identity", [1, 0.5, -1]),
        ("gaussian", "log", [1, 0.3, -0.2]),
        ("binomial", "logit", [0.2, 1.0, -0.8]),
        ("binomial", "probit", [0.1, 0.6, -0.5]),
        ("binomial", "cloglog", [-0.3, 0.6, -0.4]),
        ("binomial", "log", [-1.2, 0.15, -0.1]),
        ("poisson", "log", [0.8, 0.4, -0.3]),
        ("poisson", "identity", [4.0, 0.8, -0.6]),
        ("poisson", "sqrt", [2.0, 0.3, -0.2]),
        ("gamma", "inverse", [1.0, 0.1, -0.1]),
        ("gamma", "log", [0.5, 0.3, -0.2]),
        ("gamma", "identity", [3.0, 0.5, -0.4]),
    ]:
        X = design()
        mu = _inv(link, beta[0] + X @ beta[1:])
        y = {
            "gaussian": lambda: mu + 0.3 * rng.standard_normal(n),
            "binomial": lambda: (rng.uniform(size=n) < mu).astype(float),
            "poisson": lambda: rng.poisson(mu).astype(float),
            "gamma": lambda: rng.gamma(2.0, mu / 2.0),
        }[fam]()
        P[f"{fam}_{link}"] = dict(y=y, X=X, family=fam, link=link)
    X = design()
    m = rng.integers(1, 12, n).astype(float)
    P["binomial_trials"] = dict(
        y=rng.binomial(m.astype(int), _inv("logit", 0.3 + X @ [0.8, -0.5])) / m,
        X=X,
        family="binomial",
        link="logit",
        trials=m,
    )
    X = design()
    off = np.log(rng.uniform(0.5, 3.0, n))
    P["poisson_offset"] = dict(
        y=rng.poisson(np.exp(0.5 + off + X @ [0.3, -0.2])).astype(float), X=X, family="poisson", link="log", offset=off
    )
    X = design()
    off = np.log(rng.uniform(0.5, 3.0, n))
    P["poisson_offset_noint"] = dict(
        y=rng.poisson(np.exp(off + X @ [0.3, -0.2])).astype(float),
        X=X,
        family="poisson",
        link="log",
        offset=off,
        intercept=False,
    )
    X = design()
    w = rng.uniform(0.5, 2.0, n)
    P["poisson_weights"] = dict(
        y=rng.poisson(np.exp(0.5 + X @ [0.3, -0.2])).astype(float), X=X, family="poisson", link="log", var_weights=w
    )
    X = design()
    w = rng.uniform(0.5, 3.0, n)
    w[5] = 0.0
    mu = np.exp(0.5 + X @ [0.3, -0.2])
    P["gamma_weights"] = dict(
        y=rng.gamma(2.0 * w + 0.5, mu / (2.0 * w + 0.5)), X=X, family="gamma", link="log", var_weights=w
    )
    X = design()
    w = rng.uniform(0.5, 3.0, n)
    w[[2, 9]] = 0.0
    P["gaussian_weights"] = dict(
        y=1 + X @ [0.5, -1] + rng.standard_normal(n) / np.sqrt(np.maximum(w, 0.1)),
        X=X,
        family="gaussian",
        link="identity",
        var_weights=w,
    )
    X = design()
    P["logit_noint"] = dict(
        y=(rng.uniform(size=n) < _inv("logit", X @ [1.0, -0.8])).astype(float),
        X=X,
        family="binomial",
        link="logit",
        intercept=False,
    )
    X = design()
    mu = np.exp(1.0 + X @ [0.4, -0.3])
    P["quasipoisson"] = dict(
        y=rng.negative_binomial(2, 2 / (2 + mu)).astype(float),
        X=X,
        family="poisson",
        link="log",
        scale="X2",
        r_family="quasipoisson",
    )
    X = design()
    m = rng.integers(5, 15, n).astype(float)
    pr = _inv("logit", 0.2 + X @ [0.8, -0.5])
    P["quasibinomial"] = dict(
        y=rng.binomial(m.astype(int), rng.beta(pr * 4, (1 - pr) * 4)) / m,
        X=X,
        family="binomial",
        link="logit",
        trials=m,
        scale="X2",
        r_family="quasibinomial",
    )
    for p in P.values():
        for k in ("y", "X", "var_weights", "offset"):
            if k in p and not (k == "y" and "trials" in p):  # proportions stay successes / trials
                p[k] = short(p[k])
    return P


def sd_glm_kwargs(p, cov_type="nonrobust", **extra):
    kw = dict(
        family=p["family"],
        link=p["link"],
        intercept=p.get("intercept", True),
        scale=p.get("scale"),
        cov_type=cov_type,
        **extra,
    )
    for k in ("trials", "offset", "var_weights"):
        if k in p:
            kw[k] = p[k]
    return kw


def glm_ref(p):
    import stardust as sd

    n = len(p["y"])
    f = sd.regression.glm(p["y"], p["X"], **sd_glm_kwargs(p))
    start = [float(v) for v in f.params.tolist()]
    w = np.asarray(p.get("var_weights", np.ones(n))) * np.asarray(p.get("trials", np.ones(n)))
    rfam = p.get("r_family", "Gamma" if p["family"] == "gamma" else p["family"])
    canonical = (p["family"], p["link"]) in CANONICAL
    robust = 0 if np.any(w == 0) else (1 if canonical else 2)
    meta = [
        GLM_FAMS.index(rfam) + 1,
        GLM_LINKS.index(p["link"]) + 1,
        1 if p.get("intercept", True) else 0,
        lags(n),
        robust,
    ]
    r = run_r(
        R_GLM,
        dict(y=p["y"], X=p["X"], w=w, off=p.get("offset", np.zeros(n)), g=np.arange(n) % 9, start=start, meta=meta),
    )
    return start, r


# ---- glmnet


def glmnet_problems():
    rng = np.random.default_rng(31)
    n, k = 80, 6
    X = rng.standard_normal((n, k)) * [1, 2, 0.5, 3, 1, 0.2] + [0, 1, -2, 0, 5, 0]
    X[:, 1] += 0.6 * X[:, 0]
    y = 5 + X @ [1.0, -0.5, 0.0, 0.3, 0.0, 2.0] + 2.5 * rng.standard_normal(n)
    w = rng.uniform(0.2, 3.0, n)
    eta = X @ [0.5, -0.3, 0.0, 0.2, 0.0, 1.0]
    eta -= eta.mean()
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    yp = rng.poisson(np.exp(0.2 + 0.3 * eta)).astype(float)
    data = dict(X=short(X), y=short(y), w=short(w), yb=yb, yp=yp)
    cases = []
    lams = [0.5, 0.1, 0.01]
    for std in (True, False):
        cases.append(dict(name=f"lasso_std{int(std)}", family="gaussian", y="y", l1=1.0, std=std, lams=lams))
    cases.append(dict(name="lasso_weights", family="gaussian", y="y", l1=1.0, std=True, lams=lams, weights=True))
    for l1 in (0.5, 0.0):  # raw y: glmnet's L2 term is lambda (1 - alpha) / sd(y) (it standardizes y)
        cases.append(dict(name=f"enet{l1}_raw_y", family="gaussian", y="y", l1=l1, std=True, lams=lams))
    cases.append(
        dict(name="enet0.5_raw_y_weights", family="gaussian", y="y", l1=0.5, std=True, lams=lams, weights=True)
    )
    for l1 in (1.0, 0.5, 0.0):
        for std in (True, False):
            cases.append(
                dict(name=f"binomial{l1}_std{int(std)}", family="binomial", y="yb", l1=l1, std=std, lams=[0.05, 0.01])
            )
    cases.append(
        dict(name="binomial0.5_weights", family="binomial", y="yb", l1=0.5, std=True, lams=[0.05, 0.01], weights=True)
    )
    for l1 in (1.0, 0.5, 0.0):
        cases.append(dict(name=f"poisson{l1}_std1", family="poisson", y="yp", l1=l1, std=True, lams=[0.05, 0.01]))
    cases.append(dict(name="poisson0.5_std0", family="poisson", y="yp", l1=0.5, std=False, lams=[0.05, 0.01]))
    return data, cases


def glmnet_ref(data, c):
    n = len(data["y"])
    w = data["w"] if c.get("weights") else np.ones(n)
    meta = [["gaussian", "binomial", "poisson"].index(c["family"]) + 1, c["l1"], 1 if c["std"] else 0, 1]
    return run_r(R_GLMNET, dict(y=data[c["y"]], X=data["X"], w=w, lam=c["lams"], meta=meta))


def tolist(v):
    return np.asarray(v).tolist()


def main():
    ver = subprocess.run(
        [
            "Rscript",
            "-e",
            'cat(R.version.string, "| sandwich", format(packageVersion("sandwich")), '
            '"| lmtest", format(packageVersion("lmtest")), "| glmnet", format(packageVersion("glmnet")))',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = {"_versions": ver.strip(), "lm": {}, "glm": {}, "glmnet": {}}
    for name, p in lm_problems().items():
        data["lm"][name] = {k: (tolist(v) if isinstance(v, np.ndarray) else v) for k, v in p.items()} | {"R": lm_ref(p)}
    for name, p in glm_problems().items():
        start, r = glm_ref(p)
        data["glm"][name] = {k: (tolist(v) if isinstance(v, np.ndarray) else v) for k, v in p.items()} | {
            "start": start,
            "R": r,
        }
    gdata, cases = glmnet_problems()
    data["glmnet"] = {
        "data": {k: tolist(v) for k, v in gdata.items()},
        "cases": [c | {"R": glmnet_ref(gdata, c)} for c in cases],
    }
    OUT.write_text(json.dumps(data))
    print(OUT, OUT.stat().st_size, "bytes;", ver)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
