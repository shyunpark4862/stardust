"""R's tests on fitted models, for the comparison of sd.regression's
(tests/regression/data/model_tests_r.json, tests/regression/test_model_tests_against_r.py;
docs/design/regression.md §3.8). R is not a test dependency: run this once where Rscript with
lmtest and car is installed,

    .venv/bin/python tools/gen_model_tests_r.py

What it stores, per problem (the inputs too):
  - lm: car::linearHypothesis (classical and with vcov = sandwich's vcovHC HC3), each row's
    estimate, standard error, t and p from vcov (R arithmetic), anova(small, big), anova(big)
    (Type I), drop1(big, test = "F") and car::Anova(big, type = 3) (with the HC3 covariance
    too), lmtest's waldtest and lrtest of small against big, car::vif, dffits, dfbetas, gqtest
    (the three alternatives, a fraction, order.by), resettest (fitted and regressor) and
    bptest with White's auxiliary formula (cross terms and not).
  - glm (fitted with epsilon = 1e-14 from sd's solution; R's sequential refits of anova(big)
    start from scratch): anova(big) and anova(small, big) with R's default test,
    drop1(big) (the likelihood ratio), car::Anova(big, type = 3, test.statistic = "Wald"),
    car::linearHypothesis, lmtest::lrtest, hatvalues, cooks.distance, dffits, dfbetas.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "regression" / "data" / "model_tests_r.json"

PRELUDE = r"""
suppressMessages({library(sandwich); library(lmtest); library(car)})
D <- commandArgs(TRUE)[1]
rd <- function(nm) { A <- as.matrix(read.csv(file.path(D, paste0(nm, ".csv")), header=FALSE))
dimnames(A) <- NULL
A }
rv <- function(nm) drop(rd(nm))
out <- function(name, v) cat(name, ifelse(is.na(v), "NA", sprintf("%.17g", as.numeric(v))), "\n")
"""

R_LM = r"""
y <- rv("y"); X <- rd("X"); w <- rv("w"); Rm <- rd("R"); q <- rv("q"); g <- rv("g"); meta <- rv("meta")
icpt <- meta[1] == 1; nsmall <- meta[2]; has_factor <- meta[3] == 1; ord <- meta[4]
p <- ncol(X)
dat <- data.frame(y = y, w = w)
nm <- paste0("x", seq_len(p))
for (j in seq_len(p)) dat[[nm[j]]] <- X[, j]
form_terms <- nm
if (has_factor) {  # the last two columns are the treatment dummies of the factor g
  dat$f <- factor(g)
  form_terms <- c(nm[seq_len(p - 2)], "f")
}
fb <- as.formula(paste("y ~", if (icpt) "" else "0 +", paste(form_terms, collapse = " + ")))
fs <- as.formula(paste("y ~", if (icpt) "" else "0 +", paste(nm[seq_len(nsmall)], collapse = " + ")))
big <- lm(fb, data = dat, weights = w); small <- lm(fs, data = dat, weights = w)
V <- vcov(big)
b <- coef(big)
est <- drop(Rm %*% b); se <- sqrt(diag(Rm %*% V %*% t(Rm)))
out("t_effect", est); out("t_sd", se); out("t_tvalue", (est - q) / se)
out("t_pvalue", 2 * pt(abs((est - q) / se), df.residual(big), lower.tail = FALSE))
lh <- linearHypothesis(big, Rm, q); out("lh_F", lh$F[2]); out("lh_p", lh[2, "Pr(>F)"]); out("lh_df", lh$Df[2])
Vh <- vcovHC(big, type = "HC3")
lh <- linearHypothesis(big, Rm, q, vcov. = Vh); out("lh_hc3_F", lh$F[2]); out("lh_hc3_p", lh[2, "Pr(>F)"])
a <- anova(small, big)
out("anova2_F", a$F[2])
out("anova2_p", a[2, "Pr(>F)"])
out("anova2_ssdiff", a[2, "Sum of Sq"])
a <- anova(small, big, test = "Chisq"); out("anova2_chisq_p", a[2, "Pr(>Chi)"])
a <- anova(big)
out("anova1_df", a$Df)
out("anova1_ss", a[["Sum Sq"]])
out("anova1_F", a[["F value"]])
out("anova1_p", a[["Pr(>F)"]])
d <- drop1(big, test = "F")
out("drop1_df", d$Df[-1])
out("drop1_ss", d[["Sum of Sq"]][-1])
out("drop1_F", d[["F value"]][-1])
out("drop1_p", d[["Pr(>F)"]][-1])
if (icpt) {
  A3 <- Anova(big, type = 3); k3 <- nrow(A3) - 1
  out("anova3_F", A3$F[2:k3]); out("anova3_p", A3[["Pr(>F)"]][2:k3]); out("anova3_ss", A3[["Sum Sq"]][2:k3])
  A3 <- Anova(big, type = 3, white.adjust = "hc3")
  out("anova3_hc3_F", A3$F[2:k3]); out("anova3_hc3_p", A3[["Pr(>F)"]][2:k3])
  if (!has_factor) out("vif", vif(big))
}
wt <- waldtest(small, big); out("waldtest_F", wt$F[2]); out("waldtest_p", wt[2, "Pr(>F)"])
lr <- lrtest(small, big); out("lrtest", lr$Chisq[2]); out("lrtest_p", lr[2, "Pr(>Chisq)"])
out("dffits", dffits(big)); out("dfbetas", t(dfbetas(big)))
if (all(w == 1)) {
  big <- lm(fb, data = dat)  # no (weights) column in the model frame (resettest's regressors)
  for (alt in c("greater", "less", "two.sided")) {
    gq <- gqtest(big, alternative = alt); out(paste0("gq_", alt), c(gq$statistic, gq$p.value, gq$parameter))
  }
  gq <- gqtest(big, fraction = 0.2, order.by = X[, ord])
  out("gq_frac_order", c(gq$statistic, gq$p.value, gq$parameter))
  rs <- resettest(big, power = 2:3, type = "fitted"); out("reset_fitted", c(rs$statistic, rs$p.value, rs$parameter))
  if (!has_factor) {
    rs <- resettest(big, power = 2:3, type = "regressor")
    out("reset_regressor", c(rs$statistic, rs$p.value, rs$parameter))
  }
  # White: the auxiliary regression on the levels, squares and cross products
  Z <- X; aux <- list(Z)
  for (i in seq_len(p)) aux[[length(aux) + 1]] <- Z[, i] * Z[, i:p, drop = FALSE]
  A <- do.call(cbind, aux)
  wh <- bptest(big, varformula = ~ A, data = data.frame(A = I(A)))
  out("white", c(wh$statistic, wh$p.value, wh$parameter))
  A0 <- cbind(Z, Z^2)
  wh <- bptest(big, varformula = ~ A0, data = data.frame(A0 = I(A0)))
  out("white_nocross", c(wh$statistic, wh$p.value, wh$parameter))
}
"""

R_GLM = r"""
y <- rv("y"); X <- rd("X"); Rm <- rd("R"); meta <- rv("meta"); sb <- rv("start_big"); ss <- rv("start_small")
fams <- c("gaussian", "binomial", "poisson", "Gamma")
links <- c("identity", "log", "logit", "probit", "cloglog", "sqrt", "inverse")
fam <- get(fams[meta[1]])(link = links[meta[2]]); nsmall <- meta[3]
p <- ncol(X)
dat <- data.frame(y = y)
nm <- paste0("x", seq_len(p))
for (j in seq_len(p)) dat[[nm[j]]] <- X[, j]
ctl <- glm.control(epsilon = 1e-14, maxit = 200)
# started at sd's solution: R's deviance test stops IRLS with the parameters near sqrt(epsilon)
big <- glm(as.formula(paste("y ~", paste(nm, collapse = " + "))), family = fam, data = dat, control = ctl, start = sb)
fsm <- as.formula(paste("y ~", paste(nm[seq_len(nsmall)], collapse = " + ")))
small <- glm(fsm, family = fam, data = dat, control = ctl, start = ss)
fixed <- fams[meta[1]] %in% c("binomial", "poisson")
a <- anova(big, test = if (fixed) "Chisq" else "F")
out("anova1_df", a$Df[-1]); out("anova1_dev", a$Deviance[-1]); out("anova1_p", a[-1, ncol(a)])
if (!fixed) out("anova1_F", a$F[-1])
a <- anova(small, big, test = if (fixed) "Chisq" else "F")
out("anova2_dev", a$Deviance[2]); out("anova2_p", a[2, ncol(a)]); if (!fixed) out("anova2_F", a$F[2])
d <- drop1(big, test = "Chisq"); out("drop1_lr", d$LRT[-1]); out("drop1_p", d[["Pr(>Chi)"]][-1])
A3 <- Anova(big, type = 3, test.statistic = "Wald")
out("anova3_chisq", A3$Chisq[-1]); out("anova3_p", A3[["Pr(>Chisq)"]][-1])
lh <- linearHypothesis(big, Rm); out("lh_chisq", lh$Chisq[2]); out("lh_p", lh[2, "Pr(>Chisq)"])
if (!fixed) { lh <- linearHypothesis(big, Rm, test = "F"); out("lh_F", lh$F[2]); out("lh_F_p", lh[2, "Pr(>F)"]) }
lr <- lrtest(small, big); out("lrtest", lr$Chisq[2]); out("lrtest_p", lr[2, "Pr(>Chisq)"])
out("hat", hatvalues(big))
out("cooks", cooks.distance(big))
out("dffits", dffits(big))
out("dfbetas", t(dfbetas(big)))
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
        if parts:
            vals[parts[0]] = [None if p == "NA" else float(p) for p in parts[1:]]
    return vals


def short(v):
    return np.round(np.asarray(v, dtype=float), 6)


def lm_problems():
    rng = np.random.default_rng(20261120)
    out = {}
    n = 40
    X = short(rng.standard_normal((n, 3)))
    y = short(1.0 + X @ [0.8, -0.4, 0.0] + rng.standard_normal(n) * (0.5 + np.abs(X[:, 0])))
    R = np.array([[0, 1, -1, 0], [0, 0, 1, 1], [0, 1, 1, 0.0]])
    out["ols"] = dict(y=y, X=X, w=np.ones(n), R=R, q=np.array([0.5, 0.0, 0.2]), g=np.zeros(n), meta=[1, 1, 0, 2])
    # the factor's treatment dummies (3 levels) after two continuous columns, n = 45 (odd)
    n = 45
    g = rng.integers(0, 3, n)
    Xf = np.column_stack([short(rng.standard_normal((n, 2))), (g == 1).astype(float), (g == 2).astype(float)])
    yf = short(0.5 + Xf @ [1.0, 0.3, 0.6, -0.2] + rng.standard_normal(n))
    Rf = np.array([[0, 0, 0, 1, -1.0], [0, 1, 0, 0, 0]])
    out["factor"] = dict(y=yf, X=Xf, w=np.ones(n), R=Rf, q=np.zeros(2), g=g.astype(float), meta=[1, 2, 1, 1])
    # weights (the GQ, RESET and White parts need unit weights in R: they ignore them)
    n = 50
    Xw = short(rng.standard_normal((n, 2)))
    ww = short(rng.uniform(0.3, 3.0, n))
    yw = short(2.0 + Xw @ [1.0, -1.0] + rng.standard_normal(n) / np.sqrt(ww))
    Rw = np.array([[0, 1, 1.0]])
    out["wls"] = dict(y=yw, X=Xw, w=ww, R=Rw, q=np.array([0.1]), g=np.zeros(n), meta=[1, 1, 0, 1])
    # no intercept
    n = 36
    Xn = short(rng.standard_normal((n, 3)) + 1.0)
    yn = short(Xn @ [0.5, 0.5, -0.3] + rng.standard_normal(n))
    Rn = np.array([[1, -1, 0.0], [0, 0, 1.0]])
    out["noint"] = dict(y=yn, X=Xn, w=np.ones(n), R=Rn, q=np.zeros(2), g=np.zeros(n), meta=[0, 2, 0, 3])
    return out


def glm_problems():
    rng = np.random.default_rng(7120)
    out = {}
    n = 120
    X = short(rng.standard_normal((n, 3)))
    eta = 0.2 + X @ [0.9, -0.5, 0.0]
    R = np.array([[0, 1, 1, 0], [0, 0, 0, 1.0]])
    out["binomial_logit"] = dict(
        y=(rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float), X=X, R=R, family="binomial", link="logit"
    )
    out["binomial_probit"] = dict(
        y=(rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float), X=X, R=R, family="binomial", link="probit"
    )
    out["poisson_log"] = dict(y=rng.poisson(np.exp(eta)).astype(float), X=X, R=R, family="poisson", link="log")
    out["gamma_log"] = dict(y=short(rng.gamma(3.0, np.exp(eta) / 3.0)), X=X, R=R, family="gamma", link="log")
    out["gaussian_identity"] = dict(y=short(eta + rng.standard_normal(n)), X=X, R=R, family="gaussian", link="identity")
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
            'cat(R.version.string, "| sandwich", format(packageVersion("sandwich")), '
            '"| lmtest", format(packageVersion("lmtest")), "| car", format(packageVersion("car")))',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = {"_versions": ver.strip(), "lm": {}, "glm": {}}
    for name, p in lm_problems().items():
        # the hypothesis matrix is stored as "hyp" ("R" holds R's values)
        data["lm"][name] = {("hyp" if k == "R" else k): tolist(v) for k, v in p.items()} | {"R": run_r(R_LM, p)}
    for name, p in glm_problems().items():
        meta = [FAMS.index(p["family"]) + 1, LINKS.index(p["link"]) + 1, 1]
        import stardust as sd

        sb = sd.regression.glm(p["y"], p["X"], family=p["family"], link=p["link"]).params.tolist()
        ss = sd.regression.glm(p["y"], p["X"][:, :1], family=p["family"], link=p["link"]).params.tolist()
        data["glm"][name] = {
            ("hyp" if k == "R" else k): (tolist(v) if isinstance(v, np.ndarray) else v) for k, v in p.items()
        } | {"R": run_r(R_GLM, dict(y=p["y"], X=p["X"], R=p["R"], meta=meta, start_big=sb, start_small=ss))}
    OUT.write_text(json.dumps(data))
    print(OUT, OUT.stat().st_size, "bytes;", ver)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
