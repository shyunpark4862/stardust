"""R's nls on fixed problems, for sd.regression.nls's tests (tests/regression/data/nls_r.json;
docs/design/regression.md §3.7). R is not a test dependency: run this once where Rscript is
installed,

    .venv/bin/python tools/gen_nls_r.py

Per problem (the inputs too): R's stats::nls with the model's analytic gradient (deriv())
and nls.control(tol = 1e-10, maxiter = 1000) (1e-8 for twod), Gauss-Newton or, with bounds,
algorithm = "port": coefficients, summary()'s standard errors, t values and p-values,
sigma, the residual degrees of freedom, the deviance (weighted sum of squares), fitted values,
residuals, logLik, AIC and BIC (logLik.nls counts sigma: df = p + 1). Problems: an
exponential decay, the same with weights (some zero), R's Puromycin (treated:
Michaelis-Menten), a model of two predictors (x is (n, 2)), a logistic curve, and the decay
with a lower bound that is active at the solution (port).
"""

import json
import pathlib
import subprocess
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "regression" / "data" / "nls_r.json"

# name -> (R formula's right side, parameter names, start, columns of x, R's extra arguments)
MODELS = {
    "decay": ("a*exp(-b*x) + c", ["a", "b", "c"], [1.0, 1.0, 1.0], ["x"], ""),
    "decay_weighted": ("a*exp(-b*x) + c", ["a", "b", "c"], [1.0, 1.0, 1.0], ["x"], "weights = w"),
    "puromycin": ("Vm*x/(K+x)", ["Vm", "K"], [200.0, 0.1], ["x"], ""),
    "twod": ("a*x1^b*exp(c*x2)", ["a", "b", "c"], [1.0, 1.0, 0.0], ["x1", "x2"], ""),
    "logistic": ("Asym/(1+exp((xmid-x)/scal))", ["Asym", "xmid", "scal"], [10.0, 5.0, 1.0], ["x"], ""),
    "decay_bounded": (
        "a*exp(-b*x) + c",
        ["a", "b", "c"],
        [1.0, 1.0, 1.0],
        ["x"],
        'algorithm = "port", lower = c(0, 0, 0.6), upper = c(10, 10, 10)',
    ),
}

PUROMYCIN_CONC = [0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10]
PUROMYCIN_RATE = [76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200]


def problems():
    rng = np.random.default_rng(2031)
    out = {}
    x = np.linspace(0.0, 5.0, 40)
    y = 3.0 * np.exp(-0.7 * x) + 0.5 + 0.05 * rng.standard_normal(40)
    out["decay"] = dict(x=x.tolist(), y=y.tolist(), w=None)
    w = rng.uniform(0.5, 2.0, 40)
    w[[3, 17, 30]] = 0.0
    out["decay_weighted"] = dict(x=x.tolist(), y=y.tolist(), w=w.tolist())
    out["puromycin"] = dict(x=PUROMYCIN_CONC, y=[float(v) for v in PUROMYCIN_RATE], w=None)
    x1, x2 = rng.uniform(1.0, 3.0, 60), rng.uniform(0.0, 1.0, 60)
    y = 2.0 * x1**0.7 * np.exp(-0.5 * x2) + 0.05 * rng.standard_normal(60)
    out["twod"] = dict(x=np.column_stack([x1, x2]).tolist(), y=y.tolist(), w=None)
    x = np.linspace(0.0, 10.0, 30)
    y = 12.0 / (1.0 + np.exp((4.5 - x) / 1.3)) + 0.3 * rng.standard_normal(30)
    out["logistic"] = dict(x=x.tolist(), y=y.tolist(), w=None)
    out["decay_bounded"] = dict(out["decay"])
    return out


def r_script(name, prob):
    rhs, pn, start, cols, extra = MODELS[name]
    fmt = lambda v: ", ".join(f"{t:.17g}" for t in v)
    lines = [f"y <- c({fmt(prob['y'])})"]
    if len(cols) == 1:
        lines.append(f"x <- c({fmt(prob['x'])})")
    else:
        for j, c in enumerate(cols):
            lines.append(f"{c} <- c({fmt([r[j] for r in prob['x']])})")
    if prob["w"] is not None:
        lines.append(f"w <- c({fmt(prob['w'])})")
    args = ", ".join(cols + pn)
    lines.append(f"fm <- deriv(~ {rhs}, c({', '.join(repr(v) for v in pn)}), function({args}) {{}})")
    st = ", ".join(f"{a} = {v:.17g}" for a, v in zip(pn, start))
    ex = (", " + extra) if extra else ""
    tol = "1e-8" if name == "twod" else "1e-10"  # twod: R stops on its step factor at 1e-10
    lines.append(
        f"f <- nls(y ~ fm({args}), start = list({st}), control = nls.control(tol = {tol}, maxiter = 1000){ex})"
    )
    lines.append("s <- summary(f); cf <- s$coefficients")
    lines.append('out <- function(nm, v) cat(nm, sprintf("%.17g", v), "\\n")')
    lines.append('out("coef", cf[, 1]); out("se", cf[, 2]); out("t", cf[, 3]); out("p", cf[, 4])')
    lines.append('out("sigma", s$sigma); out("df", s$df); out("deviance", deviance(f))')
    lines.append('out("fitted", fitted(f)); out("resid", y - fitted(f))')
    lines.append('out("loglik", as.numeric(logLik(f))); out("aic", AIC(f)); out("bic", BIC(f))')
    return "\n".join(lines) + "\n"


def run(name, prob):
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "run.R"
        path.write_text(r_script(name, prob))
        res = subprocess.run(["Rscript", "--vanilla", str(path)], capture_output=True, text=True, check=True)
    vals = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if parts:
            vals[parts[0]] = [float(v) for v in parts[1:]]
    rhs, pn, start, cols, extra = MODELS[name]
    return dict(prob, formula=rhs, names=pn, start=start, extra=extra, R=vals)


def main():
    ver = subprocess.run(
        ["Rscript", "--vanilla", "-e", "cat(R.version.string)"], capture_output=True, text=True, check=True
    ).stdout
    data = {"_versions": ver.strip()}
    for name, prob in problems().items():
        data[name] = run(name, prob)
    OUT.write_text(json.dumps(data))
    print(OUT, OUT.stat().st_size, "bytes;", ver)


if __name__ == "__main__":
    main()
