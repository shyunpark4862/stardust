"""G6: nls's accuracy on the NIST StRD nonlinear problems (docs/design/regression.md §3.7).
Results: bench/regression/g6/accuracy.md (the raw output in accuracy.log).

  - LRE (NIST's log relative error, capped at the 11 certified digits; the minimum over the
    parameters) of params and bse against the certified values, from both starting points,
    for sd.regression.nls (5-point differences, and the analytic Jacobian of
    tests/regression/nist_nls.py), sd without its Gauss-Newton steps after the iteration
    (the iteration's result, the covariance from the same accurate Jacobian), scipy's
    curve_fit (method 'lm' and 'trf', its defaults: 2-point differences), R's nls (its
    defaults: Gauss-Newton, forward differences) and minpack.lm's nlsLM (its defaults).
    'fail' is an error, a non-converged fit (scipy's RuntimeError, R's error) or a
    non-finite bse.
  - The relative errors (max over the parameters) of sd's params and bse against Arb
    references of the same double data (tools/regref.py's nls: Gauss-Newton in Arb at 320
    bits with the exact Jacobian; the certified values are of the decimal data, and they
    have 11 digits), from both starts, with differences and with the analytic Jacobian.
  - The accurate Jacobian's differences against the analytic Jacobian at the certified
    parameters: fixed steps 2^e |theta_j| and the per-parameter choice nls makes.

Run: uv run --no-sync python bench/regression/g6/accuracy.py
"""

import math
import pathlib
import subprocess
import sys
import tempfile
import warnings

import numpy as np

import stardust as sd
from stardust.regression import _nls

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests" / "regression"))
sys.path.insert(0, str(ROOT / "tools"))
import nist_nls as N  # noqa: E402
import regref  # noqa: E402

RSCRIPT = "/opt/homebrew/bin/Rscript"

R_FORMULAS = {
    "Misra1a": "y ~ b1*(1-exp(-b2*x))",
    "BoxBOD": "y ~ b1*(1-exp(-b2*x))",
    "Chwirut1": "y ~ exp(-b1*x)/(b2+b3*x)",
    "Chwirut2": "y ~ exp(-b1*x)/(b2+b3*x)",
    "Lanczos1": "y ~ b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "Lanczos2": "y ~ b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "Lanczos3": "y ~ b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)",
    "Gauss1": "y ~ b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2)",
    "Gauss2": "y ~ b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2)",
    "Gauss3": "y ~ b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2)",
    "DanWood": "y ~ b1*x^b2",
    "Misra1b": "y ~ b1*(1-(1+b2*x/2)^(-2))",
    "Kirby2": "y ~ (b1 + b2*x + b3*x^2)/(1 + b4*x + b5*x^2)",
    "Hahn1": "y ~ (b1+b2*x+b3*x^2+b4*x^3)/(1+b5*x+b6*x^2+b7*x^3)",
    "Thurber": "y ~ (b1+b2*x+b3*x^2+b4*x^3)/(1+b5*x+b6*x^2+b7*x^3)",
    "Nelson": "log(y) ~ b1 - b2*x1*exp(-b3*x2)",
    "MGH17": "y ~ b1 + b2*exp(-x*b4) + b3*exp(-x*b5)",
    "Misra1c": "y ~ b1*(1-(1+2*b2*x)^(-.5))",
    "Misra1d": "y ~ b1*b2*x*((1+b2*x)^(-1))",
    "Roszman1": "y ~ b1 - b2*x - atan(b3/(x-b4))/pi",
    "ENSO": "y ~ b1 + b2*cos(2*pi*x/12) + b3*sin(2*pi*x/12) + b5*cos(2*pi*x/b4) + b6*sin(2*pi*x/b4) + "
    "b8*cos(2*pi*x/b7) + b9*sin(2*pi*x/b7)",
    "MGH09": "y ~ b1*(x^2+x*b2)/(x^2+x*b3+b4)",
    "Rat42": "y ~ b1/(1+exp(b2-b3*x))",
    "MGH10": "y ~ b1*exp(b2/(x+b3))",
    "Eckerle4": "y ~ (b1/b2)*exp(-0.5*((x-b3)/b2)^2)",
    "Rat43": "y ~ b1/((1+exp(b2-b3*x))^(1/b4))",
    "Bennett5": "y ~ b1*(b2+x)^(-1/b3)",
}
METHODS = ["sd", "sd jac", "sd no polish", "scipy lm", "scipy trf", "R nls", "R nlsLM"]


def lre_pair(d, params, bse):
    if params is None or bse is None or not all(math.isfinite(v) for v in list(params) + list(bse)):
        return None
    return (min(N.lre(v, c) for v, c in zip(params, d["params"])), min(N.lre(v, c) for v, c in zip(bse, d["bse"])))


def run_sd(d, st, jac, polish=True):
    x, y = N.data(d, sd)
    f, J = N.model(d["name"], sd)
    old = _nls._POLISH_STEPS
    _nls._POLISH_STEPS = old if polish else 0
    try:
        r = sd.regression.nls(f, x, y, p0=[float(v) for v in d[st]], jac=J if jac else None)
    except ValueError:
        return None, None
    finally:
        _nls._POLISH_STEPS = old
    if not r.converged:
        return None, r.nfev
    return lre_pair(d, r.params.tolist(), r.bse.tolist()), r.nfev


def run_scipy(d, st, method):
    from scipy.optimize import curve_fit

    x, y = N.data(d, np)
    f, _ = N.model(d["name"], np)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, pcov, info, _, _ = curve_fit(f, x, y, p0=[float(v) for v in d[st]], method=method, full_output=True)
        except RuntimeError, ValueError, FloatingPointError, OverflowError:
            return None, None
    return lre_pair(d, popt, np.sqrt(np.diag(pcov))), info["nfev"]


def r_script():
    lines = ["suppressMessages(library(minpack.lm))", "options(digits = 17)"]
    for name in N.NAMES:
        d = N.load(name)
        if name == "Nelson":
            lines.append(
                f"y <- c({', '.join(d['y'])}); x1 <- c({', '.join(r[0] for r in d['x'])}); "
                f"x2 <- c({', '.join(r[1] for r in d['x'])}); dat <- data.frame(y = y, x1 = x1, x2 = x2)"
            )
        else:
            lines.append(
                f"y <- c({', '.join(d['y'])}); x <- c({', '.join(r[0] for r in d['x'])}); "
                "dat <- data.frame(y = y, x = x)"
            )
        for st in ("start1", "start2"):
            start = ", ".join(f"b{j + 1} = {v}" for j, v in enumerate(d[st]))
            for fn in ("nls", "nlsLM"):
                lines.append(
                    f"r <- tryCatch({fn}({R_FORMULAS[name]}, data = dat, start = list({start})), "
                    "error = function(e) NULL)\n"
                    f"if (is.null(r)) cat('RES {name} {st} {fn} fail\\n') else {{ s <- summary(r)$coefficients; "
                    f"cat('RES {name} {st} {fn} ok', paste(sprintf('%.17g', s[, 1]), collapse = ','), "
                    "paste(sprintf('%.17g', s[, 2]), collapse = ','), '\\n') }"
                )
    return "\n".join(lines) + "\n"


def run_r():
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(r_script())
        path = fh.name
    out = subprocess.run([RSCRIPT, path], capture_output=True, text=True, check=True).stdout
    res = {}
    for ln in out.splitlines():
        if not ln.startswith("RES "):
            continue
        parts = ln.split()
        name, st, fn, status = parts[1:5]
        d = N.load(name)
        key = (name, st, "R nls" if fn == "nls" else "R nlsLM")
        if status != "ok":
            res[key] = None
            continue
        res[key] = lre_pair(d, [float(v) for v in parts[5].split(",")], [float(v) for v in parts[6].split(",")])
    return res


def arb_table():
    rows = []
    for name in N.NAMES:
        d = N.load(name)
        xn, yn = N.data(d, np)
        fa, ja = N.model(name, regref.ARB)
        ref = regref.nls(fa, ja, xn.tolist(), yn.tolist(), [float(v) for v in d["params"]])
        x, y = N.data(d, sd)
        f, J = N.model(name, sd)
        cells = []
        for st in ("start1", "start2"):
            for jac in (None, J):
                try:
                    r = sd.regression.nls(f, x, y, p0=[float(v) for v in d[st]], jac=jac)
                except ValueError:
                    r = None
                if r is None or not r.converged or not all(math.isfinite(v) for v in r.bse.tolist()):
                    cells.append(None)
                    continue
                ep = max(abs(a - b) / abs(b) for a, b in zip(r.params.tolist(), ref["params"]))
                eb = max(abs(a - b) / abs(b) for a, b in zip(r.bse.tolist(), ref["bse"]))
                if ep > 1e-3:  # another local minimum
                    cells.append(None)
                    continue
                cells.append((ep, eb))
        rows.append((name, cells))
    return rows


def fd_table():
    """Max over the columns of max_i |J_fd - J| / max_i |J| at the certified parameters."""
    rows = []
    for name in N.NAMES:
        d = N.load(name)
        x, y = N.data(d, sd)
        f, jac = N.model(name, sd)
        th = [float(v) for v in d["params"]]
        Ja = np.asarray(jac(x, *th).tolist())
        M = _nls._Model("nls", f, None, x, y, None, None, len(th), [-math.inf] * len(th), [math.inf] * len(th))
        v0 = M.values(th)

        def err(Jd):
            Jd = np.asarray(Jd.tolist())
            return max(np.max(np.abs(Jd[:, j] - Ja[:, j])) / np.max(np.abs(Ja[:, j])) for j in range(len(th)))

        fixed = []
        for e in (-8, -10, -12, -14):
            cols = []
            for j, t in enumerate(th):
                h = math.ldexp(1.0, math.frexp(abs(t) if t else 1.0)[1] + e)
                cols.append(M._column(th, j, h, v0))
            fixed.append(err(sd.stack(cols, axis=1)))
        M.nfev = 0
        Jc = M.accurate_jac(th, v0)
        chosen = err(Jc)
        steps = [int(math.log2(h)) - math.frexp(abs(t) if t else 1.0)[1] for h, t in zip(M.steps, th)]
        rows.append((name, fixed, chosen, M.nfev, steps))
    return rows


def main():
    res = {}
    nfev = {}
    for name in N.NAMES:
        d = N.load(name)
        for st in ("start1", "start2"):
            res[(name, st, "sd")], nfev[(name, st, "sd")] = run_sd(d, st, False)
            res[(name, st, "sd jac")], nfev[(name, st, "sd jac")] = run_sd(d, st, True)
            res[(name, st, "sd no polish")], nfev[(name, st, "sd no polish")] = run_sd(d, st, False, polish=False)
            for m in ("lm", "trf"):
                res[(name, st, "scipy " + m)], nfev[(name, st, "scipy " + m)] = run_scipy(d, st, m)
    res.update(run_r())

    def cell(v):
        return "fail" if v is None else f"{v[0]:.1f} / {v[1]:.1f}"

    for st in ("start1", "start2"):
        print(f"## LRE params / bse, {st}")
        print("| problem | " + " | ".join(METHODS) + " |")
        print("|---|" + "---:|" * len(METHODS))
        for name in N.NAMES:
            print(f"| {name} | " + " | ".join(cell(res.get((name, st, m))) for m in METHODS) + " |")
        print()
        summ = []
        for m in METHODS:
            vals = [res.get((name, st, m)) for name in N.NAMES]
            ok = [v for v in vals if v is not None]
            summ.append(
                f"{m}: fail {len(vals) - len(ok)}, min params {min(v[0] for v in ok):.1f}, "
                f"median params {np.median([v[0] for v in ok]):.1f}, min bse {min(v[1] for v in ok):.1f}, "
                f"median bse {np.median([v[1] for v in ok]):.1f}"
            )
        print("\n".join(summ))
        print()
    print("## calls of f (nfev)")
    print(
        "| problem | "
        + " | ".join(f"{m} {s}" for s in ("1", "2") for m in ("sd", "sd no polish", "scipy lm", "scipy trf"))
        + " |"
    )
    print("|---|" + "---:|" * 8)
    for name in N.NAMES:
        cells = []
        for s in ("start1", "start2"):
            for m in ("sd", "sd no polish", "scipy lm", "scipy trf"):
                v = nfev.get((name, s, m))
                cells.append("" if v is None else str(v))
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
    print("## relative error against Arb (the same double data): params / bse")
    print("| problem | differences, start1 | jac, start1 | differences, start2 | jac, start2 |")
    print("|---|---:|---:|---:|---:|")
    allc = {k: [] for k in range(4)}
    for name, cells in arb_table():
        print(f"| {name} | " + " | ".join("fail" if c is None else f"{c[0]:.1e} / {c[1]:.1e}" for c in cells) + " |")
        for k, c in enumerate(cells):
            if c is not None:
                allc[k].append(c)
    for k, lab in enumerate(("differences start1", "jac start1", "differences start2", "jac start2")):
        v = allc[k]
        print(
            f"{lab}: max params {max(c[0] for c in v):.1e}, median {np.median([c[0] for c in v]):.1e}; "
            f"max bse {max(c[1] for c in v):.1e}, median {np.median([c[1] for c in v]):.1e}"
        )
    print()
    print("## accurate differences vs the analytic Jacobian (certified parameters)")
    print("| problem | 2^-8 | 2^-10 | 2^-12 | 2^-14 | chosen | calls | chosen exponents |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, fixed, chosen, calls, steps in fd_table():
        print(f"| {name} | " + " | ".join(f"{v:.1e}" for v in fixed) + f" | {chosen:.1e} | {calls} | {steps} |")


if __name__ == "__main__":
    main()
