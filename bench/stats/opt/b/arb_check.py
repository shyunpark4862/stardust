"""Adjudicate a sweep's dumped points against Arb (python-flint).

The sweeps (density_sweep.cpp, cdf_sweep.cpp, ...) write, for every method, the points where
a kernel and the scalar (double-double) definition differ by more than 1e-15 in contract
units: `method,x,p0,p1,kernel,scalar`. This computes the true value of each in ball arithmetic
and prints, per method, the largest error of the kernel and of the scalar definition in the
method's contract units (relative; for logpdf |d| / max(1, |value|)).

    .venv/bin/python bench/stats/opt/b/arb_check.py dump.csv
"""

import sys
from collections import defaultdict

from flint import arb, ctx

ctx.prec = 400


def A(v):
    return arb(float(v))  # the double exactly


def lgam(z):
    return z.lgamma()


def lbeta(a, b):
    return lgam(a) + lgam(b) - lgam(a + b)


def gamma_logpdf(x, a, s):
    y = x / s
    return (a - 1) * y.log() - y - lgam(a) - s.log()


def invgamma_logpdf(x, a, s):
    y = s / x
    return (a + 1) * y.log() - y - lgam(a) - s.log()


def beta_logpdf(x, a, b):
    return (a - 1) * x.log() + (b - 1) * (1 - x).log() - lbeta(a, b)


def t_logpdf(t, df):
    a = df / 2
    return -(a + arb(0.5)) * (1 + t * t / df).log() - df.log() / 2 - lbeta(a, arb(0.5))


def f_logpdf(x, d1, d2):
    a, b = d1 / 2, d2 / 2
    return a * (d1 / d2).log() + (a - 1) * x.log() - (a + b) * (1 + d1 * x / d2).log() - lbeta(a, b)


def P(a, y):
    return y.gamma_lower(a, regularized=1)


def Q(a, y):
    return y.gamma_upper(a, regularized=1)


def I(a, b, x):
    return x.beta_lower(a, b, regularized=1)


LOGPDF = {
    "gamma": lambda x, p0, p1: gamma_logpdf(x, p0, p1),
    "chi2": lambda x, p0, p1: gamma_logpdf(x, p0 / 2, arb(2)),
    "invgamma": lambda x, p0, p1: invgamma_logpdf(x, p0, p1),
    "beta": lambda x, p0, p1: beta_logpdf(x, p0, p1),
    "t": lambda x, p0, p1: t_logpdf(x, p0),
    "f": lambda x, p0, p1: f_logpdf(x, p0, p1),
}


def t_cdf(t, df):
    x = df / (df + t * t)
    half = I(df / 2, arb(0.5), x) / 2
    return half if t < 0 else 1 - half


def f_cdf(x, d1, d2):
    return I(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2))


CDF = {
    "gamma": lambda x, p0, p1: (P(p0, x / p1), Q(p0, x / p1)),
    "chi2": lambda x, p0, p1: (P(p0 / 2, x / 2), Q(p0 / 2, x / 2)),
    "invgamma": lambda x, p0, p1: (Q(p0, p1 / x), P(p0, p1 / x)),
    "beta": lambda x, p0, p1: (I(p0, p1, x), I(p1, p0, 1 - x)),
    "t": lambda x, p0, p1: (t_cdf(x, p0), t_cdf(-x, p0)),
    "f": lambda x, p0, p1: (f_cdf(x, p0, p1), 1 - f_cdf(x, p0, p1)),
}


def truth(method, x, p0, p1):
    dist, m = method.split("_")
    x, p0, p1 = A(x), A(p0), A(p1)
    if m == "logpdf":
        return LOGPDF[dist](x, p0, p1), True
    if m == "pdf":
        return LOGPDF[dist](x, p0, p1).exp(), False
    lo, up = CDF[dist](x, p0, p1)
    if m == "cdf":
        return lo, False
    if m == "ccdf":
        return up, False
    # the log of a tail from the smaller side (the larger one's distance from 1 is the smaller)
    if m == "logcdf":
        return (lo.log() if lo < 0.5 else (-up).log1p()), False
    if m == "logccdf":
        return (up.log() if up < 0.5 else (-lo).log1p()), False
    raise ValueError(method)


def err(got, true, mixed):
    d = abs(arb(got) - true)
    scale = max(1.0, abs(float(true.mid()))) if mixed else abs(true)
    e = d / scale
    return float(e.upper()) if e.is_finite() else float("nan")


def main():
    worst = defaultdict(lambda: [0.0, 0.0, 0, None])
    for line in open(sys.argv[1]):
        method, x, p0, p1, kern, scal = line.strip().split(",")
        try:
            tv, mixed = truth(method, x, p0, p1)
        except Exception:
            continue
        if not tv.is_finite() or float(tv.rad()) > 1e-20 * abs(float(tv.mid())):
            continue  # (the ball is too wide to judge: a value near the bottom of the range)
        w = worst[method]
        ek, es = err(float(kern), tv, mixed), err(float(scal), tv, mixed)
        w[2] += 1
        if ek > w[0]:
            w[0], w[3] = ek, (x, p0, p1)
        w[1] = max(w[1], es)
    print("| method | points checked | kernel max | scalar max | kernel's worst point (x, p0, p1) |")
    print("|---|---:|---:|---:|---|")
    for method, (ek, es, n, pt) in worst.items():
        print(f"| `{method}` | {n} | {ek:.3g} | {es:.3g} | {pt} |")


if __name__ == "__main__":
    main()
