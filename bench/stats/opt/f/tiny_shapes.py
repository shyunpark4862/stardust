"""Beta and F tails at tiny and small shapes, judged by Arb (python-flint).

Shapes a, b (F: dfn = 2a, dfd = 2b) in [1e-5, 6], most of them with one shape tiny, and x
over the whole range with extra weight where x^a (or (1 - x)^b) is between 0.3 and 1: there
I_x(a, b) is close to 1 while x is far from 1, the place where 1 - I loses digits if it is
formed from I. For each of

    Beta.cdf/ccdf/logcdf/logccdf, F.cdf/ccdf/logcdf/logccdf,
    sd.special.betainc/betaincc/fdtr/fdtrc

prints the largest relative error (the contract's unit for these methods) and its point.
True values: Arb balls at a precision raised until the radius is below 1e-25 of the value.

    .venv/bin/python bench/stats/opt/f/tiny_shapes.py [points per family = 100000] [processes = 8] [seed]
"""

import math
import sys
from multiprocessing import Pool

import numpy as np

TINY = 2.0 ** -1022
METHODS = ["beta_cdf", "beta_ccdf", "beta_logcdf", "beta_logccdf", "f_cdf", "f_ccdf", "f_logcdf", "f_logccdf",
           "betainc", "betaincc", "fdtr", "fdtrc"]


def points(n, seed):
    rng = np.random.default_rng(seed)

    def logu(lo, hi, size):
        return np.exp(rng.uniform(math.log(lo), math.log(hi), size))

    i = np.arange(n)
    a = np.where(i % 2 == 0, logu(1e-5, 1e-2, n), logu(1e-5, 6.0, n))
    b = np.where((i // 2) % 2 == 0, rng.uniform(0.0, 6.0, n), logu(1e-5, 6.0, n))
    swap = (i // 4) % 2 == 1
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    k = i % 6
    with np.errstate(all="ignore"):
        xb = np.select(
            [k == 0, k == 1, k == 2, k == 3, k == 4],
            [logu(1e-300, 1.0, n), 1.0 - logu(1e-17, 1.0, n), rng.uniform(0.0, 1.0, n),
             np.exp(np.log(rng.uniform(0.3, 1.0, n)) / a), 1.0 - np.exp(np.log(rng.uniform(0.3, 1.0, n)) / b)],
            logu(1e-12, 1.0, n))
        # F: the same arguments mapped back (x = (b / a) w / (1 - w)), and x over the whole range
        xf = np.select(
            [k == 0, k == 1, k == 3, k == 4],
            [logu(1e-300, 1e300, n), logu(1e-12, 1e12, n), (b / a) * np.exp(np.log(rng.uniform(0.3, 1.0, n)) / a),
             (b / a) * np.exp(-np.log(rng.uniform(0.3, 1.0, n)) / b)],
            (b / a) * xb / (1.0 - xb))
    okb = (xb > 0.0) & (xb < 1.0)
    okf = (xf > 0.0) & np.isfinite(xf)
    return (a[okb], b[okb], xb[okb]), (a[okf], b[okf], xf[okf])


def evaluate(beta_pts, f_pts):
    import stardust as sd

    S = sd.stats
    A = sd.asarray
    a, b, x = (A(v) for v in beta_pts)
    X = S.Beta(a=a, b=b)
    vb = [np.asarray(v) for v in (X.cdf(x), X.ccdf(x), X.logcdf(x), X.logccdf(x), sd.special.betainc(a, b, x),
                                  sd.special.betaincc(a, b, x))]
    d1, d2, xf = A(2.0 * f_pts[0]), A(2.0 * f_pts[1]), A(f_pts[2])
    Y = S.F(dfn=d1, dfd=d2)
    vf = [np.asarray(v) for v in (Y.cdf(xf), Y.ccdf(xf), Y.logcdf(xf), Y.logccdf(xf), sd.special.fdtr(d1, d2, xf),
                                  sd.special.fdtrc(d1, d2, xf))]
    return vb, vf


def _pair(a, b, x, is_f):
    """(I, 1 - I, log I, log(1 - I)) as Arb balls settled to 1e-25, or None."""
    from flint import arb, ctx

    for prec in (300, 600, 1500, 4000):
        ctx.prec = prec
        A_, B_, X_ = arb(a), arb(b), arb(x)
        if is_f:
            s = 2 * A_ * X_ + 2 * B_
            w, v = 2 * A_ * X_ / s, 2 * B_ / s
        else:
            w, v = X_, 1 - X_
        # the series side only (the argument at most 1/2: Arb's beta_lower is slow and wide next to
        # 1), the other side as 1 minus it at this precision
        if w < 0.75:
            lo = w.beta_lower(A_, B_, regularized=1)
            up = 1 - lo
        else:
            up = v.beta_lower(B_, A_, regularized=1)
            lo = 1 - up
        if not (lo.is_finite() and up.is_finite() and lo > 0 and up > 0):
            continue
        llo = lo.log() if lo < 0.5 else (-up).log1p()
        lup = up.log() if up < 0.5 else (-lo).log1p()
        vals = (lo, up, llo, lup)
        if all(t.is_finite() and float(t.rad()) <= 1e-25 * abs(float(t.mid())) for t in vals):
            return vals
        if all(t.is_finite() and t.rad() < abs(t) * 1e-25 for t in vals):  # (values below the doubles)
            return vals
    return None


def _errors(args):
    """Relative errors of (cdf, ccdf, logcdf, logccdf, special lower, special upper); NaN = not judged."""
    from flint import arb

    is_f, rows = args
    out = np.full((len(rows), 6), np.nan)
    for r, (a, b, x, *got) in enumerate(rows):
        tv = _pair(a, b, x, is_f)
        if tv is None:
            continue
        for j, t in enumerate((tv[0], tv[1], tv[2], tv[3], tv[0], tv[1])):
            g = got[j]
            if abs(t) < TINY:  # a subnormal or underflowing value: absolute, in units of the smallest normal
                e = abs(arb(g) - t) / TINY
            else:
                e = abs(arb(g) - t) / abs(t)
            out[r, j] = float(e.upper()) if e.is_finite() else math.inf
    return out


def judge(pts, vals, is_f, pool):
    rows = np.column_stack([*pts, *vals]).tolist()
    step = 500
    chunks = [(is_f, rows[i:i + step]) for i in range(0, len(rows), step)]
    return np.vstack(pool.map(_errors, chunks, chunksize=1))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261117
    beta_pts, f_pts = points(n, seed)
    vb, vf = evaluate(beta_pts, f_pts)
    with Pool(procs, maxtasksperchild=20) as pool:
        eb = judge(beta_pts, vb, False, pool)
        ef = judge(f_pts, vf, True, pool)
    print("| method | points judged | max error | max error, x normal | above 5e-15 | worst point (x, a or dfn, b or dfd) |")
    print("|---|---:|---:|---:|---:|---|")
    cols = {"beta_cdf": (eb, 0, beta_pts, 1), "beta_ccdf": (eb, 1, beta_pts, 1), "beta_logcdf": (eb, 2, beta_pts, 1),
            "beta_logccdf": (eb, 3, beta_pts, 1), "f_cdf": (ef, 0, f_pts, 2), "f_ccdf": (ef, 1, f_pts, 2),
            "f_logcdf": (ef, 2, f_pts, 2), "f_logccdf": (ef, 3, f_pts, 2), "betainc": (eb, 4, beta_pts, 1),
            "betaincc": (eb, 5, beta_pts, 1), "fdtr": (ef, 4, f_pts, 2), "fdtrc": (ef, 5, f_pts, 2)}
    for m in METHODS:
        e, j, pts, scale = cols[m]
        col = e[:, j]
        ok = ~np.isnan(col)
        w = int(np.nanargmax(col))
        normal = np.nanmax(np.where(pts[2] >= TINY, col, 0.0))
        print(f"| `{m}` | {int(ok.sum())} | {col[w]:.3g} | {normal:.3g} | {int((col[ok] > 5e-15).sum())} | "
              f"({float(pts[2][w])!r}, {float(scale * pts[0][w])!r}, {float(scale * pts[1][w])!r}) |", flush=True)


if __name__ == "__main__":
    main()
