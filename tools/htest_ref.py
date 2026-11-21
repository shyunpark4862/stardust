"""Reference values for sd.htest's statistics, p-values and intervals, from Arb (python-flint) and
exact rationals (Python's int and fractions; docs/design/htest.md §4, §7).

    .venv/bin/python tools/htest_ref.py                  # every family, on 10 processes
    .venv/bin/python tools/htest_ref.py ttest ks_1samp   # only these families
    .venv/bin/python tools/htest_ref.py --jobs 4 ttest   # 4 processes instead of 10
    .venv/bin/python tools/htest_ref.py --list           # the families and their problems

writes tests/htest/ref/<family>.json ("_provenance": the Arb, the target, the tool, the problems
asked for and the ones left out, as every reference file; bench/README.md §2.2): for each problem
of tests/htest/problems.py that the family covers, the inputs' digest (problems.digest), the
method the reference follows ('exact', 'asymptotic', ...: scipy's choice for the problem, H12)
and each reference field as [hi, lo], a
double-double (hi the value rounded to a double, lo the rest rounded). Integer fields (counts) are
ints and text fields strings. NaN marks a degenerate input (H11). Confidence intervals
("conf_int", alpha = 0.05) are a pair of such values; "kappa" is a plain float, the p-value's
condition number (approximate: a tolerance factor of the contract, docs/design/htest.md §4).

The inputs are the problem's doubles taken as exact rationals. A statistic that is a rational
function of the data (means, sums of squares, ranks, counts, t^2, r^2, F, U, T+, Kendall's
discordant pairs, KS lattice gaps, chi^2) is computed exactly with Fractions and rounded once. The
p-values come from

  - exact null distributions counted in integers: Mann-Whitney (Gaussian binomial coefficients),
    the Wilcoxon signed rank (subset sums), Kendall's inversions (Mahonian numbers), the
    two-sample Kolmogorov-Smirnov lattice paths, Fisher's exact test (2 x 2 and r x c tables
    enumerated with their margins), the binomial with p the double's rational value;
  - Arb balls: the regularized incomplete beta (t and F), the regularized incomplete gamma
    (chi^2), erfc (normal), the one-sample Kolmogorov-Smirnov distribution by the
    Marsaglia-Tsang-Wang matrix power and the Birnbaum-Tingey sum, Kolmogorov's series with a
    bound on its tail.

Every Arb value is settled to 2^-110 relative by tools/arbref.py's rigorous() (the precision
doubles until the ball is that narrow; no "two precisions agree" heuristic). Two-sided rules with
a tolerance use the tolerance's double as an exact rational: 1 + 1e-7 for the binomial (scipy and
R) and Fisher's r x c test (R), 1 + 1e-14 for scipy's 2 x 2 Fisher test. Each problem has a time
limit (SIGALRM); a point that does not settle is left out and reported. Workers are replaced
every 50 problems (maxtasksperchild) so a long run's memory stays flat. No mpmath (the project's
rule): Arb and Fractions only.

A family is a function of a problem giving {field: value}, where a value is a Fraction (exact),
a zero-argument function giving an arb at the current precision (settled by rigorous()), a float
(nan, inf, an exact double), an int, a str, or a list of these. Later milestones add families
the same way (FAMILIES).
"""

import bisect
import json
import math
import multiprocessing
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import arb, arb_mat, fmpq
from flint import ctx as flint_ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import arbref as ar  # noqa: E402
from arbref import NotSettled, rigorous, to_dd  # noqa: E402

OUT = ROOT / "tests" / "htest" / "ref"
JOBS = 10  # at most 10 processes (user decision 2026-11-16)
RERR = Fraction(1 + 1e-7)  # the relative tolerance of binomtest's (scipy, R) and fisher.test's (R) two-sided rules
NAN = math.nan


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


# ---------------------------------------------------------------- exact helpers


def fr(v):
    """A double (or int) as its exact rational."""
    return Fraction(v) if isinstance(v, int) else Fraction(float(v))


def fvec(a):
    return [fr(v) for v in np.ravel(np.asarray(a, dtype=np.float64))]


def mean_var(xs):
    """The exact mean and sample variance (n - 1) of rationals."""
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else Fraction(0)
    return m, v


def midranks(values):
    """The midranks (Fractions, 1-based) of values in their order, and the tie group sizes."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [Fraction(0)] * len(values)
    ties = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = Fraction(i + j + 2, 2)
        ties.append(j - i + 1)
        i = j + 1
    return r, ties


def A(v):
    """An exact rational (or int, or double) as an arb at the current precision."""
    if isinstance(v, arb):
        return v
    v = Fraction(v)
    return arb(fmpq(v.numerator, v.denominator))


def sign(v):
    return (v > 0) - (v < 0)


# ---------------------------------------------------------------- distributions (Arb)


def beta_reg(x, a, b):
    """I_x(a, b), x a Fraction in [0, 1] or an arb."""
    if not isinstance(x, arb):
        if x <= 0:
            return arb(0)
        if x >= 1:
            return arb(1)
    return A(x).beta_lower(A(a), A(b), 1)


def gamma_q(s, x):
    """Q(s, x) = Gamma(s, x) / Gamma(s), the regularized upper incomplete gamma."""
    if not isinstance(x, arb) and x <= 0:
        return arb(1)
    return A(x).gamma_upper(A(s), 1)


def gamma_p(s, x):
    if not isinstance(x, arb) and x <= 0:
        return arb(0)
    return A(x).gamma_lower(A(s), 1)


def norm_sf(z):
    return (A(z) / arb(2).sqrt()).erfc() / 2


def norm_isf(p):
    """The normal's inverse survival function at p (a Fraction or arb in (0, 1))."""
    return arb(2).sqrt() * (2 * A(p)).erfcinv()


def norm_p(z, alt):
    """scipy's _get_pvalue for the normal (symmetric): z an arb."""
    if alt == "greater":
        return norm_sf(z)
    if alt == "less":
        return norm_sf(-z)
    return 2 * norm_sf(abs(z))


def t_p(t2, s, nu, alt):
    """The Student t p-value of t = s sqrt(t2) (t2 a Fraction, s its sign) with nu degrees of
    freedom (a Fraction): through I_{nu/(nu + t^2)}(nu/2, 1/2), the two tails' total."""
    x = nu / (nu + t2)
    two = beta_reg(x, nu / 2, Fraction(1, 2))
    if alt == "two-sided":
        return two
    upper = s > 0 if alt == "greater" else s < 0
    if s == 0:
        return arb(1) / 2
    return two / 2 if upper else 1 - two / 2


def f_sf(F, dfn, dfd):
    return beta_reg(dfd / (dfd + dfn * F), dfd / 2, dfn / 2)


def chi2_sf(stat, df):
    return gamma_q(Fraction(df) / 2, stat / 2)


def amax(vals):
    """The ball holding the maximum of balls."""
    lo = max(v.lower() for v in vals)
    hi = max(v.upper() for v in vals)
    return (lo + hi) / 2 + arb(0, (hi - lo).upper() / 2)


# ---------------------------------------------------------------- families

FAMILIES = {}


class Family:
    def __init__(self, name, funcs, compute, select=None, seconds=60):
        self.name, self.funcs, self.compute, self.select, self.seconds = name, funcs, compute, select, seconds

    def problems(self):
        from tests.htest import problems as P

        out = []
        for p in P.PROBLEMS.values():
            if p.func not in self.funcs or "batch" in p.tags or "axis" in p.kwargs:
                continue
            if self.select is None or self.select(p):
                out.append(p)
        return out


def family(name, funcs, select=None, seconds=60):
    def deco(fn):
        FAMILIES[name] = Family(name, funcs, fn, select, seconds)
        return fn

    return deco


# ---- H1: means and variances

KAPPA_BITS = 400  # the working precision of the condition numbers (central differences)
ALPHA = Fraction(0.05)  # the confidence intervals' level, alpha = 0.05 (its double)


def kappa(pfun, s, dfs=(), vary=()):
    """The p-value's condition number, approximately: |s dp/ds| / p, plus |d dp/dd| / p for each
    degree of freedom d at the indices `vary` (one computed in floating point, Welch's), by
    central differences in Arb at KAPPA_BITS. pfun(s, *dfs) gives the p-value as an arb; s is a
    value or a zero-argument function giving an arb. A tolerance factor, not a reference: NaN
    where the p-value is NaN or 0."""
    with ar.precision(KAPPA_BITS):
        s = s() if callable(s) else A(s)
        dfs = [A(d) for d in dfs]
        p = pfun(s, *dfs)
        if not p.is_finite() or not p > 0:
            return NAN
        h = arb(2) ** -120
        tot = arb(0)
        if not s.is_zero():
            tot += abs(pfun(s * (1 + h), *dfs) - pfun(s * (1 - h), *dfs))
        for i in vary:
            up = list(dfs)
            dn = list(dfs)
            up[i], dn[i] = dfs[i] * (1 + h), dfs[i] * (1 - h)
            tot += abs(pfun(s, *up) - pfun(s, *dn))
        return float((tot / (2 * h * p)).mid())


def t_p_arb(t, nu, alt):
    """The Student t p-value at an arb t with nu degrees of freedom (an arb)."""
    two = (nu / (nu + t * t)).beta_lower(nu / 2, arb(1) / 2, 1)
    if alt == "two-sided":
        return two
    upper = t > 0 if alt == "greater" else t < 0
    return two / 2 if upper else 1 - two / 2


def t_upper(x, nu):
    """P(T > x) for x >= 0."""
    if x.is_zero():
        return arb(1) / 2
    return (nu / (nu + x * x)).beta_lower(nu / 2, arb(1) / 2, 1) / 2


def t_pdf(x, nu):
    return (
        ((nu + 1) / 2).lgamma().exp()
        / ((nu * arb.pi()).sqrt() * (nu / 2).lgamma().exp())
        * (1 + x * x / nu) ** (-(nu + 1) / 2)
    )


def t_quantile_upper(q, nu):
    """The x > 0 with P(T > x) = q (q a Fraction below 1/2, nu a Fraction), certified to 2^-140
    relative and widened by that (an arb ball)."""
    from scipy import stats as st

    guess = float(st.t.isf(float(q), float(nu)))
    g = lambda x: A(q) - t_upper(x, A(nu))  # increasing in x
    dg = lambda x: t_pdf(x, A(nu))
    x = ar.solve_settled(g, dg, arb(guess), increasing=True, rel_bits=140)
    with ar.precision(ar.MAX_PREC):  # (the sum below at the default 53 bits would round the point)
        return ar.add_error(x, abs(x) * arb(2) ** -139)


def _t_ci(est, se2, nu, alt):
    """scipy's interval est -+ q se at ALPHA (the problem's alternative): two values, each a
    Fraction, an inf or a function of the precision."""
    if se2 == 0:
        return [est, est]
    if alt == "two-sided":
        q = t_quantile_upper(ALPHA / 2, nu)
        return [lambda: A(est) - q * A(se2).sqrt(), lambda: A(est) + q * A(se2).sqrt()]
    q = t_quantile_upper(ALPHA, nu)
    if alt == "less":
        return [-math.inf, lambda: A(est) + q * A(se2).sqrt()]
    return [lambda: A(est) - q * A(se2).sqrt(), math.inf]


def _t_result(est, diff, se2, nu, alt, df_varies=False):
    """The t test's fields: est the interval's centre, diff = est - (the null value), se2 the
    squared standard error, nu the degrees of freedom (a Fraction, or NaN: everything NaN).
    A zero standard error gives NaN statistic and p-value (user decision 3) with a zero-width
    interval."""
    if isinstance(nu, float):
        return {"statistic": NAN, "pvalue": NAN, "df": NAN, "conf_int": [NAN, NAN], "kappa": NAN}
    if se2 == 0 or nu <= 0:
        ci = [est, est] if se2 == 0 and nu > 0 else [NAN, NAN]
        return {"statistic": NAN, "pvalue": NAN, "df": nu, "conf_int": ci, "kappa": NAN}
    t2 = diff * diff / se2
    s = sign(diff)
    stat = (lambda: s * A(t2).sqrt()) if s else Fraction(0)
    return {
        "statistic": stat,
        "pvalue": lambda: t_p(t2, s, nu, alt),
        "df": nu,
        "conf_int": _t_ci(est, se2, nu, alt),
        "kappa": kappa(lambda t, v: t_p_arb(t, v, alt), stat, (nu,), (0,) if df_varies else ()),
    }


def trimmed(xs, trim):
    """Yuen's pieces (scipy's _ttest_trim_var_mean_len): the winsorized variance, the trimmed
    mean and h = n - 2g, g = int(n * trim) as the double product. The variance is NaN for
    h < 2 (a float)."""
    s = sorted(xs)
    n = len(s)
    g = int(float(n) * float(trim))
    h = n - 2 * g
    mid = s[g : n - g]
    tm = sum(mid) / h if h > 0 else NAN
    win = [s[g]] * g + mid + [s[n - g - 1]] * g
    wm = sum(win) / n
    v = sum((w - wm) ** 2 for w in win) / (h - 1) if h > 1 else NAN
    return v, tm, h


def _two_sample(m1, v1, n1, m2, v2, n2, equal_var, alt):
    """Student's (scipy's convention: a sample of one has variance 0) or Welch's t from the
    pieces (Fractions; a variance NaN for a sample too small)."""
    est = m1 - m2 if not (isinstance(m1, float) or isinstance(m2, float)) else NAN
    if n1 < 1 or n2 < 1 or isinstance(est, float):
        return _t_result(NAN, NAN, NAN, NAN, alt)
    if equal_var:
        v1 = Fraction(0) if n1 == 1 else v1
        v2 = Fraction(0) if n2 == 1 else v2
        nu = Fraction(n1 + n2 - 2)
        if nu <= 0:
            return {"statistic": NAN, "pvalue": NAN, "df": nu, "conf_int": [NAN, NAN], "kappa": NAN}
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / nu
        se2 = sp2 * (Fraction(1, 1) / n1 + Fraction(1, 1) / n2)
        return _t_result(est, est, se2, nu, alt)
    if n1 < 2 or n2 < 2 or isinstance(v1, float) or isinstance(v2, float):
        return _t_result(NAN, NAN, NAN, NAN, alt)
    a1, a2 = v1 / n1, v2 / n2
    se2 = a1 + a2
    if se2 == 0:
        return {"statistic": NAN, "pvalue": NAN, "df": NAN, "conf_int": [est, est], "kappa": NAN}
    nu = se2 * se2 / (a1 * a1 / (n1 - 1) + a2 * a2 / (n2 - 1))
    return _t_result(est, est, se2, nu, alt, df_varies=True)


@family("ttest", ("ttest_1samp", "ttest_rel", "ttest_ind", "ttest_ind_from_stats"))
def ttest(p):
    """Student's, Welch's and Yuen's t: exact means (trimmed means), variances (winsorized) and
    t^2, the p-value from the t distribution's regularized incomplete beta, the 95% interval
    from a certified t quantile."""
    f, a, kw = p.func, p.args, p.kwargs
    alt = kw.get("alternative", "two-sided")
    if f in ("ttest_1samp", "ttest_rel"):
        if f == "ttest_1samp":
            x, mu = fvec(a[0]), fr(a[1])
        else:
            x, mu = [u - v for u, v in zip(fvec(a[0]), fvec(a[1]))], Fraction(0)
        n = len(x)
        if n == 0:
            return _t_result(NAN, NAN, NAN, NAN, alt)
        if n < 2:
            return {"statistic": NAN, "pvalue": NAN, "df": Fraction(n - 1), "conf_int": [NAN, NAN], "kappa": NAN}
        m, v = mean_var(x)
        return _t_result(m, m - mu, v / n, Fraction(n - 1), alt)
    equal_var = kw.get("equal_var", True)
    if f == "ttest_ind_from_stats":
        m1, s1, n1, m2, s2, n2 = (fr(v) for v in a)
        return _two_sample(m1, s1 * s1, n1, m2, s2 * s2, n2, equal_var, alt)
    x, y = fvec(a[0]), fvec(a[1])
    trim = kw.get("trim", 0)
    if trim:
        (v1, m1, n1), (v2, m2, n2) = trimmed(x, trim), trimmed(y, trim)
    else:
        n1, n2 = len(x), len(y)
        m1, v1 = mean_var(x) if n1 else (NAN, NAN)
        m2, v2 = mean_var(y) if n2 else (NAN, NAN)
        v1 = v1 if n1 > 1 else NAN
        v2 = v2 if n2 > 1 else NAN
    return _two_sample(m1, v1, Fraction(n1), m2, v2, Fraction(n2), equal_var, alt)


def f_p_arb(F, dfn, dfd):
    return (dfd / (dfd + dfn * F)).beta_lower(dfd / 2, dfn / 2, 1)


def chi2_p_arb(s, df):
    return (s / 2).gamma_upper(df / 2, 1)


def _f_fields(F, dfn, dfd, df_varies=False):
    return {
        "statistic": F,
        "pvalue": lambda: f_sf(F, dfn, dfd),
        "df_num": dfn,
        "df_denom": dfd,
        "kappa": kappa(f_p_arb, F, (dfn, dfd), (1,) if df_varies else ()),
    }


def _nan_fields(**known):
    return {"statistic": NAN, "pvalue": NAN, **known, "kappa": NAN}


def _oneway(gs):
    """The equal-variance one-way ANOVA's fields on groups of Fractions."""
    k = len(gs)
    ns = [len(g) for g in gs]
    N = sum(ns)
    dfn, dfd = Fraction(k - 1), Fraction(N - k)
    if min(ns) == 0 or dfd <= 0:
        return _nan_fields(df_num=dfn, df_denom=dfd)
    means = [sum(g) / len(g) for g in gs]
    grand = sum(sum(g) for g in gs) / N
    ssb = sum(n * (m - grand) ** 2 for n, m in zip(ns, means))
    ssw = sum(sum((x - m) ** 2 for x in g) for g, m in zip(gs, means))
    if ssw == 0:
        return _nan_fields(df_num=dfn, df_denom=dfd)
    return _f_fields((ssb / dfn) / (ssw / dfd), dfn, dfd)


@family("f_oneway", ("f_oneway",))
def f_oneway(p):
    """One-way ANOVA and Welch's (scipy's formula): F and its degrees of freedom exact, the
    p-value from the F distribution's incomplete beta (df_num, df_denom: H19, decision 6)."""
    gs = [fvec(g) for g in p.args]
    if p.kwargs.get("equal_var", True):
        return _oneway(gs)
    k = len(gs)
    dfn = Fraction(k - 1)
    if min(len(g) for g in gs) < 2:
        return _nan_fields(df_num=dfn, df_denom=NAN)
    ns = [Fraction(len(g)) for g in gs]
    stats = [mean_var(g) for g in gs]
    if any(v == 0 for _, v in stats):
        return _nan_fields(df_num=dfn, df_denom=NAN)
    w = [n / v for n, (_, v) in zip(ns, stats)]
    sw = sum(w)
    yhat = sum(wi * m for wi, (m, _) in zip(w, stats)) / sw
    num = sum(wi * (m - yhat) ** 2 for wi, (m, _) in zip(w, stats)) / (k - 1)
    tmp = sum((1 - wi / sw) ** 2 / (n - 1) for wi, n in zip(w, ns))
    F = num / (1 + Fraction(2 * (k - 2), k * k - 1) * tmp)
    return _f_fields(F, dfn, Fraction(k * k - 1) / (3 * tmp), df_varies=True)


@family("alexandergovern", ("alexandergovern",))
def alexandergovern(p):
    """Alexander and Govern's statistic (scipy's formulas: Hill's normalizing transformation of
    each group's t) in Arb from the exact means and variances; the chi^2 (k - 1) tail."""
    gs = [fvec(g) for g in p.args]
    k = len(gs)
    if min(len(g) for g in gs) < 2:
        return _nan_fields(df=Fraction(k - 1))
    stats = [mean_var(g) for g in gs]
    if any(v == 0 for _, v in stats):
        return _nan_fields(df=Fraction(k - 1))
    se2 = [v / len(g) for g, (_, v) in zip(gs, stats)]
    inv = [1 / s for s in se2]
    mw = sum(i * m for i, (m, _) in zip(inv, stats)) / sum(inv)  # exact: the weights are rational

    def A_stat():
        tot = arb(0)
        for g, (m, _), s2 in zip(gs, stats, se2):
            t = A(m - mw) / A(s2).sqrt()
            v = arb(len(g) - 1)
            a = v - arb(1) / 2
            b = 48 * a * a
            c = (a * (t * t / v).log1p()).sqrt()
            z = (
                c
                + (c**3 + 3 * c) / b
                - (4 * c**7 + 33 * c**5 + 240 * c**3 + 855 * c) / (10 * b * b + 8 * b * c**4 + 1000 * b)
            )
            tot += z * z
        return tot

    return {
        "statistic": A_stat,
        "pvalue": lambda: chi2_sf(A_stat(), k - 1),
        "df": Fraction(k - 1),
        "kappa": kappa(chi2_p_arb, A_stat, (k - 1,)),
    }


@family("bartlett", ("bartlett",))
def bartlett(p):
    """Bartlett's T: exact variances, Arb logarithms (the two large terms cancel when the
    variances are nearly equal), the chi^2 (k - 1) tail. A group of fewer than two or a
    constant group gives NaN (H11, H20)."""
    gs = [fvec(g) for g in p.args]
    k = len(gs)
    ns = [len(g) for g in gs]
    if min(ns) < 2:
        return _nan_fields(df=Fraction(k - 1))
    vs = [mean_var(g)[1] for g in gs]
    if any(v == 0 for v in vs):
        return _nan_fields(df=Fraction(k - 1))
    N = sum(ns)
    sp2 = sum((n - 1) * v for n, v in zip(ns, vs)) / (N - k)
    C = 1 + Fraction(1, 3 * (k - 1)) * (sum(Fraction(1, n - 1) for n in ns) - Fraction(1, N - k))

    def T():
        return sum((n - 1) * (A(sp2) / A(v)).log() for n, v in zip(ns, vs)) / A(C)

    return {
        "statistic": T,
        "pvalue": lambda: chi2_sf(T(), k - 1),
        "df": Fraction(k - 1),
        "kappa": kappa(chi2_p_arb, T, (k - 1,)),
    }


def center(xs, how, ptc):
    """The exact centre of a group (scipy's levene and fligner): the median (the exact average of
    the two middle values for an even size), the mean, or scipy's trim_mean (g =
    int(proportiontocut * n) as the double product cut from each end)."""
    s = sorted(xs)
    n = len(s)
    if how == "median":
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    if how == "mean":
        return sum(s) / n
    g = int(float(ptc) * n)
    mid = s[g : n - g]
    return sum(mid) / len(mid)


def _deviations(p):
    """|x - c| of each group (exact Fractions), c the group's centre."""
    how = p.kwargs.get("center", "median")
    ptc = p.kwargs.get("proportiontocut", 0.05)
    gs = [fvec(g) for g in p.args]
    return [[abs(x - center(g, how, ptc)) for x in g] if g else [] for g in gs]


@family("levene", ("levene",))
def levene(p):
    """Levene's W (and Brown-Forsythe's): the one-way ANOVA of the exact deviations from the
    exact centres; the F (k - 1, N - k) tail in Arb."""
    return _oneway(_deviations(p))


@family("fligner", ("fligner",), seconds=600)
def fligner(p):
    """Fligner-Killeen: exact deviations from the exact centres, their exact midranks r over all
    N values, the scores Phi^-1(1/2 + r/(2(N + 1))) in Arb (the normal's inverse survival
    function at the exact rational (N + 1 - r)/(2(N + 1))), the chi^2 (k - 1) tail."""
    zs = _deviations(p)
    k = len(zs)
    if min(len(z) for z in zs) == 0:
        return _nan_fields(df=Fraction(k - 1))
    flat = [v for z in zs for v in z]
    N = len(flat)
    ranks, _ = midranks(flat)
    if len(set(ranks)) == 1:
        return _nan_fields(df=Fraction(k - 1))
    sizes = [len(z) for z in zs]

    def S():
        cache = {}
        sc = []
        for r in ranks:
            if r not in cache:
                cache[r] = norm_isf((N + 1 - r) / (2 * (N + 1)))
            sc.append(cache[r])
        mean = sum(sc, arb(0)) / N
        var = sum(((a - mean) ** 2 for a in sc), arb(0)) / (N - 1)
        tot = arb(0)
        i = 0
        for n in sizes:
            mi = sum(sc[i : i + n], arb(0)) / n
            tot += n * (mi - mean) ** 2
            i += n
        return tot / var

    return {
        "statistic": S,
        "pvalue": lambda: chi2_sf(S(), k - 1),
        "df": Fraction(k - 1),
        "kappa": kappa(chi2_p_arb, S, (k - 1,)),
    }


LAMBDA = {
    "pearson": Fraction(1),
    "log-likelihood": Fraction(0),
    "freeman-tukey": Fraction(-1, 2),
    "mod-log-likelihood": Fraction(-1),
    "neyman": Fraction(-2),
    "cressie-read": Fraction(2, 3),
}


# ---- H2: goodness of fit to a given distribution


@family("power_divergence", ("chisquare", "power_divergence"))
def power_divergence(p):
    """The Cressie-Read family: Pearson's chi^2 exact, the others in Arb (O log(O/E) and
    (O/E)^lambda), the chi^2 (k - 1 - ddof) tail. A zero count O = 0 contributes its limit: 0 for
    lambda > -1, +inf for lambda <= -1 (scipy: NaN, 0 inf, for -1 < lambda < 0 and lambda < -1)."""
    obs = fvec(p.args[0])
    k = len(obs)
    exp = fvec(p.args[1]) if len(p.args) > 1 else [sum(obs) / k] * k
    lam = p.kwargs.get("lambda_", "pearson") if p.func == "power_divergence" else "pearson"
    lam = LAMBDA[lam] if isinstance(lam, str) else fr(lam)
    df = k - 1 - p.kwargs.get("ddof", 0)
    if lam == 1:
        stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
        if df <= 0:
            return {"statistic": stat, "pvalue": NAN, "df": Fraction(df), "kappa": NAN}
        return {
            "statistic": stat,
            "pvalue": lambda: chi2_sf(stat, df),
            "df": Fraction(df),
            "kappa": kappa(chi2_p_arb, stat, (df,)),
        }
    if lam <= -1 and any(o == 0 for o in obs):
        return {"statistic": math.inf, "pvalue": 0.0, "df": Fraction(df), "kappa": NAN}

    def S():
        if lam == 0:
            return 2 * sum((A(o) * (A(o) / A(e)).log() if o else arb(0)) for o, e in zip(obs, exp))
        if lam == -1:
            return 2 * sum(A(e) * (A(e) / A(o)).log() for o, e in zip(obs, exp))
        L = A(lam)
        return 2 / (L * (L + 1)) * sum((A(o) * ((A(o) / A(e)) ** L - 1) if o else arb(0)) for o, e in zip(obs, exp))

    return {
        "statistic": S,
        "pvalue": lambda: chi2_sf(S(), df),
        "df": Fraction(df),
        "kappa": kappa(chi2_p_arb, S, (df,)),
    }


def _moments(x):
    """The exact mean and the central moments m2, m3, m4 (over n) of rationals."""
    n = len(x)
    m = sum(x) / n
    return n, m, *(sum((v - m) ** k for v in x) / n for k in (2, 3, 4))


@family("jarque_bera", ("jarque_bera",))
def jarque_bera(p):
    """JB = n/6 (S^2 + (K - 3)^2/4) with the biased moments: rational (S^2 = m3^2/m2^3); its p-value
    is the chi^2 (2) tail, exactly exp(-JB/2) (sd computes it as ChiSquared(df=2).ccdf, H17). A
    constant sample is NaN (H20)."""
    n, _, m2, m3, m4 = _moments(fvec(p.args[0]))
    if m2 == 0:
        return _nan_fields()
    jb = Fraction(n, 6) * (m3 * m3 / m2**3 + (m4 / (m2 * m2) - 3) ** 2 / 4)
    return {"statistic": jb, "pvalue": lambda: (-A(jb) / 2).exp(), "kappa": float(jb) / 2}


def _skew_z(n, b):
    """D'Agostino's Z of the sample skewness b (an arb): scipy's formula, as asinh."""
    y = b * (A(Fraction((n + 1) * (n + 3), 6 * (n - 2)))).sqrt()
    beta2 = A(Fraction(3 * (n * n + 27 * n - 70) * (n + 1) * (n + 3), (n - 2) * (n + 5) * (n + 7) * (n + 9)))
    W2 = -1 + (2 * (beta2 - 1)).sqrt()
    delta = 1 / (W2.log() / 2).sqrt()
    alpha = (2 / (W2 - 1)).sqrt()
    return delta * (y / alpha).asinh()


def _kurt_z(n, b2):
    """Anscombe and Glynn's Z of the sample kurtosis b2 (an arb): scipy's formula."""
    E = A(Fraction(3 * (n - 1), n + 1))
    varb2 = A(Fraction(24 * n * (n - 2) * (n - 3), (n + 1) ** 2 * (n + 3) * (n + 5)))
    x = (b2 - E) / varb2.sqrt()
    sb1 = (
        A(Fraction(6 * (n * n - 5 * n + 2), (n + 7) * (n + 9)))
        * A(Fraction(6 * (n + 3) * (n + 5), n * (n - 2) * (n - 3))).sqrt()
    )
    Aa = 6 + 8 / sb1 * (2 / sb1 + (1 + 4 / (sb1 * sb1)).sqrt())
    term1 = 1 - 2 / (9 * Aa)
    denom = 1 + x * (2 / (Aa - 4)).sqrt()
    if not (denom > 0 or denom < 0):
        raise NotSettled("denom straddles 0")
    q = (1 - 2 / Aa) / denom
    term2 = q.root(3) if q > 0 else -((-q).root(3))
    return (term1 - term2) / (2 / (9 * Aa)).sqrt()


def _cond(f, v):
    """|v f'(v) / f(v)| by a central difference in Arb at KAPPA_BITS (f of an arb; NaN at f = 0)."""
    with ar.precision(KAPPA_BITS):
        v = v() if callable(v) else A(v)
        fv = f(v)
        if fv.is_zero() or not fv.is_finite():
            return NAN
        h = arb(2) ** -120
        return float(abs((f(v * (1 + h)) - f(v * (1 - h))) / (2 * h * fv)).mid())


@family("dagostino", ("skewtest", "kurtosistest", "normaltest"))
def dagostino(p):
    """D'Agostino's skewness test, Anscombe and Glynn's kurtosis test and their omnibus K^2 (scipy's
    formulas in Arb) of the exact sample skewness and kurtosis; the normal and chi^2 (2) tails.
    'kappa_stat' is the statistic's condition number in the moment it transforms (the
    kurtosis test's Z crosses 0 where the kurtosis is E)."""
    n, _, m2, m3, m4 = _moments(fvec(p.args[0]))
    alt = p.kwargs.get("alternative", "two-sided")
    need = 5 if p.func == "kurtosistest" else 8
    if n < need or m2 == 0:
        return {**_nan_fields(), "kappa_stat": NAN}
    b2 = m4 / (m2 * m2)

    def b():
        return A(m3) / (A(m2) * A(m2).sqrt())

    if p.func == "skewtest":
        Z = lambda: _skew_z(n, b())  # noqa: E731
        ks = _cond(lambda v: _skew_z(n, v), b) if m3 != 0 else 1.0
        return {
            "statistic": Z,
            "pvalue": lambda: norm_p(Z(), alt),
            "kappa": kappa(lambda z: norm_p(z, alt), Z),
            "kappa_stat": ks,
        }
    if p.func == "kurtosistest":
        Z = lambda: _kurt_z(n, A(b2))  # noqa: E731
        return {
            "statistic": Z,
            "pvalue": lambda: norm_p(Z(), alt),
            "kappa": kappa(lambda z: norm_p(z, alt), Z),
            "kappa_stat": _cond(lambda v: _kurt_z(n, v), b2),
        }

    def K2():
        return _skew_z(n, b()) ** 2 + _kurt_z(n, A(b2)) ** 2

    with ar.precision(KAPPA_BITS):
        bb = b()
        k2 = K2()
        ks = (_cond(lambda v: _skew_z(n, v) ** 2 + _kurt_z(n, A(b2)) ** 2, bb) if m3 != 0 else 0.0) + _cond(
            lambda v: _skew_z(n, bb) ** 2 + _kurt_z(n, v) ** 2, b2
        )
        del k2
    return {
        "statistic": K2,
        "pvalue": lambda: (-K2() / 2).exp(),
        "kappa": kappa(chi2_p_arb, K2, (2,)),
        "kappa_stat": ks,
    }


# Royston's AS R94 constants (scipy's _swilk and R's swilk.c), as the decimals written there
_SW_C1 = [Fraction(s) for s in ("0.221157", "-0.147981", "-2.071190", "4.434685", "-2.706056")]
_SW_C2 = [Fraction(s) for s in ("0.042981", "-0.293762", "-1.752461", "5.682633", "-3.582633")]
_SW_SMALL = (
    Fraction("-2.273"),
    Fraction("0.459"),
    [Fraction(s) for s in ("0.5440", "-0.39978", "0.025054", "-0.0006714")],
    [Fraction(s) for s in ("1.3822", "-0.77857", "0.062767", "-0.0020322")],
)
_SW_LARGE = ([Fraction(s) for s in ("-1.5861", "-0.31082", "-0.083751", "0.0038915")],
             [Fraction(s) for s in ("-0.4803", "-0.082676", "0.0030302")])  # fmt: skip


def _poly(cs, u, first=0):
    """first + sum c_i u^(i+1) (Royston's corrections) when first is given, else sum c_i u^i."""
    return sum(A(c) * u ** (i + 1) for i, c in enumerate(cs)) + first


def _sw_coefficients(n):
    """AS R94's a_1..a_n (arbs) at the current precision."""
    half = n // 2
    m = [-norm_isf((Fraction(i + 1) - Fraction(3, 8)) / (n + Fraction(1, 4))) for i in range(half)]
    mtm = 2 * sum(v * v for v in m)
    u = 1 / arb(n).sqrt()
    mn = -m[0]
    an = _poly(_SW_C1, u, mn / mtm.sqrt())
    a = [arb(0)] * n
    if n > 5:
        mnm1 = -m[1]
        anm1 = _poly(_SW_C2, u, mnm1 / mtm.sqrt())
        phi = (mtm - 2 * mn * mn - 2 * mnm1 * mnm1) / (1 - 2 * an * an - 2 * anm1 * anm1)
    else:
        phi = (mtm - 2 * mn * mn) / (1 - 2 * an * an)
    for i in range(half):
        a[i] = m[i] / phi.sqrt()
        a[n - 1 - i] = -a[i]
    a[0], a[n - 1] = -an, an
    if n > 5:
        a[1], a[n - 2] = -anm1, anm1
    return a


def _sw_p(n, one_m_w):
    """Royston's p-value of 1 - W (an arb)."""
    y = one_m_w.log()
    if n <= 11:
        g0, g1, cm, cs = _SW_SMALL
        gamma = A(g0 + g1 * n)
        mu = A(sum(c * Fraction(n) ** i for i, c in enumerate(cm)))
        sigma = A(sum(c * Fraction(n) ** i for i, c in enumerate(cs))).exp()
        if not (gamma - y > 0):
            if gamma - y < 0 or (gamma - y).is_zero():
                return arb(0)
            raise NotSettled("gamma - log(1 - W) straddles 0")
        gw = -(gamma - y).log()
    else:
        cm, cs = _SW_LARGE
        ln = arb(n).log()
        mu = sum(A(c) * ln**i for i, c in enumerate(cm))
        sigma = sum(A(c) * ln**i for i, c in enumerate(cs)).exp()
        gw = y
    return norm_sf((gw - mu) / sigma)


@family("shapiro", ("shapiro",), seconds=900)
def shapiro(p):
    """The Shapiro-Wilk W of AS R94 (the coefficients from the exact normal quantiles of
    (i - 3/8)/(n + 1/4) and Royston's decimal constants) and Royston's p-value, in Arb; 'kappa' is
    the p-value's condition number in 1 - W (the quantity sd forms directly)."""
    x = sorted(fvec(p.args[0]))
    n = len(x)
    _, m, m2, _, _ = _moments(x)
    if n < 3 or m2 == 0:
        return _nan_fields()
    sst = m2 * n

    def parts():
        if n == 3:
            c = A(Fraction(1, 2)).sqrt()
            S = c * (A(x[2]) - A(x[0]))
        else:
            a = _sw_coefficients(n)
            S = sum(ai * A(xi) for ai, xi in zip(a, x))
        return S * S / A(sst)

    def W():
        w = parts()
        return amax([w, A(Fraction(3, 4))]) if n == 3 else w

    def pv():
        w = W()
        if n == 3:
            q = 6 / arb.pi() * (w.sqrt().asin() - arb.pi() / 3)
            return q
        return _sw_p(n, 1 - w)

    if n == 3:
        return {
            "statistic": W,
            "pvalue": pv,
            "kappa": _cond(lambda v: 6 / arb.pi() * (v.sqrt().asin() - arb.pi() / 3), W),
        }
    return {"statistic": W, "pvalue": pv, "kappa": _cond(lambda v: _sw_p(n, v), lambda: 1 - parts())}


# Stephens' critical values (scipy's _Avals_*) and their levels, by family
_AD = {
    "norm": ((0.561, 0.631, 0.752, 0.873, 1.035), (15, 10, 5, 2.5, 1), lambda N: 1.0 + 0.75 / N + 2.25 / N / N),
    "expon": ((0.916, 1.062, 1.321, 1.591, 1.959), (15, 10, 5, 2.5, 1), lambda N: 1.0 + 0.6 / N),
    "logistic": ((0.426, 0.563, 0.660, 0.769, 0.906, 1.010), (25, 10, 5, 2.5, 1, 0.5), lambda N: 1.0 + 0.25 / N),
    "gumbel_r": ((0.474, 0.637, 0.757, 0.877, 1.038), (25, 10, 5, 2.5, 1), lambda N: 1.0 + 0.2 / math.sqrt(N)),
}


def _ad_table(dist, N):
    """scipy's critical values (np.around(Avals / den, 3) in doubles) and levels (sig / 100)."""
    av, sig, den = _AD[dist]
    d = den(N)
    crit = [round((a / d) * 1000.0) / 1000.0 for a in av]
    return [Fraction(c) for c in crit], [Fraction(s / 100) for s in sig]


def _ad_interp(a2, xp, fp):
    """np.interp at a ball a2: the segment of its midpoint, the ball's radius times the largest
    slope added (the interpolant is continuous)."""
    mid = a2.mid()
    if float(mid) <= xp[0]:
        v = A(fp[0])
    elif float(mid) >= xp[-1]:
        v = A(fp[-1])
    else:
        j = max(i for i in range(len(xp) - 1) if xp[i] <= float(mid))
        v = A(fp[j]) + A((fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])) * (arb(mid) - A(xp[j]))
    smax = max(abs((fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i])) for i in range(len(xp) - 1))
    return ar.add_error(v, arb(a2.rad()) * A(smax))


def _logistic_mle(x):
    """The logistic's maximum likelihood (location, scale) of rationals, by Newton's steps on
    (alpha, beta), w = beta x - alpha, at the current precision; the last step's size is added to
    the balls."""
    xs = np.asarray([float(v) for v in x])
    al, be = (
        float(np.mean(xs)) * (math.pi / math.sqrt(3) / np.std(xs, ddof=1)),
        math.pi / math.sqrt(3) / np.std(xs, ddof=1),
    )
    al, be = arb(al), arb(be)
    X = [A(v) for v in x]
    N = len(X)
    step = arb(1)
    for _ in range(200):
        g1 = g2 = h11 = h12 = h22 = arb(0)
        for v in X:
            th = ((be * v - al) / 2).tanh()
            h = -(1 - th * th) / 2
            g1 += th
            g2 -= v * th
            h11 += h
            h12 -= h * v
            h22 += h * v * v
        g2 += N / be
        h22 -= N / (be * be)
        det = h11 * h22 - h12 * h12
        da = -(h22 * g1 - h12 * g2) / det
        db = -(-h12 * g1 + h11 * g2) / det
        al, be = (al + da).mid(), (be + db).mid()
        al, be = arb(al), arb(be)
        step = abs(da) + abs(db)
        if step.mid() < arb(2) ** (-flint_ctx.prec + 8) * (1 + abs(be.mid())):
            break
    r = arb(abs(step.mid())) * 16 + arb(2) ** (-flint_ctx.prec + 8)
    return ar.add_error(al, r), ar.add_error(be, r)


def _gumbel_mle(x):
    """The Gumbel's (maximum's) maximum likelihood (loc, scale) of rationals: Newton on the scale's
    equation mean(x) - sum x e^(-x/b) / sum e^(-x/b) - b = 0 at the current precision."""
    xs = np.asarray([float(v) for v in x])
    X = [A(v) for v in x]
    N = len(X)
    mean = sum(X) / N
    xmin = A(min(x))

    def f(b):
        w = [(-(v - xmin) / b).exp() for v in X]
        s0 = sum(w)
        m1 = sum(v * e for v, e in zip(X, w)) / s0
        m2 = sum(v * v * e for v, e in zip(X, w)) / s0
        return mean - m1 - b, -(m2 - m1 * m1) / (b * b) - 1, s0

    b = arb(float(np.std(xs, ddof=1)) * math.sqrt(6) / math.pi)
    step = arb(1)
    for _ in range(200):
        v, d, _ = f(b)
        step = v / d
        b = arb((b - step).mid())
        if abs(step.mid()) < arb(2) ** (-flint_ctx.prec + 8) * abs(b.mid()):
            break
    b = ar.add_error(b, arb(abs(step.mid())) * 16 + arb(2) ** (-flint_ctx.prec + 8) * abs(b))
    _, _, s0 = f(b)
    loc = -b * (-xmin / b + s0.log() - arb(N).log())
    return loc, b


def _ad_terms(dist, x):
    """The (log F(w_i), log(1 - F(w_i))) of the sorted rationals x under the fitted family."""
    N = len(x)
    _, m, m2, _, _ = _moments(x)
    if dist == "norm":
        s = A(m2 * N / (N - 1)).sqrt()
        ws = [(A(v) - A(m)) / s for v in x]
        lc = [(-w / arb(2).sqrt()).erfc().log() - arb(2).log() for w in ws]
        ls = [(w / arb(2).sqrt()).erfc().log() - arb(2).log() for w in ws]
    elif dist == "expon":
        ws = [A(v / m) for v in x]
        lc = [(1 - (-w).exp()).log() for w in ws]
        ls = [-w for w in ws]
    elif dist == "logistic":
        al, be = _logistic_mle(x)
        ws = [be * A(v) - al for v in x]
        lc = [-(1 + (-w).exp()).log() for w in ws]
        ls = [-(1 + w.exp()).log() for w in ws]
    else:
        loc, sc = _gumbel_mle(x)
        ws = [(A(v) - loc) / sc for v in x]
        lc = [-(-w).exp() for w in ws]
        ls = [(1 - (-(-w).exp()).exp()).log() for w in ws]
    return lc, ls


@family("anderson", ("anderson",), seconds=900)
def anderson(p):
    """A^2 = -N - sum (2i - 1)/N (log F(w_(i)) + log(1 - F(w_(N+1-i)))) under the fitted family
    (the normal's mean and standard deviation, the exponential's mean, the logistic's and the
    Gumbel's maximum likelihood estimates solved in Arb), and scipy's interpolated p-value from
    its tables computed in doubles as numpy does. 'kappa_stat' is the condition number of the sum
    (A^2 is a sum of N terms of size ~1 cancelling to ~1), 'kappa' the interpolation's."""
    x = sorted(fvec(p.args[0]))
    dist = p.kwargs.get("dist", "norm")
    N = len(x)
    _, _, m2, _, _ = _moments(x)
    if N < 2 or m2 == 0:
        return {**_nan_fields(), "kappa_stat": NAN}
    xp, fp = _ad_table(dist, float(N))

    def A2():
        lc, ls = _ad_terms(dist, x)
        return -N - sum(A(Fraction(2 * i + 1, N)) * (lc[i] + ls[N - 1 - i]) for i in range(N))

    with ar.precision(200):
        lc, ls = _ad_terms(dist, x)
        tot = sum(abs(float((A(Fraction(2 * i + 1, N)) * (lc[i] + ls[N - 1 - i])).mid())) for i in range(N))
        a2 = float((-N - sum(A(Fraction(2 * i + 1, N)) * (lc[i] + ls[N - 1 - i]) for i in range(N))).mid())
        pf = float(_ad_interp(arb(a2), xp, fp).mid())
    ks = (N + tot) / abs(a2)
    slope = 0.0
    for j in range(len(xp) - 1):
        if xp[j] <= a2 < xp[j + 1]:
            slope = float((fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]))
    return {
        "statistic": A2,
        "pvalue": lambda: _ad_interp(A2(), xp, fp),
        "kappa": abs(a2 * slope / pf),
        "kappa_stat": ks,
    }


def _cdf(name):
    """The cdf of scipy's named standard distribution, of a rational, as a ball."""
    if name == "norm":
        return lambda v: (-A(v) / arb(2).sqrt()).erfc() / 2
    if name == "uniform":
        return lambda v: arb(0) if v <= 0 else arb(1) if v >= 1 else A(v)
    if name == "expon":
        return lambda v: arb(0) if v <= 0 else 1 - (-A(v)).exp()
    raise KeyError(name)


def _phi(x):
    return (-A(fr(x)) / arb(2).sqrt()).erfc() / 2


def _ks_d(x, name="norm"):
    """D+ and D- of the sorted sample against the named distribution, as balls."""
    xs = sorted(fr(v) for v in np.ravel(x))
    n = len(xs)
    F = [_cdf(name)(v) for v in xs]
    dplus = amax([A(Fraction(i + 1, n)) - f for i, f in enumerate(F)])
    dminus = amax([f - A(Fraction(i, n)) for i, f in enumerate(F)])
    return n, dplus, dminus


def _decided_floor(v):
    """floor(v) of a ball that does not straddle an integer (else NotSettled)."""
    lo, hi = math.floor(float(v.lower())), math.floor(float(v.upper()))
    if lo != hi or v.lower() == lo and not v.is_exact():
        raise NotSettled("an integer boundary inside the ball")
    return lo


def kolmogorov_cdf_exact(n, d):
    """P(D_n < d) by Marsaglia, Tsang and Wang (2003): k = ceil(n d), h = k - n d, the m x m
    matrix H (m = 2k - 1) and n!/n^n (H^n)_{kk}, in Arb (d a ball)."""
    nd = n * d
    k = _decided_floor(nd) + 1
    h = k - nd
    m = 2 * k - 1
    rows = []
    for i in range(m):
        row = []
        for j in range(m):
            e = i - j + 1
            row.append(arb(1) / arb.fac_ui(e) if e >= 0 else arb(0))
        rows.append(row)
    for i in range(m):
        rows[i][0] -= h ** (i + 1) / arb.fac_ui(i + 1)
        rows[m - 1][i] -= h ** (m - i) / arb.fac_ui(m - i)
    if (2 * h - 1) > 0:
        rows[m - 1][0] += (2 * h - 1) ** m / arb.fac_ui(m)
    elif not ((2 * h - 1) < 0 or (2 * h - 1).is_zero()):  # (exactly 0: the term is 0)
        raise NotSettled("2h - 1 straddles 0")
    H = arb_mat(rows)
    P = H**n
    return P[k - 1, k - 1] * arb.fac_ui(n) / arb(n) ** n


def smirnov_sf(n, d):
    """P(D_n^+ >= d), Birnbaum and Tingey (1951): d sum_{j <= n(1 - d)} C(n, j) (1 - d - j/n)^(n - j)
    (d + j/n)^(j - 1), positive terms."""
    top = _decided_floor(n * (1 - d))
    s = arb(0)
    for j in range(top + 1):
        a = 1 - d - arb(j) / n
        s += arb(math.comb(n, j)) * a ** (n - j) * (d + arb(j) / n) ** (j - 1)
    return d * s


def kolmogorov_sf(lam):
    """Kolmogorov's limit Q(lambda) = 2 sum_{k >= 1} (-1)^(k-1) exp(-2 k^2 lambda^2), its tail bounded
    by the first term left out (alternating, decreasing)."""
    s = arb(0)
    k = 1
    while True:
        t = (-2 * k * k * lam * lam).exp()
        s += t if k % 2 else -t
        nxt = (-2 * (k + 1) ** 2 * lam * lam).exp()
        if nxt < arb(2) ** (-ar.TARGET_BITS - 20) * abs(s) or k > 10000:
            return ar.add_error(2 * s, 2 * nxt)
        k += 1


@family("ks_1samp", ("ks_1samp",), seconds=900)
def ks_1samp(p):
    """One-sample Kolmogorov-Smirnov against the named distribution (Arb erfc for the normal): the
    exact two-sided distribution (scipy's kstwo; 'auto' and 'exact'), twice the one-sided one
    ('approx'), Kolmogorov's limit ('asymp'); the one-sided tests by Smirnov's exact distribution
    (scipy's ksone, whatever the method). 'kappa' is the p-value's condition number in D."""
    alt = p.kwargs.get("alternative", "two-sided")
    method = p.kwargs.get("method", "auto")

    def stat():
        n, dp, dm = _ks_d(p.args[0], p.args[1])
        return n, (dp if alt == "greater" else dm if alt == "less" else amax([dp, dm]))

    n = int(np.size(p.args[0]))

    def pfun(d):
        if alt != "two-sided":
            return smirnov_sf(n, d)
        if method in ("auto", "exact"):
            if d > arb(0.5):  # (both tails cannot be crossed)
                return 2 * smirnov_sf(n, d)
            return 1 - kolmogorov_cdf_exact(n, d)
        if method == "asymp":
            return kolmogorov_sf(arb(n).sqrt() * d)
        return 2 * smirnov_sf(n, d)

    used = method if alt == "two-sided" else "exact"
    return {
        "statistic": lambda: stat()[1],
        "pvalue": lambda: pfun(stat()[1]),
        "method": "exact" if used == "auto" else used,
        "kappa": kappa(pfun, lambda: stat()[1]),
    }


def _series(term, tail_bits=20):
    """sum_{k >= 0} term(k) until a term is below 2^-(prec + tail_bits) of the sum; that term's
    size is added to the ball (the terms fall faster than geometrically there)."""
    tot = arb(0)
    for k in range(100000):
        t = term(k)
        tot += t
        if k > 0 and abs(t) < arb(2) ** (-flint_ctx.prec - tail_bits) * abs(tot):
            return ar.add_error(tot, abs(t))
    raise NotSettled("series")


def cvm_cdf_inf(x):
    """Csörgő and Faraway's V(x) (eq. 1.2, 1.3), x an arb."""
    pi = arb.pi()

    def term(k):
        y = arb(4 * k + 1)
        q = y * y / (16 * x)
        u = (A(k + Fraction(1, 2)).lgamma() - arb(k + 1).lgamma()).exp() / (pi * pi.sqrt() * x.sqrt())
        return u * y.sqrt() * (-q).exp() * q.bessel_k(A(Fraction(1, 4)))

    return _series(term)


def cvm_psi1_mod(x):
    """Their psi1(x) without V(x)/12 (eq. 1.10; scipy's _psi1_mod), x an arb."""
    pi = arb.pi()
    sx = 2 * x.sqrt()
    y1, y2 = x ** A(Fraction(3, 4)), x ** A(Fraction(5, 4))
    k14, k34, k54 = A(Fraction(1, 4)), A(Fraction(3, 4)), A(Fraction(5, 4))

    def ed2(y):
        z = y * y / 4
        return (-z).exp() * (y / 2) ** A(Fraction(3, 2)) * (z.bessel_k(k14) + z.bessel_k(k34)) / pi.sqrt()

    def ed3(y):
        z = y * y / 4
        return (
            (-z).exp()
            * (y / 2) ** A(Fraction(5, 2))
            * (2 * z.bessel_k(k14) + 3 * z.bessel_k(k34) - z.bessel_k(k54))
            / pi.sqrt()
        )

    def term(k):
        m = 2 * k + 1
        g1 = (A(k + Fraction(1, 2)).lgamma() - arb(k + 1).lgamma()).exp()
        g3 = g1 * A(k + Fraction(1, 2))
        e1 = m * g1 * ed2((4 * k + 3) / sx) / (9 * y1)
        e2 = g1 * ed3((4 * k + 1) / sx) / (72 * y2)
        e3 = 2 * (m + 2) * g3 * ed3((4 * k + 5) / sx) / (12 * y2)
        e4 = 7 * m * g1 * ed2((4 * k + 1) / sx) / (144 * y1)
        e5 = 7 * m * g1 * ed2((4 * k + 5) / sx) / (144 * y1)
        return -(e1 + e2 + e3 + e4 + e5) / pi

    return _series(term)


def cvm_sf(x, n):
    """max(0, 1 - F_n(x)) by Csörgő and Faraway's form (1 below 1/(12n), 0 from n/3)."""
    lo, hi = A(Fraction(1, 12 * n)), A(Fraction(n, 3))
    if x < lo or (x - lo).is_zero():
        return arb(1)
    if x > hi or (x - hi).is_zero():
        return arb(0)
    if not (x > lo and x < hi):
        raise NotSettled("on the support's end")
    v = 1 - (cvm_cdf_inf(x) * A(1 + Fraction(1, 12 * n)) + cvm_psi1_mod(x) / n)
    if v < 0:
        return arb(0)
    if v > 0:
        return v
    raise NotSettled("1 - F_n straddles 0")


@family("cramervonmises", ("cramervonmises",), seconds=900)
def cramervonmises(p):
    """W^2 = 1/(12n) + sum ((2i - 1)/(2n) - F(x_(i)))^2 against the named distribution (Arb), and
    max(0, 1 - F_n(W^2)) by Csörgő and Faraway's finite-sample form with its Bessel series in Arb
    (scipy's formula, its series not cut at 1e-7). n < 2 is NaN."""
    xs = sorted(fr(v) for v in np.ravel(p.args[0]))
    n = len(xs)
    if n < 2:
        return _nan_fields()
    F = _cdf(p.args[1])

    def W():
        return A(Fraction(1, 12 * n)) + sum((A(Fraction(2 * i + 1, 2 * n)) - F(v)) ** 2 for i, v in enumerate(xs))

    return {"statistic": W, "pvalue": lambda: cvm_sf(W(), n), "kappa": kappa(lambda w: cvm_sf(w, n), W)}


def mwu_counts(m, n):
    """The number of arrangements with U = u, u = 0..mn (the Gaussian binomial coefficient's
    coefficients), by c(m, n, u) = c(m - 1, n, u - n) + c(m, n - 1, u)."""
    prev = [[1] for _ in range(n + 1)]  # m = 0: one arrangement, U = 0
    for i in range(1, m + 1):
        cur = [[1]]  # n = 0
        for j in range(1, n + 1):
            a, b = prev[j], cur[j - 1]  # c(i - 1, j, u - j), c(i, j - 1, u)
            c = [0] * (i * j + 1)
            for u, v in enumerate(a):
                c[u + j] += v
            for u, v in enumerate(b):
                c[u] += v
            cur.append(c)
        prev = cur
    return prev[n]


def subset_sum_counts(scores, c):
    """{sum: count} of the sums of c of the integer scores (with repeats) drawn without
    replacement: the exact null of a two-sample linear rank statistic, conditional on the scores
    (with ties, the doubled midranks: decision H45). A dynamic programme over the distinct scores,
    in Python's integers."""
    from collections import Counter

    dp = [{} for _ in range(c + 1)]
    dp[0][0] = 1
    for v, t in sorted(Counter(scores).items()):
        new = [dict(d) for d in dp]
        for j in range(c):
            for s, cnt in dp[j].items():
                for cc in range(1, min(t, c - j) + 1):
                    key = s + cc * v
                    new[j + cc][key] = new[j + cc].get(key, 0) + cnt * math.comb(t, cc)
        dp = new
    return dp[c]


def signed_counts(scores):
    """{sum: count} of sum a_i B_i over the 2^n sign choices (B_i in {0, 1}) of integer scores."""
    d = {0: 1}
    for a in scores:
        nd = dict(d)
        for s, c in d.items():
            nd[s + a] = nd.get(s + a, 0) + c
        d = nd
    return d


def tails(counts, v):
    """(P(S <= v), P(S >= v)) of {sum: count}, exact."""
    tot = sum(counts.values())
    return (
        Fraction(sum(c for s, c in counts.items() if s <= v), tot),
        Fraction(sum(c for s, c in counts.items() if s >= v), tot),
    )


def exact_p(le, ge, alt, greater_is="ge"):
    """The exact p-value from the two tails: 'less' the lower, 'greater' the upper (or the reverse
    when greater_is='le': Ansari-Bradley's), two-sided min(1, 2 min) (scipy's rule)."""
    up, dn = (ge, le) if greater_is == "ge" else (le, ge)
    if alt == "greater":
        return up
    if alt == "less":
        return dn
    return min(Fraction(1), 2 * min(le, ge))


def _z_fields(num, var, alt, zfield="zstatistic", two_sided_neg=False, extra=None):
    """The normal approximation's fields: z = num / sqrt(var) (num, var Fractions), p, kappa."""
    if var <= 0:  # (every value tied: no normal approximation, H11)
        out = {"statistic": NAN, "pvalue": NAN, "kappa": NAN}
        out.update(extra or {})
        if zfield:
            out[zfield] = NAN
        return out

    def Z():
        return A(num) / A(var).sqrt()

    out = {"pvalue": lambda: norm_p(Z(), alt), "kappa": kappa(lambda z: norm_p(z, alt), Z)}
    if zfield:
        if two_sided_neg and alt == "two-sided":
            out[zfield] = lambda: -abs(Z())
        else:
            out[zfield] = Z
    if extra:
        out.update(extra)
    return out


@family("mannwhitneyu", ("mannwhitneyu",), seconds=300)
def mannwhitneyu(p):
    """Mann-Whitney U from exact midranks. Exact p-values from integer counts: without ties the
    Gaussian binomial's (scipy's rule: U = max(U1, U2) and twice the tail for two-sided), with
    ties the distribution of the doubled rank sum conditional on the midranks (H45; both tails
    counted, two-sided min(1, 2 min)). The normal approximation with scipy's tie and continuity
    corrections, and its z (H47)."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    n1, n2 = len(x), len(y)
    r, ties = midranks(x + y)
    u1 = sum(r[:n1]) - Fraction(n1 * (n1 + 1), 2)
    u2 = n1 * n2 - u1
    alt = p.kwargs.get("alternative", "two-sided")
    method = p.kwargs.get("method", "auto")
    has_ties = any(t > 1 for t in ties)
    if method == "auto":
        method = "asymptotic" if (n1 > 8 and n2 > 8) or has_ties else "exact"
    if method == "exact":
        if not has_ties:
            U, f = (u1, 1) if alt == "greater" else (u2, 1) if alt == "less" else (max(u1, u2), 2)
            c = mwu_counts(n1, n2)
            tot = sum(c)
            pv = min(Fraction(1), f * Fraction(sum(c[math.ceil(U) :]), tot))
        else:
            le, ge = tails(subset_sum_counts([int(2 * v) for v in r], n1), int(2 * sum(r[:n1])))
            pv = exact_p(le, ge, alt)
        return {"statistic": u1, "pvalue": pv, "method": "exact"}
    n = n1 + n2
    s2 = Fraction(n1 * n2, 12) * ((n + 1) - Fraction(sum(t**3 - t for t in ties), n * (n - 1)))
    num = u1 - Fraction(n1 * n2, 2)
    if p.kwargs.get("use_continuity", True):
        sg = 1 if alt == "greater" else -1 if alt == "less" else sign(num)
        num -= Fraction(sg, 2)
    return _z_fields(num, s2, alt, extra={"statistic": u1, "method": "asymptotic"})


def signed_rank_counts(n):
    """The number of subsets of {1..n} with sum t, t = 0..n(n+1)/2."""
    c = [1]
    for i in range(1, n + 1):
        nxt = c + [0] * i
        for t, v in enumerate(c):
            nxt[t + i] += v
        c = nxt
    return c


@family("wilcoxon", ("wilcoxon",), seconds=300)
def wilcoxon(p):
    """The Wilcoxon signed rank test: exact midranks of |d| (d = x - y exact), T+. Exact p-values
    from counts: without ties or zeros the subset sums of 1..n; with them the distribution of
    sum r_i B_i over the nonzero differences' midranks (H45; zsplit adds half the zeros' ranks,
    pratt's zeros keep their ranks out of it), which is the signs' enumeration scipy's 'auto'
    does up to n = 13. The normal approximation with scipy's zero methods and tie correction,
    and its z (-|z| for two-sided, scipy's)."""
    d = fvec(p.args[0]) if len(p.args) == 1 else [a - b for a, b in zip(fvec(p.args[0]), fvec(p.args[1]))]
    zm = p.kwargs.get("zero_method", "wilcox")
    alt = p.kwargs.get("alternative", "two-sided")
    method = p.kwargs.get("method", "auto")
    n_all = len(d)
    zeros = sum(1 for v in d if v == 0)
    if zm == "wilcox":
        d = [v for v in d if v != 0]
    r, ties = midranks([abs(v) for v in d])
    rp = sum((ri for ri, v in zip(r, d) if v > 0), Fraction(0))
    rm = sum((ri for ri, v in zip(r, d) if v < 0), Fraction(0))
    rz = sum((ri for ri, v in zip(r, d) if v == 0), Fraction(0)) / 2
    if zm == "zsplit":
        rp, rm = rp + rz, rm + rz
    has_ties = any(t > 1 for t in ties)
    if method == "auto":
        if n_all > 50:
            method = "asymptotic"
        elif not (has_ties or zeros) or n_all <= 13:
            method = "exact"
        else:
            method = "asymptotic"
    stat = min(rp, rm) if alt == "two-sided" else rp
    count = len(d)
    if count == 0:
        return {"statistic": NAN, "pvalue": NAN, "method": method, "kappa": NAN}
    if method == "exact":
        if not (has_ties or zeros):
            c = signed_rank_counts(count)
            tot = sum(c)
            cdf = Fraction(sum(c[: math.ceil(rp) + 1]), tot)
            sf = Fraction(sum(c[math.floor(rp) :]), tot)
        else:
            konst = 2 * rz if zm == "zsplit" else 0
            counts = signed_counts([int(2 * ri) for ri, v in zip(r, d) if v != 0])
            cdf, sf = tails(counts, int(2 * rp - konst))
        return {"statistic": stat, "pvalue": exact_p(cdf, sf, alt), "method": "exact"}
    mn = Fraction(count * (count + 1), 4)
    se = Fraction(count * (count + 1) * (2 * count + 1))
    tt = ties
    if zm == "pratt":
        mn -= Fraction(zeros * (zeros + 1), 4)
        se -= zeros * (zeros + 1) * (2 * zeros + 1)
        tt = [t for t, g in zip(*_tie_groups([abs(v) for v in d])) if g != 0]
    se2 = (se - Fraction(sum(t**3 - t for t in tt), 2)) / 24
    num = rp - mn
    if p.kwargs.get("correction", False):
        sg = 1 if alt == "greater" else -1 if alt == "less" else sign(num)
        num -= Fraction(sg, 2)
    return _z_fields(num, se2, alt, two_sided_neg=True, extra={"statistic": stat, "method": "asymptotic"})


@family("ranksums", ("ranksums",))
def ranksums(p):
    """The rank sum z (scipy's: no tie or continuity correction) and its normal p-value."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    n1, n2 = len(x), len(y)
    r, _ = midranks(x + y)
    num = sum(r[:n1]) - Fraction(n1 * (n1 + n2 + 1), 2)
    out = _z_fields(
        num, Fraction(n1 * n2 * (n1 + n2 + 1), 12), p.kwargs.get("alternative", "two-sided"), zfield="statistic"
    )
    return out


@family("kruskal", ("kruskal",))
def kruskal(p):
    """Kruskal-Wallis H from exact midranks with scipy's tie correction (exact), chi^2 (k - 1)."""
    gs = [fvec(a) for a in p.args]
    k = len(gs)
    allv = [v for g in gs for v in g]
    N = len(allv)
    if any(len(g) == 0 for g in gs):
        return _nan_fields(df=Fraction(k - 1))
    r, ties = midranks(allv)
    pos, R = 0, []
    for g in gs:
        R.append(sum(r[pos : pos + len(g)]))
        pos += len(g)
    c = 1 - Fraction(sum(t**3 - t for t in ties), N**3 - N)
    if c == 0:
        return _nan_fields(df=Fraction(k - 1))
    H = (Fraction(12, N * (N + 1)) * sum(Ri**2 / len(g) for Ri, g in zip(R, gs)) - 3 * (N + 1)) / c
    return {
        "statistic": H,
        "pvalue": lambda: chi2_sf(H, k - 1),
        "df": Fraction(k - 1),
        "kappa": kappa(chi2_p_arb, H, (k - 1,)),
    }


def friedman_exact_sf(k, n, stat):
    """P(chi^2_F >= stat) under the exact null without ties: the k! orders of each of the n blocks
    equally likely, a dynamic programme over the vectors of column rank sums (Python integers).
    The chi^2 approximation's quality (docs/design/htest.md §4), not a contract."""
    from itertools import permutations

    perms = list(permutations(range(1, k + 1)))
    d = {tuple([0] * k): 1}
    for _ in range(n):
        nd = {}
        for key, c in d.items():
            for pm in perms:
                nk = tuple(a + b for a, b in zip(key, pm))
                nd[nk] = nd.get(nk, 0) + c
        d = nd
    tot = len(perms) ** n
    hit = 0
    for key, c in d.items():
        q = Fraction(12, n * k * (k + 1)) * sum(v * v for v in key) - 3 * n * (k + 1)
        if q >= stat:
            hit += c
    return Fraction(hit, tot)


@family("friedmanchisquare", ("friedmanchisquare",), seconds=300)
def friedmanchisquare(p):
    """The Friedman chi^2 from exact within-block midranks with scipy's tie correction (exact),
    chi^2 (k - 1). Small problems without ties also get "pvalue_exact": the exact permutation
    p-value (the approximation's quality; not a contract)."""
    cols = [fvec(a) for a in p.args]
    k, n = len(cols), len(cols[0])
    R = [Fraction(0)] * k
    T = 0
    any_ties = False
    for j in range(n):
        r, ties = midranks([c[j] for c in cols])
        for i in range(k):
            R[i] += r[i]
        T += sum(t * (t * t - 1) for t in ties)
        any_ties = any_ties or any(t > 1 for t in ties)
    c = 1 - Fraction(T, k * (k * k - 1) * n)
    if n == 0 or c == 0:
        return _nan_fields(df=Fraction(k - 1))
    stat = (Fraction(12, k * n * (k + 1)) * sum(v * v for v in R) - 3 * n * (k + 1)) / c
    out = {
        "statistic": stat,
        "pvalue": lambda: chi2_sf(stat, k - 1),
        "df": Fraction(k - 1),
        "kappa": kappa(chi2_p_arb, stat, (k - 1,)),
    }
    if not any_ties and math.factorial(k) ** n <= 10**8:
        out["pvalue_exact"] = friedman_exact_sf(k, n, stat)
    return out


@family("brunnermunzel", ("brunnermunzel",))
def brunnermunzel(p):
    """Brunner and Munzel's W from exact midranks (pooled and within each sample), its
    Satterthwaite df (exact), the t or normal p-value at -W (scipy's). Both placement variances 0
    (or a sample below 2) is NaN (H11)."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    nx, ny = len(x), len(y)
    alt = p.kwargs.get("alternative", "two-sided")
    dist = p.kwargs.get("distribution", "t")
    nan = _nan_fields(**({"df": NAN} if dist == "t" else {}))
    if nx < 2 or ny < 2:
        return nan
    rc, _ = midranks(x + y)
    rx, _ = midranks(x)
    ry, _ = midranks(y)
    rcx, rcy = rc[:nx], rc[nx:]
    mcx, mcy = sum(rcx) / nx, sum(rcy) / ny
    mx, my = sum(rx) / nx, sum(ry) / ny
    Sx = sum((a - b - mcx + mx) ** 2 for a, b in zip(rcx, rx)) / (nx - 1)
    Sy = sum((a - b - mcy + my) ** 2 for a, b in zip(rcy, ry)) / (ny - 1)
    V = nx * Sx + ny * Sy
    if V == 0:
        return nan
    num = Fraction(nx * ny, nx + ny) * (mcy - mcx)

    def W():
        return A(num) / A(V).sqrt()

    if dist == "normal":
        return {"statistic": W, "pvalue": lambda: norm_p(-W(), alt), "kappa": kappa(lambda w: norm_p(-w, alt), W)}
    df = V**2 / ((nx * Sx) ** 2 / (nx - 1) + (ny * Sy) ** 2 / (ny - 1))
    return {
        "statistic": W,
        "pvalue": lambda: t_p(num * num / V, -sign(num), df, alt),
        "df": df,
        "kappa": kappa(lambda w, nu: t_p_arb(-w, nu, alt), W, (df,), (0,)),
    }


@family("ansari", ("ansari",), seconds=600)
def ansari(p):
    """Ansari-Bradley's AB = sum over x of min(r, N + 1 - r), exact midranks. Exact p-values from
    the counts of the doubled scores' sums (conditional on them with ties, H45); scipy's normal
    approximation (its tie-corrected variance) otherwise. 'greater' is the lower tail (scipy)."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    n, m = len(x), len(y)
    N = n + m
    r, ties = midranks(x + y)
    sym = [min(ri, N + 1 - ri) for ri in r]
    AB = sum(sym[:n])
    alt = p.kwargs.get("alternative", "two-sided")
    method = p.kwargs.get("method", "auto")
    repeats = any(t > 1 for t in ties)
    if method == "auto":
        method = "exact" if m < 55 and n < 55 and not repeats else "asymptotic"
    if method == "exact":
        le, ge = tails(subset_sum_counts([int(2 * s) for s in sym], n), int(2 * AB))
        return {"statistic": AB, "pvalue": exact_p(le, ge, alt, greater_is="le"), "method": "exact"}
    mean = Fraction(n * (N + 1) ** 2, 4 * N) if N % 2 else Fraction(n * (N + 2), 4)
    if repeats:
        fac = sum(s * s for s in sym)
        var = (
            (m * n * (16 * N * fac - (N + 1) ** 4) / (16 * N**2 * (N - 1)))
            if N % 2
            else (m * n * (16 * fac - N * (N + 2) ** 2) / (16 * N * (N - 1)))
        )
    else:
        var = (
            Fraction(n * m * (N + 1) * (3 + N**2), 48 * N**2)
            if N % 2
            else Fraction(m * n * (N + 2) * (N - 2), 48 * (N - 1))
        )
    return _z_fields(mean - AB, var, alt, zfield=None, extra={"statistic": AB, "method": "asymptotic"})


@family("mood", ("mood",))
def mood(p):
    """Mood's z: T the sum over x of its tie run's mean of (I - (N + 1)/2)^2 (Mielke's, scipy's;
    without ties sum (r - (N + 1)/2)^2), E = m (N^2 - 1)/12 and Mielke's variance, exact."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    m, n = len(x), len(y)
    N = m + n
    if m == 0 or n == 0 or N < 3:
        return _nan_fields()
    allv = sorted((v, i < m) for i, v in enumerate(x + y))
    c = Fraction(N + 1, 2)
    T = Fraction(0)
    tie = 0
    i = 0
    while i < N:
        j = i
        while j + 1 < N and allv[j + 1][0] == allv[i][0]:
            j += 1
        t = j - i + 1
        phi = sum((Fraction(I) - c) ** 2 for I in range(i + 1, j + 2)) / t
        T += phi * sum(1 for q in range(i, j + 1) if allv[q][1])
        tie += t * (t * t - 1) * (t * t - 4 + 15 * (N - (j + 1) - i) ** 2)
        i = j + 1
    E = Fraction(m * (N * N - 1), 12)
    var = Fraction(m * n * (N + 1) * (N * N - 4), 180) - Fraction(m * n * tie, 180 * N * (N - 1))
    return _z_fields(T - E, var, p.kwargs.get("alternative", "two-sided"), zfield="statistic")


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _cressie_read(obs, exp, lam):
    """The Cressie-Read statistic of observed (>= 0) and expected (> 0) rationals: Pearson's exact,
    the others in Arb (power_divergence's forms; an observed 0 its limit)."""
    if lam == 1:
        return sum((o - e) ** 2 / e for o, e in zip(obs, exp))

    def S():
        if lam == 0:
            return 2 * sum((A(o) * (A(o) / A(e)).log() if o else arb(0)) for o, e in zip(obs, exp))
        if lam == -1:
            return 2 * sum(A(e) * (A(e) / A(o)).log() for o, e in zip(obs, exp))
        L = A(lam)
        return 2 / (L * (L + 1)) * sum((A(o) * ((A(o) / A(e)) ** L - 1) if o else arb(0)) for o, e in zip(obs, exp))

    return S


@family("median_test", ("median_test",))
def median_test(p):
    """Mood's median test: the exact grand median (the mean of the two middle values unrounded),
    the 2 x k table by exact comparisons, scipy's chi2_contingency of it (expected R_i C_j / N,
    Yates' correction for k = 2, the Cressie-Read lambda_), chi^2 (k - 1). A table with an empty
    row (or, with ties='ignore', column) is NaN (scipy raises)."""
    gs = [fvec(a) for a in p.args]
    k = len(gs)
    med = _median([v for g in gs for v in g])
    ties = p.kwargs.get("ties", "below")
    above = [sum(1 for v in g if v > med) + (sum(1 for v in g if v == med) if ties == "above" else 0) for g in gs]
    below = [sum(1 for v in g if v < med) + (sum(1 for v in g if v == med) if ties == "below" else 0) for g in gs]
    table = [above, below]
    out = {"median": med, "table": table, "df": Fraction(k - 1)}
    R = [sum(above), sum(below)]
    C = [a + b for a, b in zip(above, below)]
    if 0 in R or 0 in C:
        return {**_nan_fields(), **out}
    T = sum(R)
    obs, exp = [], []
    corr = p.kwargs.get("correction", True) and k == 2
    for i in range(2):
        for j in range(k):
            E = Fraction(R[i] * C[j], T)
            d = table[i][j] - E
            if corr:
                d = Fraction(0) if abs(d) <= Fraction(1, 2) else d - Fraction(sign(d), 2)
            obs.append(E + d)
            exp.append(E)
    lam = p.kwargs.get("lambda_", 1)
    lam = LAMBDA[lam] if isinstance(lam, str) else fr(lam)
    S = _cressie_read(obs, exp, lam)
    return {
        **out,
        "statistic": S,
        "pvalue": lambda: chi2_sf(S() if callable(S) else S, k - 1),
        "kappa": kappa(chi2_p_arb, S, (k - 1,)),
    }


def page_row_counts(row):
    """{sum: count} of 2 sum_j j r_pi(j) over the n! orders of one row's values (doubled, ints)."""
    from itertools import permutations

    out = {}
    for pm in permutations(row):
        s = sum((j + 1) * v for j, v in enumerate(pm))
        out[s] = out.get(s, 0) + 1
    return out


@family("page_trend_test", ("page_trend_test",), seconds=600)
def page_trend_test(p):
    """Page's L from exact within-row midranks (or the given ranks) and the predicted ranks. Exact:
    each row's distribution over the permutations of its doubled values (conditional with ties,
    H45), convolved, P(L' >= L); asymptotic: scipy's normal (no tie correction)."""
    data = np.asarray(p.args[0], dtype=np.float64)
    m, n = data.shape
    ranked = p.kwargs.get("ranked", False)
    pred = p.kwargs.get("predicted_ranks")
    pred = list(range(1, n + 1)) if pred is None else [int(v) for v in pred]
    rows = [fvec(r) if ranked else midranks(fvec(r))[0] for r in data]
    Cs = [sum(rw[j] for rw in rows) for j in range(n)]
    L = sum(pj * c for pj, c in zip(pred, Cs))
    method = p.kwargs.get("method", "auto")
    if method == "auto":
        method = "asymptotic" if n > 8 or (m > 12 and n > 3) or m > 20 else "exact"
    if method == "exact":
        dist = {0: 1}
        cache = {}
        for rw in rows:
            key = tuple(sorted(int(2 * v) for v in rw))
            if key not in cache:
                cache[key] = page_row_counts(key)
            nd = {}
            for a, ca in dist.items():
                for b, cb in cache[key].items():
                    nd[a + b] = nd.get(a + b, 0) + ca * cb
            dist = nd
        _, ge = tails(dist, int(2 * L))
        return {"statistic": L, "pvalue": ge, "method": "exact"}
    E0 = Fraction(m * n * (n + 1) ** 2, 4)
    V0 = Fraction(m * n**2 * (n + 1) * (n**2 - 1), 144)
    out = _z_fields(L - E0, V0, "greater", zfield=None, extra={"statistic": L, "method": "asymptotic"})
    return out


def _tie_groups(values):
    """(group sizes, group values) of the sorted values."""
    s = sorted(values)
    sizes, vals = [], []
    for v in s:
        if vals and vals[-1] == v:
            sizes[-1] += 1
        else:
            sizes.append(1)
            vals.append(v)
    return sizes, vals


def ks2_paths(n1, n2, a, b, h, kind):
    """The number of lattice paths (0, 0) -> (n1, n2) on which i a - j b reaches h ('greater'),
    j b - i a reaches h ('less') or |i a - j b| reaches h ('two-sided'): paths that stay inside,
    counted, subtracted from the total."""

    def inside(i, j):
        g = i * a - j * b
        if kind == "greater":
            return g < h
        if kind == "less":
            return -g < h
        return abs(g) < h

    row = [0] * (n2 + 1)
    for i in range(n1 + 1):
        new = [0] * (n2 + 1)
        for j in range(n2 + 1):
            if not inside(i, j):
                continue
            if i == 0 and j == 0:
                new[j] = 1
            else:
                new[j] = (row[j] if i > 0 else 0) + (new[j - 1] if j > 0 else 0)
        row = new
    return math.comb(n1 + n2, n1) - row[n2]


def pooled_blocks(x, y):
    """The distinct values of the pooled rationals in order, each with (count in x, count in y)."""
    cnt = {}
    for v in x:
        cnt.setdefault(v, [0, 0])[0] += 1
    for v in y:
        cnt.setdefault(v, [0, 0])[1] += 1
    return [(v, cnt[v][0], cnt[v][1]) for v in sorted(cnt)]


def ks2_tail_count(n1, n2, blocks, h, kind):
    """The number of arrangements of the labels (n1 x's, n2 y's) over the pooled positions, with
    the blocks of tied values as observed (each block takes k of x's values in C(t, k) ways), on
    which the gap g = i a - j b at some block's end reaches h ('greater'), -h ('less') or either
    ('two-sided'): a count over the blocks in exact integers, a state that reaches the boundary
    counted once with all its completions (C(positions left, x's left)). Without ties (blocks of
    one) it is ks2_paths'."""
    L = math.lcm(n1, n2)
    a, b = L // n1, L // n2
    N = n1 + n2

    def out(g):
        if kind == "greater":
            return g >= h
        if kind == "less":
            return -g >= h
        return abs(g) >= h

    states, p, absorbed = {0: 1}, 0, 0
    for t in blocks:
        new = {}
        for i, cnt in states.items():
            j = p - i
            for k in range(max(0, t - (n2 - j)), min(t, n1 - i) + 1):
                i2 = i + k
                c = cnt * math.comb(t, k)
                if out(i2 * a - (p + t - i2) * b):
                    absorbed += c * math.comb(N - p - t, n1 - i2)
                else:
                    new[i2] = new.get(i2, 0) + c
        states, p = new, p + t
    return absorbed


def _ks2_stat(x, y, alt):
    """(h, L, location, sign) of scipy's two-sample statistic: the gaps at the distinct values,
    the first value of the extreme, D- when it is larger than D+ (two-sided)."""
    n1, n2 = len(x), len(y)
    L = math.lcm(n1, n2)
    a, b = L // n1, L // n2
    i = j = 0
    gmax = gmin = None
    for v, cx, cy in pooled_blocks(x, y):
        i, j = i + cx, j + cy
        g = i * a - j * b
        if gmax is None or g > gmax[0]:
            gmax = (g, v)
        if gmin is None or g < gmin[0]:
            gmin = (g, v)
    if alt == "greater" or (alt == "two-sided" and not -gmin[0] > gmax[0]):
        return gmax[0], L, gmax[1], 1
    return -gmin[0], L, gmin[1], -1


@family("ks_2samp", ("ks_2samp",), seconds=900)
def ks_2samp(p):
    """Two-sample Kolmogorov-Smirnov: D = h / lcm(n1, n2) from the ECDFs at the distinct values
    (as scipy computes it), its location and sign. method 'auto' (exact up to max(n1, n2) = 10000)
    and 'exact': the share of the label arrangements whose gap reaches h, with ties conditional on
    the observed blocks (H55: R's psmirnov, not scipy's distribution without ties); 'asymp':
    scipy's, the one-sample exact distribution at round(n1 n2 / (n1 + n2)) two-sided (the
    Marsaglia-Tsang-Wang matrix in Arb) and Hodges' formula one-sided. Empty samples are NaN."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    n1, n2 = len(x), len(y)
    alt = p.kwargs.get("alternative", "two-sided")
    method = p.kwargs.get("method", "auto")
    used = "exact" if method == "exact" or (method == "auto" and max(n1, n2) <= 10000) else "asymp"
    if n1 == 0 or n2 == 0:
        return {"statistic": NAN, "pvalue": NAN, "method": used}
    h, L, loc, sgn = _ks2_stat(x, y, alt)
    out = {"statistic": Fraction(h, L), "statistic_location": loc, "statistic_sign": sgn, "method": used}
    if used == "exact":
        blocks = [cx + cy for _, cx, cy in pooled_blocks(x, y)]
        out["pvalue"] = (
            Fraction(1) if h == 0 else Fraction(ks2_tail_count(n1, n2, blocks, h, alt), math.comb(n1 + n2, n1))
        )
        return out
    m, n = max(n1, n2), min(n1, n2)
    en_double = float(m) * float(n) / (float(m) + float(n))
    nr = int(np.round(en_double))
    en = Fraction(m * n, m + n)

    def pfun(d):
        if alt == "two-sided":
            if d > arb(0.5):
                return 2 * smirnov_sf(nr, d)
            return 1 - kolmogorov_cdf_exact(nr, d)
        z = A(en).sqrt() * d
        v = (-2 * z * z - 2 * z * (m + 2 * n) / A(m * n * (m + n)).sqrt() / 3).exp()
        return v if v < 1 else arb(1)

    out["pvalue"] = lambda: pfun(A(Fraction(h, L)))
    out["kappa"] = kappa(pfun, Fraction(h, L))
    return out


def cvm2_u4(x, y):
    """4U of the two-sample Cramér-von Mises statistic from the exact midranks: nx sum (2r - 2i)^2 +
    ny sum (2s - 2j)^2, an integer; and the blocks' sizes."""
    nx, ny = len(x), len(y)
    u4, i, j, pos, blocks = 0, 0, 0, 0, []
    for _, cx, cy in pooled_blocks(x, y):
        t = cx + cy
        twoR = 2 * pos + t + 1
        u4 += nx * sum((twoR - 2 * (i + m)) ** 2 for m in range(1, cx + 1))
        u4 += ny * sum((twoR - 2 * (j + m)) ** 2 for m in range(1, cy + 1))
        i, j, pos = i + cx, j + cy, pos + t
        blocks.append((t, cx))
    return u4, blocks


def cvm2_tail_count(nx, ny, blocks, u4):
    """The number of label arrangements (blocks of ties as observed, C(t, k) ways for k x's in a
    block) whose 4U is at least u4: exact integers over the blocks, the partial sums of a state
    kept in a dictionary; a partial sum that reaches u4 is counted with all its completions at
    once (the increments are never negative)."""
    N = nx + ny
    states = {0: {0: 1}}
    pos, absorbed = 0, 0
    for t, _ in blocks:
        twoR = 2 * pos + t + 1
        new = {}
        for i, sums in states.items():
            j = pos - i
            for k in range(max(0, t - (ny - j)), min(t, nx - i) + 1):
                inc = nx * sum((twoR - 2 * (i + m)) ** 2 for m in range(1, k + 1))
                inc += ny * sum((twoR - 2 * (j + m)) ** 2 for m in range(1, t - k + 1))
                mult = math.comb(t, k)
                rest = math.comb(N - pos - t, nx - i - k)
                tgt = new.setdefault(i + k, {})
                for v, c in sums.items():
                    w = v + inc
                    if w >= u4:
                        absorbed += c * mult * rest
                    else:
                        tgt[w] = tgt.get(w, 0) + c * mult
        states, pos = new, pos + t
    return absorbed


@family("cramervonmises_2samp", ("cramervonmises_2samp",), seconds=1800)
def cramervonmises_2samp(p):
    """The two-sample Cramér-von Mises T = U/(k N) - (4k - 1)/(6N) from exact midranks (exact).
    method 'auto' (exact up to max(nx, ny) = 20) and 'exact': the share of the label arrangements
    with U at least the observed (ties conditional on the observed blocks, as the rank tests'
    H45); 'asymptotic': scipy's tn = 1/6 + (T - E T)/sqrt(45 Var T) (Anderson 1962) and
    max(0, 1 - V(tn)) with Csörgő and Faraway's V in Arb (1 for tn < 0.003). n < 2 is NaN."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    nx, ny = len(x), len(y)
    method = p.kwargs.get("method", "auto")
    used = "exact" if method == "exact" or (method == "auto" and max(nx, ny) <= 20) else "asymptotic"
    if nx < 2 or ny < 2:
        return {"statistic": NAN, "pvalue": NAN, "method": used}
    u4, blocks = cvm2_u4(x, y)
    k, N = nx * ny, nx + ny
    T = Fraction(6 * u4 - 4 * k * (4 * k - 1), 24 * k * N)
    out = {"statistic": T, "method": used}
    if used == "exact":
        out["pvalue"] = Fraction(cvm2_tail_count(nx, ny, blocks, u4), math.comb(N, nx))
        return out
    et = Fraction(N + 1, 6 * N)
    vt = Fraction((N + 1) * (4 * k * N - 3 * (nx * nx + ny * ny) - 2 * k), 45 * N * N * 4 * k)

    def pfun(t):
        tn = A(Fraction(1, 6)) + (t - A(et)) / (45 * A(vt)).sqrt()
        if tn < A(Fraction(3, 1000)):
            return arb(1)
        if not tn > A(Fraction(3, 1000)):
            raise NotSettled("tn at 0.003")
        v = 1 - cvm_cdf_inf(tn)
        if v < 0:
            return arb(0)
        return v

    out["pvalue"] = lambda: pfun(A(T))
    out["kappa"] = kappa(pfun, T)
    return out


AD_B0 = (0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085)
AD_B1 = (-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615)
AD_B2 = (-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154)
AD_SIG = (0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001)


def ad_ksamp_a2kn(samples, variant):
    """Scholz and Stephens' A2kN (eq. 7 'midrank', 6 'right', 3 'continuous') of rationals, exact."""
    Z = sorted(v for s in samples for v in s)
    N = len(Z)
    zs, ls = [], []
    for v in Z:
        if zs and zs[-1] == v:
            ls[-1] += 1
        else:
            zs.append(v)
            ls.append(1)
    tot = Fraction(0)
    for s in samples:
        s = sorted(s)
        n = len(s)
        acc = Fraction(0)
        if variant == "continuous":
            for jj in range(1, N):
                M = bisect.bisect_right(s, Z[jj - 1])
                acc += Fraction((N * M - jj * n) ** 2, jj * (N - jj))
            tot += acc / n
            continue
        left = 0
        for jj, (z, lj) in enumerate(zip(zs, ls)):
            le = bisect.bisect_right(s, z)
            lt = bisect.bisect_left(s, z)
            if variant == "midrank":
                M = Fraction(le + lt, 2)
                B = left + Fraction(lj, 2)
                acc += Fraction(lj, N) * (N * M - B * n) ** 2 / (B * (N - B) - Fraction(N * lj, 4))
            elif jj + 1 < len(zs):
                B = left + lj
                acc += Fraction(lj, N) * Fraction((N * le - B * n) ** 2, B * (N - B))
            left += lj
        tot += acc / n
    if variant == "midrank":
        tot *= Fraction(N - 1, N)
    if variant == "continuous":
        tot /= N
    return tot, N, len(zs)


def _ad_crit(k):
    """scipy's critical values b0 + b1/sqrt(m) + b2/m in doubles (numpy's operations), exact."""
    m = k - 1
    sm = math.sqrt(m)
    return [Fraction(b0 + b1 / sm + b2 / m) for b0, b1, b2 in zip(AD_B0, AD_B1, AD_B2)]


def ad_ksamp_p(a2, k):
    """scipy's p-value of the standardized statistic (an arb): 0.25 below the smallest critical
    value, 0.001 above the largest, else exp of np.polyfit's least-squares quadratic of log(level)
    on the critical values (the normal equations solved exactly with the doubles numpy forms)."""
    crit = _ad_crit(k)
    lo, hi = min(crit), max(crit)
    if a2 < A(lo):
        return A(Fraction(0.25))
    if a2 > A(hi):
        return A(Fraction(0.001))
    if not (a2 > A(lo) and a2 < A(hi)):
        raise NotSettled("the statistic on the table's end")
    S = [sum(c**e for c in crit) for e in range(5)]
    Y = [sum(A(c**e) * A(Fraction(sg)).log() for c, sg in zip(crit, AD_SIG)) for e in range(3)]
    M = arb_mat([[A(S[i + j]) for j in range(3)] for i in range(3)])
    coef = M.solve(arb_mat([[Y[0]], [Y[1]], [Y[2]]]))
    return (coef[0, 0] + coef[1, 0] * a2 + coef[2, 0] * a2 * a2).exp()


@family("anderson_ksamp", ("anderson_ksamp",), seconds=1800)
def anderson_ksamp(p):
    """The k-sample Anderson-Darling statistic: A2kN exact, sigma_N^2 from H (exact), h and g
    (Arb sums), A2 = (A2kN - (k - 1)) / sigma_N; the p-value scipy's table interpolation
    (ad_ksamp_p). An empty sample, a single distinct value or N < 4 is NaN."""
    samples = [fvec(s) for s in p.args[0]]
    variant = p.kwargs.get("variant", "midrank")
    k = len(samples)
    pooled = [v for s in samples for v in s]
    if any(len(s) == 0 for s in samples) or len(set(pooled)) < 2 or len(pooled) < 4:
        return _nan_fields()
    a2kn, N, _ = ad_ksamp_a2kn(samples, variant)
    H = sum(Fraction(1, len(s)) for s in samples)

    def stat():
        h = sum((arb(1) / i for i in range(1, N)), arb(0))
        suffix, run = {}, arb(0)
        for j in range(N - 1, 0, -1):
            run += arb(1) / j
            suffix[j] = run
        g = sum((suffix[i + 1] / (N - i) for i in range(1, N - 1)), arb(0))
        Ha = A(H)
        a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * Ha
        b = (2 * g - 4) * k**2 + 8 * h * k + (2 * g - 14 * h - 4) * Ha - 8 * h + 4 * g - 6
        c = (6 * h + 2 * g - 2) * k**2 + (4 * h - 4 * g + 6) * k + (2 * h - 6) * Ha + 4 * h
        d = (2 * h + 6) * k**2 - 4 * h * k
        sig2 = (a * N**3 + b * N**2 + c * N + d) / ((N - 1) * (N - 2) * (N - 3))
        return (A(a2kn) - (k - 1)) / sig2.sqrt()

    return {
        "statistic": stat,
        "pvalue": lambda: ad_ksamp_p(stat(), k),
        "kappa": kappa(lambda s: ad_ksamp_p(s, k), stat),
    }


def _percentile_linear(zs, q):
    """numpy's linear percentile (q a Fraction in [0, 1]) of sorted rationals, exact."""
    idx = q * (len(zs) - 1)
    lo = math.floor(idx)
    frac = idx - lo
    if frac == 0:
        return zs[lo]
    return zs[lo] + frac * (zs[lo + 1] - zs[lo])


@family("epps_singleton_2samp", ("epps_singleton_2samp",), lambda p: "rank-deficient" not in p.id, seconds=1800)
def epps_singleton_2samp(p):
    """The Epps-Singleton W: sigma = iqr/2 of the pooled sample (numpy's linear percentiles, exact),
    g = (cos(t x / sigma), sin(t x / sigma)) in Arb, the biased covariances and their combination
    est = (N/nx) cov_x + (N/ny) cov_y, W = N d' est^-1 d (full rank: the problems here), times
    the small-sample correction below 25 values; chi^2 (2 len(t)), df = 2 len(t) (H59). n < 5 is NaN."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    nx, ny = len(x), len(y)
    t = [fr(v) for v in np.ravel(np.asarray(p.kwargs.get("t", (0.4, 0.8)), dtype=np.float64))]
    if nx < 5 or ny < 5:
        return _nan_fields(df=NAN)
    zs = sorted(x + y)
    sigma = (_percentile_linear(zs, Fraction(3, 4)) - _percentile_linear(zs, Fraction(1, 4))) / 2
    N = nx + ny
    T = len(t)

    def moments(vals):
        g = []
        for v in vals:
            args = [A(tk / sigma * v) for tk in t]
            g.append([a.cos() for a in args] + [a.sin() for a in args])
        n = len(vals)
        mean = [sum((row[i] for row in g), arb(0)) / n for i in range(2 * T)]
        cov = [[sum(((row[i] - mean[i]) * (row[j] - mean[j]) for row in g), arb(0)) / n for j in range(2 * T)]
               for i in range(2 * T)]  # fmt: skip
        return mean, cov

    def stat():
        mx, cx = moments(x)
        my, cy = moments(y)
        est = arb_mat([[A(Fraction(N, nx)) * cx[i][j] + A(Fraction(N, ny)) * cy[i][j] for j in range(2 * T)]
                       for i in range(2 * T)])  # fmt: skip
        d = arb_mat([[mx[i] - my[i]] for i in range(2 * T)])
        sol = est.solve(d)
        w = N * sum((d[i, 0] * sol[i, 0] for i in range(2 * T)), arb(0))
        if max(nx, ny) < 25:
            w = w / (1 + A(N) ** A(fr(-0.45)) + A(fr(10.1)) * (A(nx) ** A(fr(-1.7)) + A(ny) ** A(fr(-1.7))))
        return w

    return {
        "statistic": stat,
        "pvalue": lambda: chi2_p_arb(stat(), arb(2 * T)),
        "df": Fraction(2 * T),
        "kappa": kappa(lambda w: chi2_p_arb(w, arb(2 * T)), stat),
    }


def mahonian(n, top):
    """The number of permutations of n with k inversions, k = 0..top."""
    c = [1] + [0] * top
    for j in range(2, n + 1):
        acc = [0] * (top + 1)
        run = 0
        for k in range(top + 1):
            run += c[k]
            if k - j >= 0:
                run -= c[k - j]
            acc[k] = run
        c = acc
    return c


@family("kendalltau", ("kendalltau",), seconds=300)
def kendalltau(p):
    """Kendall's tau-b (tau-c): pair counts exact; the exact p-value from the Mahonian numbers
    (scipy's _kendall_p_exact), the normal one with scipy's tie-corrected variance."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    n = len(x)
    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = sign(x[i] - x[j]) * sign(y[i] - y[j])
            con += s > 0
            dis += s < 0
    tot = n * (n - 1) // 2
    sx, vx = _tie_groups(x)
    sy, vy = _tie_groups(y)
    xtie = sum(t * (t - 1) // 2 for t in sx)
    ytie = sum(t * (t - 1) // 2 for t in sy)
    cmd = con - dis
    alt = p.kwargs.get("alternative", "two-sided")
    if n < 2 or xtie == tot or ytie == tot:  # a constant sample: NaN (scipy's); ties: asymptotic
        return {"statistic": NAN, "pvalue": NAN, "method": "asymptotic" if xtie or ytie else "exact"}
    if p.kwargs.get("variant", "b") == "b":
        tau2 = Fraction(cmd * cmd, (tot - xtie) * (tot - ytie))
        stat = (lambda: sign(cmd) * A(tau2).sqrt()) if cmd else Fraction(0)
    else:
        mc = min(len(vx), len(vy))
        stat = Fraction(2 * cmd) / (Fraction(n * n * (mc - 1), mc))
    method = p.kwargs.get("method", "auto")
    if method == "auto":
        method = "exact" if xtie == 0 and ytie == 0 and (n <= 33 or min(dis, tot - dis) <= 1) else "asymptotic"
    if method == "exact":
        c = tot - dis
        right = c >= tot - c
        cm = min(c, tot - c)
        counts = mahonian(n, cm)
        fac = math.factorial(n)
        prob = Fraction(2 * sum(counts), fac)
        mass = Fraction(counts[cm], fac)
        if alt != "two-sided":
            prob = prob / 2 if right == (alt == "greater") else 1 - prob / 2 + mass
        return {"statistic": stat, "pvalue": min(Fraction(1), prob), "method": "exact"}
    x0 = sum(t * (t - 1) * (t - 2) for t in sx)
    y0 = sum(t * (t - 1) * (t - 2) for t in sy)
    x1 = sum(t * (t - 1) * (2 * t + 5) for t in sx)
    y1 = sum(t * (t - 1) * (2 * t + 5) for t in sy)
    m = Fraction(n * (n - 1))
    var = (m * (2 * n + 5) - x1 - y1) / 18 + (2 * xtie * ytie) / m
    if n > 2:
        var += x0 * y0 / (9 * m * (n - 2))
    return {
        "statistic": stat,
        "pvalue": lambda: norm_p(A(cmd) / A(var).sqrt(), alt),
        "method": "asymptotic",
        "kappa": kappa(lambda zz: norm_p(zz, alt), lambda: A(cmd) / A(var).sqrt()) if cmd else NAN,
    }


def _r2(x, y):
    """(r^2, the sign of r): exact; None for a constant sample."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None, 0
    return sxy * sxy / (sxx * syy), sign(sxy)


def _corr_result(r2, s, n, alt, ci=False):
    """Pearson's r = s sqrt(r2) and scipy's p-value of it (the beta distribution of r: two-sided
    I_q((n - 2)/2, 1/2), q = 1 - r^2 exact), kappa the p-value's condition number in q (sd forms q
    without cancellation, H5), and the Fisher z interval at alpha = 0.05 (scipy's, n <= 3: (-1, 1)).
    n < 2 or a constant sample: NaN; n = 2: r = +-1 and p = 1 (scipy's)."""
    if r2 is None or n < 2:
        out = {"statistic": NAN, "pvalue": NAN}
        if ci:
            out["conf_int"] = [NAN, NAN]
        return out
    nu = Fraction(n - 2)
    q = 1 - r2

    def pq(qq):
        two = beta_reg(qq, nu / 2, Fraction(1, 2))
        if alt == "two-sided":
            return two
        upper = s > 0 if alt == "greater" else s < 0
        return two / 2 if upper else 1 - two / 2

    if n == 2:
        out = {"statistic": Fraction(s), "pvalue": Fraction(1)}
    else:
        out = {"statistic": (lambda: s * A(r2).sqrt()) if s else Fraction(0), "pvalue": lambda: pq(q)}
        out["kappa"] = kappa(pq, q) if 0 < q < 1 else NAN
    if ci:
        if n <= 3:
            out["conf_int"] = [Fraction(-1), Fraction(1)]
            return out
        h = (lambda: norm_isf(ALPHA / 2)) if alt == "two-sided" else (lambda: norm_isf(ALPHA))
        se = Fraction(1, n - 3)
        if q == 0:
            end = Fraction(s)  # z = +-inf: tanh gives the end itself
            lo, hi = (end if alt != "less" else Fraction(-1)), (end if alt != "greater" else Fraction(1))
        else:

            def z():
                r = s * A(r2).sqrt()
                return ((1 + r) / (1 - r)).log() / 2

            lo = (lambda: (z() - h() * A(se).sqrt()).tanh()) if alt != "less" else Fraction(-1)
            hi = (lambda: (z() + h() * A(se).sqrt()).tanh()) if alt != "greater" else Fraction(1)
        out["conf_int"] = [lo, hi]
    return out


@family("pearsonr", ("pearsonr", "pointbiserialr"))
def pearsonr(p):
    """Pearson's r with r^2 exact (so 1 - r^2 has no cancellation), the p-value
    I_{1 - r^2}((n - 2)/2, 1/2) (scipy's beta distribution of r), the Fisher z interval."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    r2, s = _r2(x, y)
    return _corr_result(r2, s, len(x), p.kwargs.get("alternative", "two-sided"), ci=p.func == "pearsonr")


@family("spearmanrho", ("spearmanrho", "spearmanr"))
def spearmanrho(p):
    """Spearman's rho: Pearson's r of the exact midranks and the t approximation (n - 2); +-inf
    take their places in the order."""
    x, y = ([Fraction(v) if math.isfinite(v) else v for v in a] for a in p.args[:2])
    r2, s = _r2(midranks(x)[0], midranks(y)[0])
    return _corr_result(r2, s, len(x), p.kwargs.get("alternative", "two-sided"))


def _weights_of_ranks(rank):
    """The hyperbolic weigher (scipy's default): 1/(r + 1) of each element's rank."""
    return [Fraction(1, r + 1) for r in rank]


def _lex_rank(a, b):
    """The decreasing lexicographic rank by (a, b) (the largest gets 0; equal pairs by position,
    scipy's stable lexsort)."""
    n = len(a)
    order = sorted(range(n), key=lambda i: (a[i], b[i], i))
    rank = [0] * n
    for k in range(n):
        rank[order[n - 1 - k]] = k
    return rank


def _wtau_parts(x, y, w, additive):
    """Vigna's weighted tau by its definition, every pair: (sum w_ij sgn dx sgn dy, the product of
    the two sums of w_ij over the pairs not tied in x and in y), w_ij = w_i + w_j (additive) or
    w_i w_j; None where one of the sums is 0 (a constant sample)."""
    n = len(x)
    # (the weights scaled to integers by their common denominator: the ratio does not change)
    L = math.lcm(*[f.denominator for f in w]) if w else 1
    w = [int(f * L) for f in w]
    num = dx = dy = 0
    for i in range(n):
        for j in range(i + 1, n):
            wij = w[i] + w[j] if additive else w[i] * w[j]
            sx, sy = sign(x[i] - x[j]), sign(y[i] - y[j])
            num += wij * sx * sy
            if sx:
                dx += wij
            if sy:
                dy += wij
    return None if dx == 0 or dy == 0 else (num, dx * dy)


@family("weightedtau", ("weightedtau",), seconds=600)
def weightedtau(p):
    """The weighted tau by its pairwise definition with exact rational weights (scipy's options:
    rank True, None, False or an array taken by its order; the hyperbolic weigher; additive)."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    n = len(x)
    rank = p.kwargs.get("rank", True)
    add = p.kwargs.get("additive", True)
    if isinstance(rank, np.ndarray):
        levels = sorted(set(rank.tolist()))
        ranks = [levels.index(v) for v in rank.tolist()]
        parts = [_wtau_parts(x, y, _weights_of_ranks(ranks), add)]
    elif rank is False:
        parts = [_wtau_parts(x, y, _weights_of_ranks(range(n)), add)]
    else:
        parts = [_wtau_parts(x, y, _weights_of_ranks(_lex_rank(x, y)), add)]
        if rank is True:
            parts.append(_wtau_parts(y, x, _weights_of_ranks(_lex_rank(y, x)), add))
    if n == 0 or any(q is None for q in parts):
        return {"statistic": NAN, "pvalue": NAN}

    def tau():
        ts = [A(num) / A(den).sqrt() for num, den in parts]
        return sum(ts) / len(ts)

    return {"statistic": tau, "pvalue": NAN}


@family("somersd", ("somersd",))
def somersd(p):
    """Somers' D(Y|X) with every pair counted: P - Q (twice the concordant minus discordant
    pairs), the row sums, T = sum over the observations of (concordant - discordant with it)^2,
    exact; the p-value of Z = (P - Q) / sqrt(4 S), S = T - (P - Q)^2 / N, in Arb (scipy's)."""
    a = p.args[0]
    if np.ndim(a) == 2:
        t = np.asarray(a, dtype=np.int64)
        xs, ys = [], []
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                xs += [Fraction(i)] * int(t[i, j])
                ys += [Fraction(j)] * int(t[i, j])
        table = t.tolist()
    else:
        xs, ys = fvec(a), fvec(p.args[1])
        ux, uy = sorted(set(xs)), sorted(set(ys))
        table = [[0] * len(uy) for _ in ux]
        for u, v in zip(xs, ys):
            table[ux.index(u)][uy.index(v)] += 1
    N = len(xs)
    if len(table) <= 1 or len(table[0]) <= 1:
        return {"statistic": NAN, "pvalue": NAN, "table": table}
    c = [0] * N
    for k in range(N):
        for m in range(N):
            c[k] += sign(xs[m] - xs[k]) * sign(ys[m] - ys[k])
    pq = sum(c)
    rows2 = sum(sum(r) ** 2 for r in table)
    d = Fraction(pq, N * N - rows2)
    S = Fraction(sum(v * v for v in c)) - Fraction(pq * pq, N)
    alt = p.kwargs.get("alternative", "two-sided")
    out = {"statistic": d, "table": table}
    if S <= 0:
        out["pvalue"] = NAN
        return out

    def z():
        return A(pq) / (4 * A(S)).sqrt()

    out["pvalue"] = lambda: norm_p(z(), alt)
    out["kappa"] = kappa(lambda zz: norm_p(zz, alt), z) if pq else NAN
    return out


@family("chatterjeexi", ("chatterjeexi",))
def chatterjeexi(p):
    """Chatterjee's xi (exact; x's ties broken by position, as sd) and the normal p-value
    P(Z >= xi / sd), the null variance by scipy's _xi_std in exact rationals (Theorem 2.1's 2/5 for
    a continuous y)."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    n = len(x)
    cont = p.kwargs.get("y_continuous", False)
    order = sorted(range(n), key=lambda i: (x[i], i))
    ys = [y[i] for i in order]
    r = [sum(1 for v in ys if v <= t) for t in ys]
    l = [sum(1 for v in ys if v >= t) for t in ys]  # noqa: E741
    num = sum(abs(r[i + 1] - r[i]) for i in range(n - 1))
    if cont:
        xi = 1 - Fraction(3 * num, n * n - 1)
        var = Fraction(2, 5) / n
    else:
        den = 2 * sum((n - v) * v for v in l)
        if den == 0:
            return {"statistic": NAN, "pvalue": NAN}
        xi = 1 - Fraction(n * num, den)
        u = sorted(r)
        v, acc = [], 0
        for e in u:
            acc += e
            v.append(acc)
        an = Fraction(sum((2 * n - 2 * i + 1) * u[i - 1] ** 2 for i in range(1, n + 1)), n**4)
        bn = Fraction(sum((v[i - 1] + (n - i) * u[i - 1]) ** 2 for i in range(1, n + 1)), n**5)
        cn = Fraction(sum((2 * n - 2 * i + 1) * u[i - 1] for i in range(1, n + 1)), n**3)
        dn = Fraction(sum(e * (n - e) for e in l), n**3)
        var = (an - 2 * bn + cn**2) / dn**2 / n

    def z():
        return A(xi) / A(var).sqrt()

    return {
        "statistic": xi,
        "pvalue": lambda: norm_sf(z()),
        "kappa": kappa(lambda zz: norm_sf(zz), z) if xi else NAN,
    }


def _hyper_pmf(x, r1, c1, N):
    """P(X = x), X the top-left cell of a 2 x 2 table with row sum r1, column sum c1, total N."""
    return Fraction(math.comb(c1, x) * math.comb(N - c1, r1 - x), math.comb(N, r1))


def _tables(rows, cols):
    """Every table of non-negative integers with these margins."""
    if len(rows) == 1:
        yield [list(cols)]
        return
    r0, rest = rows[0], rows[1:]

    def first(cols_left, i, left):
        if i == len(cols_left) - 1:
            if left <= cols_left[i]:
                yield [left]
            return
        for v in range(min(left, cols_left[i]) + 1):
            for tail in first(cols_left, i + 1, left - v):
                yield [v] + tail

    for row in first(list(cols), 0, r0):
        for t in _tables(rest, [c - v for c, v in zip(cols, row)]):
            yield [row] + t


def _rxc_tail(t, rerr):
    """The r x c Fisher p-value of the table t (lists of ints): sum over every table with its
    margins of the probability, the tables at most rerr times as probable as t. Exact: the columns
    filled one at a time, the state the sorted remaining row sums with {prod x!: number of paths}
    (integers), the probability K / prod x!, K = prod R! prod C! / N!."""
    rows = [sum(r) for r in t]
    cols = [sum(c) for c in zip(*t)]
    N = sum(rows)
    fact = [1] * (N + 1)
    for k in range(2, N + 1):
        fact[k] = fact[k - 1] * k
    K = Fraction(math.prod(fact[r] for r in rows) * math.prod(fact[c] for c in cols), fact[N])
    d_obs = math.prod(fact[v] for r in t for v in r)

    def splits(key, total):
        def rec(i, left):
            if i == len(key) - 1:
                if left <= key[i]:
                    yield (left,)
                return
            for v in range(min(key[i], left) + 1):
                for tail in rec(i + 1, left - v):
                    yield (v,) + tail

        yield from rec(0, total)

    stage = {tuple(sorted(rows, reverse=True)): {1: 1}}
    for c in cols:
        nxt = {}
        for key, ds in stage.items():
            for x in splits(key, c):
                child = tuple(sorted((k - v for k, v in zip(key, x)), reverse=True))
                w = math.prod(fact[v] for v in x)
                dst = nxt.setdefault(child, {})
                for d, m in ds.items():
                    dd = d * w
                    dst[dd] = dst.get(dd, 0) + m
        stage = nxt
    (ds,) = stage.values()
    # P = K / d <= rerr K / d_obs  <=>  d rerr >= d_obs
    tot = sum(Fraction(m, d) for d, m in ds.items() if d * rerr >= d_obs)
    return min(Fraction(1), K * tot), K / d_obs


@family("fisher_exact", ("fisher_exact",), seconds=600)
def fisher_exact(p):
    """Fisher's exact test, the null in exact rationals. 2 x 2 (the hypergeometric): scipy's
    two-sided rule -- p = 1 when the observed table is as probable as the mode to 1e-14; else the
    observed tail and the other side's tables less probable than (1 + 1e-14) times the observed
    one (at most as probable, below the mode). r x c (the multiple hypergeometric over every table
    with the margins, by an exact dynamic programme over the columns): R's fisher.test rule, the
    tables at most (1 + 1e-7) times as probable; the statistic the observed table's probability
    (scipy's). The 2 x 2 statistic is the sample odds ratio ad/bc. A zero margin of a 2 x 2 table:
    NaN and 1 (scipy's); an r x c table with one nonzero row or column: 1 and 1."""
    t = [[int(v) for v in row] for row in np.asarray(p.args[0])]
    rows = [sum(r) for r in t]
    cols = [sum(c) for c in zip(*t)]
    N = sum(rows)
    alt = p.kwargs.get("alternative", "two-sided")
    if len(t) == 2 and len(t[0]) == 2:
        (a, b), (c, d) = t
        if 0 in rows or 0 in cols:
            return {"statistic": NAN, "pvalue": Fraction(1)}
        stat = Fraction(a * d, b * c) if b * c else math.inf
        lo, hi = max(0, rows[0] - (N - cols[0])), min(rows[0], cols[0])
        pm = {x: _hyper_pmf(x, rows[0], cols[0], N) for x in range(lo, hi + 1)}
        if alt == "less":
            pv = sum(v for x, v in pm.items() if x <= a)
        elif alt == "greater":
            pv = sum(v for x, v in pm.items() if x >= a)
        else:
            eps = Fraction(1e-14)
            g = (1 + eps) * pm[a]
            mode = (cols[0] + 1) * (rows[0] + 1) // (N + 2)
            if abs(pm[a] - pm[mode]) <= eps * max(pm[a], pm[mode]):
                pv = Fraction(1)
            elif a < mode:
                pv = sum(v for x, v in pm.items() if x <= a) + sum(v for x, v in pm.items() if x >= mode and v < g)
            else:
                pv = sum(v for x, v in pm.items() if x >= a) + sum(v for x, v in pm.items() if x <= mode and v <= g)
        return {"statistic": stat, "pvalue": min(Fraction(1), pv)}
    if sum(1 for r in rows if r) <= 1 or sum(1 for c in cols if c) <= 1:
        return {"statistic": Fraction(1), "pvalue": Fraction(1), "rule": "R"}
    pv, stat = _rxc_tail(t, RERR)
    return {"statistic": stat, "pvalue": pv, "rule": "R"}


def _cressie_read(O, E, lam):
    """The Cressie-Read statistic of rationals O and E (E > 0): Pearson's exact (lam = 1), the
    others as an arb function; O = 0 contributes its limit (0 for lam > -1, inf for lam <= -1)."""
    if lam == 1:
        return sum((o - e) ** 2 / e for o, e in zip(O, E))
    if lam <= -1 and any(o == 0 for o in O):
        return math.inf

    def S():
        if lam == 0:
            return 2 * sum((A(o) * (A(o) / A(e)).log() if o else arb(0)) for o, e in zip(O, E))
        if lam == -1:
            return 2 * sum(A(e) * (A(e) / A(o)).log() for o, e in zip(O, E))
        L = A(lam)
        return 2 / (L * (L + 1)) * sum((A(o) * ((A(o) / A(e)) ** L - 1) if o else arb(0)) for o, e in zip(O, E))

    return S


@family("chi2_contingency", ("chi2_contingency",))
def chi2_contingency(p):
    """The chi^2 test of independence: exact expected counts R_i C_j / N of the rational counts
    (weighted tables too), Yates's correction (one degree of freedom: each O moved towards E by
    min(1/2, |O - E|)), the Cressie-Read statistic (Pearson's exact, the others in Arb), the chi^2
    tail; df = 0 gives 0 and 1, a zero margin NaN (H11; scipy raises)."""
    t = [fvec(r) for r in np.asarray(p.args[0])]
    rows = [sum(r) for r in t]
    cols = [sum(c) for c in zip(*t)]
    N = sum(rows)
    df = (len(rows) - 1) * (len(cols) - 1)
    E = [[r * c / N if N else Fraction(0) for c in cols] for r in rows]
    out = {"df": Fraction(df), "expected_freq": [[float(e) for e in r] for r in E]}
    if any(e == 0 for r in E for e in r):
        out.update(statistic=NAN, pvalue=NAN, kappa=NAN)
        return out
    if df == 0:
        out.update(statistic=Fraction(0), pvalue=Fraction(1), kappa=NAN)
        return out
    O = [list(r) for r in t]
    if df == 1 and p.kwargs.get("correction", True):
        for i in range(len(O)):
            for j in range(len(O[0])):
                diff = E[i][j] - O[i][j]
                O[i][j] += sign(diff) * min(Fraction(1, 2), abs(diff))
    lam = p.kwargs.get("lambda_")
    lam = Fraction(1) if lam is None else (LAMBDA[lam] if isinstance(lam, str) else fr(lam))
    S = _cressie_read([o for r in O for o in r], [e for r in E for e in r], lam)
    if S == math.inf:
        out.update(statistic=math.inf, pvalue=0.0, kappa=NAN)
        return out
    out.update(statistic=S, pvalue=lambda: chi2_sf(S() if callable(S) else S, df), kappa=kappa(chi2_p_arb, S, (df,)))
    return out


# ---- H6: Barnard, Boschloo, McNemar, Cochran's Q, the odds ratio, the relative risk


def _poly_sup(Acoef, N):
    """sup over [0, 1] of f(pi) = sum_s A_s pi^s (1 - pi)^(N - s) (A_s >= 0 integers) as a
    function giving a ball: f at 0 and 1 and at every real root in [0, 1] of f' (flint's
    certified root isolation at the current precision), the largest."""
    from flint import arb_poly, fmpz_poly

    co = [0] * (N + 1)
    for s, a in enumerate(Acoef):
        if a:
            for j in range(N - s + 1):
                co[s + j] += a * math.comb(N - s, j) * (-1 if j % 2 else 1)
    f = fmpz_poly(co)
    d = f.derivative()

    def sup():
        F = arb_poly(f)
        vals = [arb(Acoef[0]), arb(Acoef[N])]
        if d != 0:
            for r, _m in d.complex_roots():
                if not r.imag.contains(0):
                    continue
                x = r.real
                if x.upper() < 0 or x.lower() > 1:
                    continue
                vals.append(F(x))
        return amax(vals)

    return sup


def _wald2(x1, x2, n1, n2, pooled):
    """Barnard's T^2 (a Fraction; inf for a zero variance with p1 != p2) and the sign of T."""
    D = x1 * n2 - x2 * n1
    if D == 0:
        return Fraction(0), 0
    N, s = n1 + n2, x1 + x2
    if pooled:
        den = Fraction(n1 * n2 * s * (N - s), N)
    else:
        den = Fraction(x1 * (n1 - x1) * n2**3 + x2 * (n2 - x2) * n1**3, n1 * n2)
    return (math.inf if den == 0 else Fraction(D * D) / den), (1 if D > 0 else -1)


def _signed_ge(a, b):
    """T_a >= T_b for (T^2, sign) pairs."""
    (ta, sa), (tb, sb) = a, b
    if sa != sb:
        return sa > sb
    return ta >= tb if sa >= 0 else ta <= tb


@family("barnard_exact", ("barnard_exact",), seconds=600)
def barnard_exact(p):
    """Barnard's exact test: the region of the tables (x1, x2) with |T| >= |T_obs| (or T <= T_obs,
    T >= T_obs) by exact T^2 (rationals; scipy compares rounded doubles), and the supremum over
    pi of sum C(n1, x1) C(n2, x2) pi^s (1 - pi)^(N - s) at the critical points (flint's roots of
    the integer polynomial's derivative) and the ends. The columns are the samples (scipy's)."""
    t = [[int(v) for v in row] for row in np.asarray(p.args[0])]
    n1, n2 = t[0][0] + t[1][0], t[0][1] + t[1][1]
    if n1 == 0 or n2 == 0:
        return {"statistic": NAN, "pvalue": Fraction(1)}
    alt = p.kwargs.get("alternative", "two-sided")
    pooled = p.kwargs.get("pooled", True)
    obs = _wald2(t[0][0], t[0][1], n1, n2, pooled)
    N = n1 + n2
    Acoef = [0] * (N + 1)
    for x1 in range(n1 + 1):
        for x2 in range(n2 + 1):
            w = _wald2(x1, x2, n1, n2, pooled)
            if alt == "two-sided":
                inside = w[0] >= obs[0]
            elif alt == "less":
                inside = _signed_ge(obs, w)
            else:
                inside = _signed_ge(w, obs)
            if inside:
                Acoef[x1 + x2] += math.comb(n1, x1) * math.comb(n2, x2)
    sup = _poly_sup(Acoef, N)
    t2, sg = obs
    stat = (sg * math.inf) if t2 == math.inf else (Fraction(0) if sg == 0 else (lambda: sg * A(t2).sqrt()))
    return {"statistic": stat, "pvalue": sup}


def _boschloo_side(t, n1, n2, less):
    N = n1 + n2
    tol = Fraction(1.0 + 1e-13)

    def fisher(x1, s):  # P(X1 <= x1 | s) or P(X1 >= x1 | s)
        lo, hi = max(0, s - n2), min(n1, s)
        js = range(lo, x1 + 1) if less else range(x1, hi + 1)
        return Fraction(sum(math.comb(n1, j) * math.comb(n2, s - j) for j in js), math.comb(N, s))

    obs = fisher(t[0][0], t[0][0] + t[0][1])
    Acoef = [0] * (N + 1)
    for x1 in range(n1 + 1):
        for x2 in range(n2 + 1):
            if fisher(x1, x1 + x2) <= obs * tol:
                Acoef[x1 + x2] += math.comb(n1, x1) * math.comb(n2, x2)
    return obs, _poly_sup(Acoef, N)


@family("boschloo_exact", ("boschloo_exact",), seconds=600)
def boschloo_exact(p):
    """Boschloo's exact test: the one-sided Fisher p-value of every table (exact rationals), the
    region the tables at most (1 + 1e-13) times as significant as the observed one (scipy's
    tolerance, its double as a rational), the supremum as barnard_exact's. Two-sided: the side of
    the smaller supremum (exactly; a tie takes 'greater', scipy's strict '<'), doubled, at most 1."""
    t = [[int(v) for v in row] for row in np.asarray(p.args[0])]
    n1, n2 = t[0][0] + t[1][0], t[0][1] + t[1][1]
    if n1 == 0 or n2 == 0:
        return {"statistic": NAN, "pvalue": NAN}
    alt = p.kwargs.get("alternative", "two-sided")
    if alt != "two-sided":
        stat, sup = _boschloo_side(t, n1, n2, alt == "less")
        return {"statistic": stat, "pvalue": sup}
    (sl, pl), (sg, pg) = _boschloo_side(t, n1, n2, True), _boschloo_side(t, n1, n2, False)
    with ar.precision(600):
        vl, vg = pl(), pg()
        less = bool(vl < vg)
        if not less and not bool(vl >= vg):
            raise NotSettled("the two sides' suprema are not separated at 600 bits")

    def two():
        v = 2 * (pl() if less else pg())
        if v < 1:
            return v
        if v > 1:
            return arb(1)
        raise NotSettled("twice the supremum is not separated from 1")

    return {"statistic": sl if less else sg, "pvalue": two}


@family("mcnemar", ("mcnemar",))
def mcnemar(p):
    """McNemar's test (statsmodels'): exact -- min(b, c) and 2 P(B <= min(b, c)), B ~ Binomial(b + c,
    1/2), at most 1 (exact rationals); otherwise ((|b - c| - correction)^2 / (b + c)) exact and its
    chi^2(1) tail in Arb (b + c = 0: NaN, H11)."""
    t = np.asarray(p.args[0])
    b, c = int(t[0, 1]), int(t[1, 0])
    n = b + c
    if p.kwargs.get("exact", True):
        k = min(b, c)
        pv = min(Fraction(1), Fraction(2 * sum(math.comb(n, i) for i in range(k + 1)), 2**n))
        return {"statistic": Fraction(k), "pvalue": pv}
    if n == 0:
        return {"statistic": NAN, "pvalue": NAN, "kappa": NAN}
    stat = Fraction((abs(b - c) - (1 if p.kwargs.get("correction", True) else 0)) ** 2, n)
    return {"statistic": stat, "pvalue": lambda: chi2_sf(stat, 1), "kappa": kappa(chi2_p_arb, stat, (1,))}


@family("cochrans_q", ("cochrans_q",))
def cochrans_q(p):
    """Cochran's Q (statsmodels' SAS form) exact and its chi^2 (k - 1) tail in Arb; NaN when every
    block is all 0 or all 1."""
    x = [[int(v) for v in row] for row in np.asarray(p.args[0])]
    k = len(x[0])
    R = [sum(r) for r in x]
    C = [sum(c) for c in zip(*x)]
    T = sum(R)
    den = k * T - sum(r * r for r in R)
    if den == 0:
        return {"statistic": NAN, "pvalue": NAN, "df": Fraction(k - 1), "kappa": NAN}
    q = Fraction((k - 1) * (k * sum(c * c for c in C) - T * T), den)
    return {
        "statistic": q,
        "pvalue": lambda: chi2_sf(q, k - 1),
        "df": Fraction(k - 1),
        "kappa": kappa(chi2_p_arb, q, (k - 1,)),
    }


def _nchg_root(t, coef_of):
    """The positive root of sum_k coef_of(k) C_k w^(k - lo) (an integer polynomial after clearing
    denominators), C_k = C(n, k) C(M - n, N - k): a function giving its ball (flint's roots)."""
    from flint import fmpz_poly

    (a, b), (c, d) = t
    M, n, N = a + b + c + d, a + b, a + c
    lo, hi = max(0, n + N - M), min(n, N)
    cs = [coef_of(k) * math.comb(n, k) * math.comb(M - n, N - k) for k in range(lo, hi + 1)]
    den = math.lcm(*[Fraction(v).denominator for v in cs])
    poly = fmpz_poly([int(Fraction(v) * den) for v in cs])

    def root():
        pos = [r.real for r, _m in poly.complex_roots() if r.imag.contains(0) and r.real > 0]
        if len(pos) != 1:
            raise NotSettled(f"{len(pos)} positive roots")
        return pos[0]

    return root


@family("odds_ratio", ("odds_ratio",))
def odds_ratio(p):
    """The odds ratio (scipy.stats.contingency's). Conditional: the MLE of Fisher's noncentral
    hypergeometric, E X = x, and the exact interval P(X >= x) = alpha_t, P(X <= x) = alpha_t
    (alpha_t = 0.05 / 2 two-sided, 0.05 one-sided: conf_int_less, conf_int_greater), each the positive
    root of an integer polynomial (flint); 0 or inf at the ends of the support. Sample: ad / bc and
    Woolf's OR exp(-+ z se). A zero margin: NaN and (0, inf)."""
    t = [[int(v) for v in row] for row in np.asarray(p.args[0])]
    (a, b), (c, d) = t
    kind = p.kwargs.get("kind", "conditional")
    if 0 in (a + b, c + d, a + c, b + d):
        z = [0.0, math.inf]
        return {"statistic": NAN, "conf_int": z, "conf_int_less": z, "conf_int_greater": z}
    M, n, N = a + b + c + d, a + b, a + c
    lo, hi = max(0, n + N - M), min(n, N)
    out = {}

    def ends(at):
        if kind == "conditional":
            low = 0.0 if a == lo else _nchg_root(t, lambda k: (1 if k >= a else 0) - at)
            high = math.inf if a == hi else _nchg_root(t, lambda k: (1 if k <= a else 0) - at)
            return low, high
        r = Fraction(a * d, b * c) if b * c else math.inf
        if 0 in (a, b, c, d):  # se = inf: IEEE's ends, as scipy's
            if r == 0:
                return 0.0, NAN
            return NAN, math.inf
        se2 = Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d)

        def end(sg):
            return lambda: A(r) * (sg * norm_isf(at) * A(se2).sqrt()).exp()

        return end(-1), end(1)

    if kind == "conditional":
        out["statistic"] = 0.0 if a == lo else (math.inf if a == hi else _nchg_root(t, lambda k: k - a))
    else:
        out["statistic"] = Fraction(a * d, b * c) if b * c else math.inf
    lo2, hi2 = ends(ALPHA / 2)
    lo1, hi1 = ends(ALPHA)
    out["conf_int"] = [lo2, hi2]
    out["conf_int_less"] = [0.0, hi1]
    out["conf_int_greater"] = [lo1, math.inf]
    return out


@family("relative_risk", ("relative_risk",))
def relative_risk(p):
    """The relative risk (a / n1) / (c / n2) exact, Katz's interval rr exp(-+ z se) in Arb, se^2 =
    1/a - 1/n1 + 1/c - 1/n2 (scipy's ends where a case count is 0)."""
    a, n1, c, n2 = (int(v) for v in p.args)
    if a == 0 and c == 0:
        return {"relative_risk": NAN, "conf_int": [NAN, NAN]}
    if a == 0:
        return {"relative_risk": Fraction(0), "conf_int": [0.0, NAN]}
    if c == 0:
        return {"relative_risk": math.inf, "conf_int": [NAN, math.inf]}
    rr = Fraction(a * n2, n1 * c)
    se2 = Fraction(1, a) - Fraction(1, n1) + Fraction(1, c) - Fraction(1, n2)

    def end(sg):
        return lambda: A(rr) * (sg * norm_isf(ALPHA / 2) * A(se2).sqrt()).exp()

    return {"relative_risk": rr, "conf_int": [end(-1), end(1)]}


@family("binomtest", ("binomtest",), seconds=300)
def binomtest(p):
    """The binomial test with p the double's exact rational q = A/D: the terms
    C(n, j) A^j (D - A)^(n - j) are integers (streamed, not stored), the tails their sums over D^n;
    two-sided by scipy's rule (the other side's terms at most (1 + 1e-7) pmf(k), as scipy's binary
    search finds them)."""
    k, n = int(p.args[0]), int(p.args[1])
    q = fr(p.kwargs.get("p", 0.5))
    alt = p.kwargs.get("alternative", "two-sided")
    a, D = q.numerator, q.denominator
    b = D - a
    if a == 0 or b == 0:
        raise NotSettled("p = 0 or 1: not a case of this family")

    def weights():
        c, ap, bp = 1, 1, b**n
        for j in range(n + 1):
            yield j, c * ap * bp
            if j < n:
                c = c * (n - j) // (j + 1)
                ap *= a
                bp //= b

    total = D**n
    if alt == "less":
        s = sum(w for j, w in weights() if j <= k)
    elif alt == "greater":
        s = sum(w for j, w in weights() if j >= k)
    elif k * D == a * n:
        s = total
    else:
        wk = math.comb(n, k) * a**k * b ** (n - k)
        num, den = RERR.numerator, RERR.denominator  # w_j <= w_k (1 + 1e-7), in integers

        def small(w):
            return w * den <= wk * num

        if k * D < a * n:  # k below the mean: its tail, and the far side's terms from ceil(pn) on
            lo = -(-a * n // D)
            s = sum(w for j, w in weights() if j <= k or (j >= lo and small(w)))
        else:
            hi = a * n // D
            s = sum(w for j, w in weights() if j >= k or (j <= hi and small(w)))
    return {"statistic": Fraction(k, n), "pvalue": min(Fraction(1), Fraction(s, total))}


@family("combine_pvalues", ("combine_pvalues",), lambda p: p.kwargs.get("method", "fisher") != "mudholkar_george")
def combine_pvalues(p):
    """Fisher's (-2 sum log p, chi^2 (2k) upper tail), Pearson's (2 sum log(1 - p), the chi^2 lower
    tail of its negative), Tippett's (1 - (1 - min p)^k, exact) and Stouffer's (the normal isf,
    weighted) combinations, in Arb."""
    ps = fvec(p.args[0])
    k = len(ps)
    method = p.kwargs.get("method", "fisher")
    if method == "fisher":

        def S():
            return -2 * sum(A(v).log() for v in ps)

        return {"statistic": S, "pvalue": lambda: gamma_q(k, S() / 2)}
    if method == "pearson":

        def S():
            return 2 * sum((1 - A(v)).log() for v in ps)

        return {"statistic": S, "pvalue": lambda: gamma_p(k, -S() / 2)}
    if method == "tippett":
        m = min(ps)
        return {"statistic": m, "pvalue": 1 - (1 - m) ** k}
    w = fvec(p.kwargs["weights"]) if p.kwargs.get("weights") is not None else [Fraction(1)] * k

    def Z():
        return sum(A(wi) * norm_isf(v) for wi, v in zip(w, ps)) / A(sum(wi * wi for wi in w)).sqrt()

    return {"statistic": Z, "pvalue": lambda: norm_sf(Z())}


def _multitest(ps, method):
    """statsmodels' multipletests corrections (or scipy's false_discovery_control) in exact
    rationals: the sorted p-values, the step-down/up accumulations, clipped at 1."""
    n = len(ps)
    order = sorted(range(n), key=lambda i: ps[i])
    s = [ps[i] for i in order]
    if method == "bonferroni":
        c = [v * n for v in s]
    elif method == "sidak":
        c = [1 - (1 - v) ** n for v in s]
    elif method in ("holm-sidak", "holm"):
        raw = [(1 - (1 - v) ** (n - i)) if method == "holm-sidak" else v * (n - i) for i, v in enumerate(s)]
        c, run = [], Fraction(0)
        for v in raw:
            run = max(run, v)
            c.append(run)
    elif method == "simes-hochberg":
        raw = [v * (n - i) for i, v in enumerate(s)]
        c, run = [None] * n, None
        for i in range(n - 1, -1, -1):
            run = raw[i] if run is None else min(run, raw[i])
            c[i] = run
    elif method == "hommel":
        a = list(s)
        for m in range(n, 1, -1):
            cim = min(m * v / (j + 1) for j, v in enumerate(s[n - m :]))
            for i in range(n - m, n):
                a[i] = max(a[i], cim)
            for i in range(n - m):
                a[i] = max(a[i], min(m * s[i], cim))
        c = a
    elif method in ("fdr_bh", "fdr_by", "bh", "by"):
        cm = sum(Fraction(1, i) for i in range(1, n + 1)) if method in ("fdr_by", "by") else Fraction(1)
        raw = [v * n * cm / (i + 1) for i, v in enumerate(s)]
        c, run = [None] * n, None
        for i in range(n - 1, -1, -1):
            run = raw[i] if run is None else min(run, raw[i])
            c[i] = run
    else:
        raise KeyError(method)
    out = [None] * n
    for i, v in zip(order, c):
        out[i] = min(Fraction(1), v)
    return out


@family(
    "multipletests",
    ("multipletests", "false_discovery_control"),
    lambda p: p.kwargs.get("method") not in ("fdr_tsbh", "fdr_tsbky"),
)
def multipletests(p):
    """The corrected p-values of multipletests (statsmodels' definitions) and
    false_discovery_control (scipy's), exact."""
    ps = fvec(p.args[0])
    return {"pvalues_corrected": _multitest(ps, p.kwargs.get("method", "holm-sidak"))}


# ---------------------------------------------------------------- building


def _encode(v):
    if isinstance(v, Fraction):
        hi = float(v)
        lo = float(v - Fraction(hi)) if math.isfinite(hi) else 0.0
        return [hi, lo]
    if isinstance(v, (str, int)) and not isinstance(v, bool):
        return v
    if isinstance(v, float):
        return [v, 0.0]
    if isinstance(v, list):
        return [_encode(e) for e in v]
    if callable(v):
        return list(to_dd(rigorous(v)))
    raise TypeError(type(v))


def _point(task):
    name, pid = task
    from tests.htest import problems as P

    fam = FAMILIES[name]
    p = P.PROBLEMS[pid]
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(fam.seconds)
    t0 = time.monotonic()
    try:
        fields = fam.compute(p)
        rec = {"digest": P.digest(p)}
        for k, v in fields.items():
            rec[k] = v if k.startswith("kappa") else _encode(v)  # (a plain float: a tolerance factor)
        return name, pid, rec, time.monotonic() - t0, None
    except Exception as e:  # a point that cannot be settled (time limit, a ball on a boundary)
        return name, pid, None, time.monotonic() - t0, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def build(names, jobs):
    tasks = [(n, p.id) for n in names for p in FAMILIES[n].problems()]
    res = {n: {} for n in names}
    dropped = {n: [] for n in names}
    start = time.monotonic()
    # forked workers replaced every 50 problems: a long run's memory stays flat
    with multiprocessing.get_context("fork").Pool(jobs, maxtasksperchild=50) as pool:
        for name, pid, rec, secs, err in pool.imap_unordered(_point, tasks):
            if rec is None:
                dropped[name].append(pid)
                print(f"  left out {pid}: {err}", flush=True)
            else:
                res[name][pid] = rec
            if secs > 20:
                print(f"  {pid}: {secs:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        order = [p.id for p in FAMILIES[name].problems() if p.id in res[name]]
        prov = {
            **ar.provenance(ar.TARGET_BITS, "exact rationals"),
            "tool": "tools/htest_ref.py",
            "n_requested": len(order) + len(dropped[name]),
            "dropped": sorted(dropped[name]),
        }
        lines = [f"{json.dumps('_provenance')}: {json.dumps(prov)}"]
        lines += [f"{json.dumps(pid)}: {json.dumps(res[name][pid])}" for pid in order]
        (OUT / f"{name}.json").write_text("{\n" + ",\n".join(lines) + "\n}\n")
        print(f"{name}: {len(order)} problems", flush=True)
    left = sum(len(v) for v in dropped.values())
    print(f"{len(tasks)} problems, {left} left out, {(time.monotonic() - start):.0f} s", flush=True)


def main(argv=None):
    import argparse

    from bench import cli

    ap = argparse.ArgumentParser(prog="tools/htest_ref.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="FAMILY", help=", ".join(FAMILIES))
    ap.add_argument("--list", action="store_true", help="list the families and their problems")
    cli.add_jobs(ap, default=JOBS)
    a = ap.parse_args(argv)
    if a.list:
        for n, f in FAMILIES.items():
            print(f"{n}: {', '.join(p.id for p in f.problems())}")
        return 0
    bad = [n for n in a.names if n not in FAMILIES]
    if bad:
        ap.error(f"unknown family {bad[0]!r} (known: {', '.join(FAMILIES)})")
    with ar.logged("htest_ref"):
        build(a.names or list(FAMILIES), a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
