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


@family("mannwhitneyu", ("mannwhitneyu",), seconds=300)
def mannwhitneyu(p):
    """Mann-Whitney U from exact midranks; the exact p-value from the integer counts (scipy's rule:
    U = max(U1, U2) and twice the tail for two-sided), the normal one with scipy's tie and
    continuity corrections."""
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
    U, f = (u1, 1) if alt == "greater" else (u2, 1) if alt == "less" else (max(u1, u2), 2)
    if method == "exact":
        c = mwu_counts(n1, n2)
        tot = sum(c)
        pv = min(Fraction(1), f * Fraction(sum(c[math.ceil(U) :]), tot))
        return {"statistic": u1, "pvalue": pv, "method": "exact"}
    n = n1 + n2
    s2 = Fraction(n1 * n2, 12) * ((n + 1) - Fraction(sum(t**3 - t for t in ties), n * (n - 1)))
    num = u1 - Fraction(n1 * n2, 2)
    if p.kwargs.get("use_continuity", True):
        sg = 1 if alt == "greater" else -1 if alt == "less" else sign(num)
        num -= Fraction(sg, 2)
    return {"statistic": u1, "pvalue": lambda: norm_p(A(num) / A(s2).sqrt(), alt), "method": "asymptotic"}


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
    """The Wilcoxon signed rank test: exact midranks of |d|, T+; the exact p-value from the subset
    counts (no ties or zeros), the normal one with scipy's zero methods and tie correction."""
    d = fvec(p.args[0]) if len(p.args) == 1 else [a - b for a, b in zip(fvec(p.args[0]), fvec(p.args[1]))]
    zm = p.kwargs.get("zero_method", "wilcox")
    alt = p.kwargs.get("alternative", "two-sided")
    method = p.kwargs.get("method", "auto")
    n_all = len(d)
    zeros = sum(1 for v in d if v == 0)
    if zm == "wilcox":
        d = [v for v in d if v != 0]
    r, ties = midranks([abs(v) for v in d])
    rp = sum(ri for ri, v in zip(r, d) if v > 0)
    rm = sum(ri for ri, v in zip(r, d) if v < 0)
    if zm == "zsplit":
        rz = sum(ri for ri, v in zip(r, d) if v == 0) / 2
        rp, rm = rp + rz, rm + rz
    has_ties = any(t > 1 for t in ties)
    if method == "auto":
        if n_all > 50:
            method = "asymptotic"
        elif not (has_ties or zeros):
            method = "exact"
        elif n_all <= 13:
            raise NotSettled("scipy enumerates the signs (a permutation test): not a reference here")
        else:
            method = "asymptotic"
    stat = min(rp, rm) if alt == "two-sided" else rp
    count = len(d)
    if method == "exact":
        c = signed_rank_counts(count)
        tot = sum(c)
        cdf = Fraction(sum(c[: math.ceil(rp) + 1]), tot)
        sf = Fraction(sum(c[math.floor(rp) :]), tot)
        pv = cdf if alt == "less" else sf if alt == "greater" else min(Fraction(1), 2 * min(sf, cdf))
        return {"statistic": stat, "pvalue": pv, "method": "exact"}
    mn = Fraction(count * (count + 1), 4)
    se = Fraction(count * (count + 1) * (2 * count + 1))
    tt = ties
    if zm == "pratt":
        mn -= Fraction(zeros * (zeros + 1), 4)
        se -= zeros * (zeros + 1) * (2 * zeros + 1)
        tt = [t for t, g in zip(*_tie_groups([abs(v) for v in d])) if g != 0]
    se2 = (se - Fraction(sum(t**3 - t for t in tt), 2)) / 24

    def pv():
        z = (A(rp) - A(mn)) / A(se2).sqrt()
        if p.kwargs.get("correction", False):
            sg = 1 if alt == "greater" else -1 if alt == "less" else sign(rp - mn)
            z -= sg * arb(1) / 2 / A(se2).sqrt()
        return norm_p(z, alt)

    return {"statistic": stat, "pvalue": pv, "method": "asymptotic"}


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


@family("ks_2samp", ("ks_2samp",), lambda p: p.kwargs.get("method", "auto") != "asymp", seconds=600)
def ks_2samp(p):
    """Two-sample Kolmogorov-Smirnov, exact: D = h / lcm(n1, n2) from the ECDFs at the distinct
    values (ties as scipy computes them), the p-value the share of lattice paths whose gap reaches
    h (the distribution without ties, as scipy's exact method)."""
    x, y = sorted(fvec(p.args[0])), sorted(fvec(p.args[1]))
    n1, n2 = len(x), len(y)
    L = math.lcm(n1, n2)
    a, b = L // n1, L // n2
    alt = p.kwargs.get("alternative", "two-sided")
    gp = gm = 0
    i = j = 0
    for v in sorted(set(x + y)):
        while i < n1 and x[i] <= v:
            i += 1
        while j < n2 and y[j] <= v:
            j += 1
        gp, gm = max(gp, i * a - j * b), max(gm, j * b - i * a)
    h = gp if alt == "greater" else gm if alt == "less" else max(gp, gm)
    if h == 0:
        return {"statistic": Fraction(0), "pvalue": Fraction(1), "method": "exact"}
    pv = Fraction(ks2_paths(n1, n2, a, b, h, alt), math.comb(n1 + n2, n1))
    return {"statistic": Fraction(h, L), "pvalue": pv, "method": "exact"}


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
    var = (m * (2 * n + 5) - x1 - y1) / 18 + (2 * xtie * ytie) / m + x0 * y0 / (9 * m * (n - 2))
    return {"statistic": stat, "pvalue": lambda: norm_p(A(cmd) / A(var).sqrt(), alt), "method": "asymptotic"}


def _r2(x, y):
    """(r^2, the sign of r): exact."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy * sxy / (sxx * syy), sign(sxy)


def _corr_result(r2, s, n, alt, ci=False):
    nu = Fraction(n - 2)
    one_minus = 1 - r2

    def pv():
        two = beta_reg(one_minus, nu / 2, Fraction(1, 2))
        if alt == "two-sided":
            return two
        upper = s > 0 if alt == "greater" else s < 0
        return two / 2 if upper else 1 - two / 2

    out = {"statistic": (lambda: s * A(r2).sqrt()) if s else Fraction(0), "pvalue": pv}
    if ci and n > 3:
        level = Fraction(0.95)

        def z():
            r = s * A(r2).sqrt()
            return ((1 + r) / (1 - r)).log() / 2

        h = (lambda: -norm_isf(Fraction(1, 2) + level / 2)) if alt == "two-sided" else (lambda: -norm_isf(level))
        se = Fraction(1, n - 3)
        lo = (lambda: (z() - h() * A(se).sqrt()).tanh()) if alt != "less" else Fraction(-1)
        hi = (lambda: (z() + h() * A(se).sqrt()).tanh()) if alt != "greater" else Fraction(1)
        out["confidence_interval"] = [lo, hi]
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
    """Spearman's rho: Pearson's r of the exact midranks and the t approximation (n - 2)."""
    x, y = fvec(p.args[0]), fvec(p.args[1])
    r2, s = _r2(midranks(x)[0], midranks(y)[0])
    return _corr_result(r2, s, len(x), p.kwargs.get("alternative", "two-sided"))


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


@family("fisher_exact", ("fisher_exact",), seconds=600)
def fisher_exact(p):
    """Fisher's exact test, the null in exact rationals. 2 x 2 (the hypergeometric): scipy's
    two-sided rule -- p = 1 when the observed table is as probable as the mode to 1e-14; else the
    observed tail and the other side's tables less probable than (1 + 1e-14) times the observed
    one (at most as probable, below the mode). r x c (the multiple hypergeometric, every table with
    the margins): R's fisher.test rule, the tables at most (1 + 1e-7) times as probable. The 2 x 2
    statistic is the sample odds ratio ad/bc."""
    t = [[int(v) for v in row] for row in np.asarray(p.args[0])]
    rows = [sum(r) for r in t]
    cols = [sum(c) for c in zip(*t)]
    N = sum(rows)
    alt = p.kwargs.get("alternative", "two-sided")
    if len(t) == 2 and len(t[0]) == 2:
        (a, b), (c, d) = t
        stat = Fraction(a * d, b * c) if b * c else (math.inf if a * d else NAN)
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
    num = math.prod(math.factorial(r) for r in rows) * math.prod(math.factorial(c) for c in cols)

    def prob(tab):
        return Fraction(num, math.factorial(N) * math.prod(math.factorial(v) for r in tab for v in r))

    p0 = prob(t)
    pv = sum(q for q in (prob(tab) for tab in _tables(rows, cols)) if q <= p0 * RERR)
    return {"pvalue": min(Fraction(1), pv), "rule": "R"}


@family("chi2_contingency", ("chi2_contingency",))
def chi2_contingency(p):
    """The chi^2 test of independence: exact expected counts, Yates's correction (one degree of
    freedom), Pearson's statistic exact or the log-likelihood ratio in Arb, the chi^2 tail."""
    t = [fvec(r) for r in np.asarray(p.args[0])]
    rows = [sum(r) for r in t]
    cols = [sum(c) for c in zip(*t)]
    N = sum(rows)
    E = [[r * c / N for c in cols] for r in rows]
    df = (len(rows) - 1) * (len(cols) - 1)
    O = [list(r) for r in t]
    if df == 1 and p.kwargs.get("correction", True):
        for i in range(len(O)):
            for j in range(len(O[0])):
                diff = E[i][j] - O[i][j]
                O[i][j] += sign(diff) * min(Fraction(1, 2), abs(diff))
    lam = p.kwargs.get("lambda_")
    pairs = [(o, e) for ro, re in zip(O, E) for o, e in zip(ro, re)]
    out = {"df": df, "expected_freq": [[float(e) for e in r] for r in E]}
    if lam is None or lam == "pearson":
        stat = sum((o - e) ** 2 / e for o, e in pairs)
        out.update(statistic=stat, pvalue=lambda: chi2_sf(stat, df))
        return out

    def S():
        return 2 * sum((A(o) * (A(o) / A(e)).log() if o else arb(0)) for o, e in pairs)

    out.update(statistic=S, pvalue=lambda: chi2_sf(S(), df))
    return out


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
