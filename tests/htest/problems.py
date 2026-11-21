"""The problem collection of sd.htest (docs/design/htest.md §7): fixed inputs for every test of
H1-H8, shared by the scipy and statsmodels generator (tools/gen_htest_scipy.py), the R generator
(tools/gen_htest_r.py), the reference tool (tools/htest_ref.py) and the tests.

A problem is one call: the function's scipy (or statsmodels) name, its positional inputs and
scipy's keyword options. The kinds of input each test meets: small samples (the exact null
distributions), large ones, ties and zeros, extreme tails (p far below 1e-16), data with a large
offset (cancellation in the statistic), degenerate input, and batches (axis=-1).

The inputs are drawn by numpy's PCG64 with fixed seeds and rounded where ties are wanted; every
data file stores each problem's digest (digest()), so a numpy whose streams changed shows up as
a test failure instead of silently new inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np


@dataclasses.dataclass(frozen=True)
class Problem:
    id: str  # "<func>/<case>"
    func: str  # scipy.stats' name (statsmodels' for its functions)
    args: tuple  # positional inputs: numpy arrays, numbers, or a distribution's name ('norm')
    kwargs: dict  # scipy's keyword options
    tags: tuple = ()  # small, large, ties, tail, offset, degenerate, batch, exact, asymptotic
    milestone: str = ""  # H1-H8


def digest(p: Problem) -> str:
    """SHA-256 of the problem's inputs (their float64 bytes, and the text of the rest)."""
    h = hashlib.sha256()
    for a in p.args:
        if isinstance(a, np.ndarray) or isinstance(a, list) and a and isinstance(a[0], np.ndarray):
            for x in a if isinstance(a, list) else [a]:
                x = np.ascontiguousarray(x, dtype=np.float64)
                h.update(repr(x.shape).encode())
                h.update(x.tobytes())
        else:
            h.update(repr(a).encode())
    h.update(repr(sorted(p.kwargs.items())).encode())
    return h.hexdigest()


def _g(seed):
    return np.random.Generator(np.random.PCG64(seed))


def normal(seed, n, loc=0.0, scale=1.0):
    return _g(seed).normal(loc, scale, size=n)


def rounded(seed, n, step=1.0, loc=0.0, scale=3.0):
    """Normal values rounded to multiples of step: ties."""
    return np.round(_g(seed).normal(loc, scale, size=n) / step) * step


def ints(seed, n, k):
    return _g(seed).integers(0, k, size=n).astype(np.float64)


PROBLEMS: dict[str, Problem] = {}


def add(milestone, func, case, *args, tags=(), **kwargs):
    pid = f"{func}/{case}"
    if pid in PROBLEMS:
        raise ValueError(f"duplicate problem {pid}")
    PROBLEMS[pid] = Problem(pid, func, tuple(args), dict(kwargs), tuple(tags), milestone)


# ---- H1: means and variances --------------------------------------------------------------


def _h1():
    m = "H1"
    add(m, "ttest_1samp", "small", normal(1, 8, 0.4), 0.0, tags=("small",))
    add(m, "ttest_1samp", "small-less", normal(1, 8, 0.4), 0.0, alternative="less", tags=("small",))
    add(m, "ttest_1samp", "small-greater", normal(1, 8, 0.4), 0.0, alternative="greater", tags=("small",))
    add(m, "ttest_1samp", "large", normal(2, 2000, 0.05), 0.0, tags=("large",))
    add(m, "ttest_1samp", "tail", normal(3, 60, 3.0), 0.0, tags=("tail",))
    add(m, "ttest_1samp", "deep-tail", normal(4, 400, 2.0), 0.0, tags=("tail",))
    add(m, "ttest_1samp", "offset", normal(5, 200, 1e8 + 0.1), 1e8, tags=("offset",))
    add(m, "ttest_1samp", "constant", np.array([1.0, 1.0, 1.0]), 0.0, tags=("degenerate",))
    add(m, "ttest_1samp", "two", np.array([1.0, 2.0]), 0.0, tags=("small",))
    add(m, "ttest_1samp", "batch", normal(6, (5, 30), 0.2), 0.0, axis=-1, tags=("batch",))
    a, b = normal(7, 12, 0.3), normal(8, 12)
    add(m, "ttest_rel", "small", a, b, tags=("small",))
    add(m, "ttest_rel", "large", normal(9, 1500, 0.1), normal(10, 1500), tags=("large",))
    off = normal(11, 100, 1e9)
    add(m, "ttest_rel", "offset", off + normal(12, 100, 0.05, 1e-3), off, tags=("offset",))
    add(m, "ttest_rel", "batch", normal(13, (4, 25), 0.3), normal(14, (4, 25)), axis=-1, tags=("batch",))
    x, y = normal(15, 9), normal(16, 13, 0.8)
    add(m, "ttest_ind", "small", x, y, tags=("small",))
    add(m, "ttest_ind", "small-welch", x, y, equal_var=False, tags=("small",))
    add(m, "ttest_ind", "small-greater", x, y, alternative="greater", tags=("small",))
    add(m, "ttest_ind", "unequal-var", normal(17, 20, 0, 1), normal(18, 7, 0.5, 6), equal_var=False)
    add(m, "ttest_ind", "large", normal(19, 3000, 0.05), normal(20, 2500), tags=("large",))
    add(m, "ttest_ind", "tail", normal(21, 100, 3.0), normal(22, 100), tags=("tail",))
    add(m, "ttest_ind", "offset", normal(23, 50, 1e8 + 0.5), normal(24, 60, 1e8), tags=("offset",))
    add(m, "ttest_ind", "trim", normal(25, 30, 0.6), normal(26, 25), trim=0.2)
    add(m, "ttest_ind", "batch", normal(27, (6, 20), 0.4), normal(28, (6, 15)), axis=-1, tags=("batch",))
    add(m, "ttest_ind_from_stats", "student", 0.3, 1.1, 15, -0.2, 0.9, 12)
    add(m, "ttest_ind_from_stats", "welch", 0.3, 1.1, 15, -0.2, 3.9, 12, equal_var=False)
    g3 = [normal(29, 8, 0.0), normal(30, 10, 0.6), normal(31, 9, 0.2)]
    add(m, "f_oneway", "small", *g3, tags=("small",))
    add(m, "f_oneway", "welch", *g3, equal_var=False, tags=("small",))
    add(m, "f_oneway", "large", normal(32, 800), normal(33, 900, 0.1), normal(34, 700, 0.05), tags=("large",))
    add(m, "f_oneway", "offset", normal(35, 40, 1e8), normal(36, 40, 1e8 + 0.3), normal(37, 40, 1e8 - 0.2))
    add(m, "f_oneway", "tail", normal(38, 60), normal(39, 60, 2.5), normal(40, 60, -2.5), tags=("tail",))
    add(m, "f_oneway", "batch", normal(41, (5, 12)), normal(42, (5, 14), 0.5), axis=-1, tags=("batch",))
    add(m, "alexandergovern", "small", *g3, tags=("small",))
    add(m, "alexandergovern", "large", normal(43, 500), normal(44, 400, 0.1, 2), normal(45, 300, 0, 0.5))
    add(m, "bartlett", "small", normal(46, 10), normal(47, 12, 0, 2), normal(48, 9, 0, 0.7), tags=("small",))
    base = normal(49, 400)
    add(m, "bartlett", "near-equal", base, base[::-1] * (1 + 1e-9), base * (1 - 1e-9) + 0.5)
    add(m, "bartlett", "large", normal(50, 2000), normal(51, 2000, 0, 1.05), tags=("large",))
    for c in ("median", "mean", "trimmed"):
        add(m, "levene", c, normal(52, 15), normal(53, 18, 0, 2), normal(54, 12, 0, 0.6), center=c)
        add(m, "fligner", c, normal(55, 15), normal(56, 18, 0, 2), normal(57, 12, 0, 0.6), center=c)
    add(m, "levene", "large", normal(58, 1000), normal(59, 1200, 0, 1.1), tags=("large",))
    add(m, "fligner", "ties", rounded(60, 20), rounded(61, 25, scale=5), tags=("ties",))
    # (H1's implementation: degenerate inputs, offsets beyond the doubles' rounding of a - b,
    # Yuen's trimmed test, Welch's ANOVA, the tails, and every k-sample test batched)
    add(m, "ttest_1samp", "offset-constant", np.full(5, 1e8), 1e8, tags=("degenerate", "offset"))
    add(m, "ttest_rel", "offset-wide", 3e9 + normal(1001, 3, 0, 1e-3), 1e9 + normal(1002, 3, 0, 1e-3), tags=("offset",))
    add(m, "ttest_ind", "constant-both", np.full(4, 2.0), np.full(5, 3.0), tags=("degenerate",))
    add(m, "ttest_ind", "one-element", np.array([0.7]), normal(1003, 8), tags=("small",))
    add(m, "ttest_ind", "trim-welch", normal(1004, 40, 0.3, 1.0), normal(1005, 30, 0, 3.0), trim=0.1, equal_var=False)
    add(m, "ttest_ind", "trim-ties", rounded(1006, 25), rounded(1007, 20, loc=1.0), trim=0.25, tags=("ties",))
    add(m, "ttest_ind", "offset-welch", normal(1008, 50, 1e8 + 0.5, 1), normal(1009, 60, 1e8, 3), equal_var=False,
        tags=("offset",))  # fmt: skip
    add(m, "ttest_ind", "deep-tail-greater", normal(1010, 500, 1.0), normal(1011, 500), alternative="greater",
        tags=("tail",))  # fmt: skip
    add(m, "ttest_ind_from_stats", "zero-std", 1.0, 0.0, 5, 2.0, 0.0, 6, tags=("degenerate",))
    add(m, "ttest_ind_from_stats", "welch-less", 0.3, 1.1, 15, -0.2, 3.9, 12, equal_var=False, alternative="less")
    add(m, "f_oneway", "two-groups", x, y, tags=("small",))
    add(m, "f_oneway", "constant-groups", np.full(3, 2.0), np.full(2, 3.0), np.full(4, 5.0), tags=("degenerate",))
    add(m, "f_oneway", "welch-offset", normal(1012, 30, 1e8, 1), normal(1013, 40, 1e8 + 0.4, 2),
        normal(1014, 35, 1e8 - 0.3, 0.5), equal_var=False, tags=("offset",))  # fmt: skip
    add(m, "f_oneway", "welch-large", normal(1015, 2000), normal(1016, 1500, 0.05, 2), normal(1017, 1000, 0, 0.5),
        equal_var=False, tags=("large",))  # fmt: skip
    add(m, "f_oneway", "welch-tail", normal(1018, 100, 0, 1), normal(1019, 100, 3, 2), normal(1020, 100, -3, 1),
        equal_var=False, tags=("tail",))  # fmt: skip
    add(m, "f_oneway", "welch-batch", normal(1021, (5, 12)), normal(1022, (5, 14), 0.5, 2), axis=-1, equal_var=False,
        tags=("batch",))  # fmt: skip
    add(m, "alexandergovern", "offset", normal(1023, 40, 1e8, 1), normal(1024, 30, 1e8 + 0.5, 2),
        normal(1025, 35, 1e8, 0.5), tags=("offset",))  # fmt: skip
    add(m, "alexandergovern", "tail", normal(1026, 200), normal(1027, 200, 2, 2), normal(1028, 200, -2, 1),
        tags=("tail",))  # fmt: skip
    add(m, "alexandergovern", "batch", normal(1029, (4, 15)), normal(1030, (4, 18), 0.3, 2), axis=-1, tags=("batch",))
    add(m, "bartlett", "offset", normal(1031, 100, 1e8, 1), normal(1032, 120, 1e8, 1.3), normal(1033, 90, 1e8, 0.8),
        tags=("offset",))  # fmt: skip
    add(m, "bartlett", "tail", normal(1034, 300, 0, 1), normal(1035, 300, 0, 3), tags=("tail",))
    add(m, "bartlett", "two", normal(1036, 15), normal(1037, 20, 0, 2))
    add(m, "bartlett", "batch", normal(1038, (4, 15)), normal(1039, (4, 20), 0, 2), axis=-1, tags=("batch",))
    add(m, "levene", "offset", normal(1040, 40, 1e8, 1), normal(1041, 41, 1e8, 2), normal(1042, 30, 1e8, 1.5),
        tags=("offset",))  # fmt: skip
    add(m, "levene", "trimmed-0.2", normal(52, 15), normal(53, 18, 0, 2), normal(54, 12, 0, 0.6), center="trimmed",
        proportiontocut=0.2)  # fmt: skip
    add(m, "levene", "tail", normal(1043, 500, 0, 1), normal(1044, 500, 0, 3), tags=("tail",))
    add(m, "levene", "batch", normal(1045, (4, 15)), normal(1046, (4, 20), 0, 2), axis=-1, tags=("batch",))
    add(m, "fligner", "offset", normal(1047, 40, 1e8, 1), normal(1048, 41, 1e8, 2), normal(1049, 30, 1e8, 1.5),
        tags=("offset",))  # fmt: skip
    add(m, "fligner", "large", normal(1050, 1500), normal(1051, 1200, 0, 1.2), normal(1052, 1000, 0, 0.8),
        tags=("large",))  # fmt: skip
    add(m, "fligner", "trimmed-0.2", normal(55, 15), normal(56, 18, 0, 2), normal(57, 12, 0, 0.6), center="trimmed",
        proportiontocut=0.2)  # fmt: skip
    add(m, "fligner", "tail", normal(1053, 500, 0, 1), normal(1054, 500, 0, 3), tags=("tail",))
    add(m, "fligner", "batch", normal(1055, (4, 15)), normal(1056, (4, 20), 0, 2), axis=-1, tags=("batch",))


# ---- H2: goodness of fit to a given distribution ------------------------------------------


def _h2():
    m = "H2"
    obs = np.array([16.0, 18, 16, 14, 12, 12])
    add(m, "chisquare", "uniform", obs)
    add(m, "chisquare", "expected", obs, np.array([16.0, 16, 16, 16, 16, 8]))
    add(m, "chisquare", "ddof", obs, ddof=1)
    near = np.array([1e6 + 1e-3, 1e6 - 1e-3, 1e6])
    add(m, "chisquare", "near-equal", near, np.full(3, sum(near) / 3), tags=("offset",))
    add(m, "chisquare", "tail", np.array([500.0, 10, 10, 10]), tags=("tail",))
    add(m, "chisquare", "batch", ints(62, (4, 6), 30) + 1, axis=-1, tags=("batch",))
    for lam in ("pearson", "log-likelihood", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"):
        add(m, "power_divergence", lam, obs, lambda_=lam)
    add(m, "power_divergence", "lambda-0.3", obs, lambda_=0.3)
    e = np.array([1e6, 1e6, 1e6])
    add(m, "power_divergence", "g-near-equal", e + np.array([1e-3, -1e-3, 0.0]), e, lambda_="log-likelihood")
    add(m, "power_divergence", "g-tail", np.array([900.0, 50, 50]), lambda_="log-likelihood", tags=("tail",))
    for n in (20, 500, 5000):
        x = normal(63 + n, n)
        add(m, "skewtest", f"n{n}", x)
        add(m, "kurtosistest", f"n{n}", x)
        add(m, "normaltest", f"n{n}", x)
    sk = _g(70).exponential(size=300)
    add(m, "skewtest", "skewed", sk, tags=("tail",))
    add(m, "skewtest", "skewed-less", sk, alternative="less")
    add(m, "kurtosistest", "heavy", _g(71).standard_t(3, size=300), alternative="greater")
    add(m, "normaltest", "skewed", sk, tags=("tail",))
    add(m, "skewtest", "batch", normal(72, (4, 30)), axis=-1, tags=("batch",))
    for n in (8, 50, 2000):
        add(m, "jarque_bera", f"n{n}", normal(73 + n, n))
    add(m, "jarque_bera", "skewed", _g(74).exponential(size=2000), tags=("tail",))
    add(m, "jarque_bera", "batch", normal(75, (5, 40)), axis=-1, tags=("batch",))
    for n in (3, 10, 50, 500, 3000):
        add(m, "shapiro", f"n{n}", normal(76 + n, n))
    add(m, "shapiro", "ties", rounded(77, 40), tags=("ties",))
    add(m, "shapiro", "skewed", _g(78).exponential(size=500), tags=("tail",))
    add(m, "shapiro", "offset", normal(79, 50, 1e8), tags=("offset",))
    add(m, "anderson", "norm", normal(80, 50), method="interpolate")
    add(m, "anderson", "norm-large", normal(81, 2000), method="interpolate")
    add(m, "anderson", "expon", _g(82).exponential(size=60), dist="expon", method="interpolate")
    for n in (10, 100, 1000):
        add(m, "ks_1samp", f"n{n}", normal(83 + n, n, 0.1), "norm", tags=("exact",) if n <= 100 else ())
    add(m, "ks_1samp", "less", normal(84, 30, 0.3), "norm", alternative="less")
    add(m, "ks_1samp", "greater", normal(84, 30, 0.3), "norm", alternative="greater")
    add(m, "ks_1samp", "tail", normal(85, 80, 1.5), "norm", tags=("tail",))
    add(m, "ks_1samp", "asymp", normal(86, 40), "norm", method="asymp")
    add(m, "ks_1samp", "approx", normal(86, 40), "norm", method="approx")
    add(m, "cramervonmises", "n10", normal(87, 10), "norm")
    add(m, "cramervonmises", "n200", normal(88, 200, 0.2), "norm")
    _h2_more(m)


def _h2_more(m):
    """H2's own problems (2026-11-20): zero counts in the Cressie-Read family, the degenerate
    inputs, the D'Agostino transforms at their small-n bounds and at zero skewness, the families of
    anderson, the deep tails of the Kolmogorov-Smirnov distributions (where scipy's two-sided
    tail is 0 or 2 smirnov and R's exact one is 1 - cdf), and the Cramér-von Mises tails."""
    zc = np.array([12.0, 0, 9, 15, 4])
    for lam in ("freeman-tukey", "cressie-read", "log-likelihood", "pearson"):
        add(m, "power_divergence", f"zero-{lam}", zc, lambda_=lam, tags=("ties",))
    add(m, "power_divergence", "lambda-small", obs_h2(), lambda_=1e-7)
    add(m, "power_divergence", "lambda-near-minus-1", obs_h2(), lambda_=-0.999)
    add(m, "power_divergence", "cr-near-equal", np.array([1e6 + 2e-3, 1e6 - 1e-3, 1e6 - 1e-3]), lambda_="cressie-read",
        tags=("offset",))  # fmt: skip
    add(m, "power_divergence", "ft-near-equal", np.array([5e5 + 1e-3, 5e5 - 1e-3, 5e5]), lambda_="freeman-tukey",
        tags=("offset",))  # fmt: skip
    add(m, "chisquare", "one-cell", np.array([7.0]), tags=("degenerate",))
    add(m, "chisquare", "ddof-minus-1", obs_h2(), np.array([15.0, 15, 15, 15, 15, 13]), ddof=-1, sum_check=False)
    add(m, "chisquare", "large", ints(64, 3000, 50) + 1, tags=("large",))
    add(m, "skewtest", "n8", normal(130, 8), tags=("small",))
    add(m, "skewtest", "symmetric", np.array([-4.0, -3, -2, -1, 0, 1, 2, 3, 4]), tags=("small",))
    add(m, "skewtest", "offset", normal(131, 200, 1e8) + _g(132).exponential(size=200), tags=("offset",))
    add(m, "skewtest", "constant", np.full(20, 3.25), tags=("degenerate",))
    add(m, "skewtest", "tail", _g(133).exponential(size=5000) ** 2, tags=("tail",))
    add(m, "kurtosistest", "n5", normal(134, 5), tags=("small",))
    add(m, "kurtosistest", "light-less", _g(135).uniform(size=400), alternative="less", tags=("tail",))
    add(m, "kurtosistest", "offset", normal(136, 300, 1e8), tags=("offset",))
    add(m, "kurtosistest", "constant", np.full(20, -1.5), tags=("degenerate",))
    add(m, "kurtosistest", "batch", normal(137, (4, 30)), axis=-1, tags=("batch",))
    add(m, "normaltest", "n8", normal(138, 8), tags=("small",))
    add(m, "normaltest", "tail", _g(133).exponential(size=5000) ** 2, tags=("tail",))
    add(m, "normaltest", "batch", normal(139, (4, 30)), axis=-1, tags=("batch",))
    add(m, "jarque_bera", "offset", normal(140, 300, 1e8), tags=("offset",))
    add(m, "jarque_bera", "constant", np.full(10, 2.5), tags=("degenerate",))
    add(m, "jarque_bera", "n2", np.array([1.0, 4.0]), tags=("small",))
    for n in (4, 5, 6, 11, 12, 5000, 20000):
        add(m, "shapiro", f"n{n}", normal(141 + n, n), tags=("large",) if n >= 5000 else ())
    add(m, "shapiro", "n3-low", np.array([0.0, 0.0001, 1.0]), tags=("small",))
    add(m, "shapiro", "constant", np.full(10, 4.0), tags=("degenerate",))
    add(m, "shapiro", "tail", _g(142).exponential(size=2000) ** 3, tags=("tail",))
    add(m, "shapiro", "batch", normal(143, (4, 25)), axis=-1, tags=("batch",))
    add(m, "anderson", "norm-n8", normal(144, 8), method="interpolate", tags=("small",))
    add(m, "anderson", "norm-skewed", _g(145).exponential(size=100), method="interpolate", tags=("tail",))
    add(m, "anderson", "logistic", _g(146).logistic(1.0, 2.0, size=80), dist="logistic", method="interpolate")
    add(m, "anderson", "logistic-normal", normal(147, 300), dist="logistic", method="interpolate")
    add(m, "anderson", "gumbel", _g(148).gumbel(0.5, 1.5, size=70), dist="gumbel_r", method="interpolate")
    add(m, "anderson", "gumbel-normal", normal(149, 400, 3.0), dist="gumbel_r", method="interpolate")
    add(m, "anderson", "expon-large", _g(150).exponential(2.0, size=3000), dist="expon", method="interpolate")
    add(m, "anderson", "constant", np.full(10, 1.0), method="interpolate", tags=("degenerate",))
    add(m, "anderson", "batch", normal(151, (4, 30)), method="interpolate", axis=-1, tags=("batch",))
    add(m, "ks_1samp", "n1", np.array([0.7]), "norm", tags=("small", "exact"))
    add(m, "ks_1samp", "n2", np.array([-0.3, 1.9]), "norm", tags=("small", "exact"))
    add(m, "ks_1samp", "n141", normal(152, 141, 0.2), "norm", tags=("exact",))
    add(m, "ks_1samp", "n400", normal(153, 400, 0.08), "norm", tags=("exact",))
    add(m, "ks_1samp", "deep", normal(154, 60, 4.0), "norm", tags=("tail",))
    add(m, "ks_1samp", "deep-400", normal(155, 400, 2.5), "norm", tags=("tail",))
    add(m, "ks_1samp", "greater-deep", normal(156, 300, -1.5), "norm", alternative="greater", tags=("tail",))
    add(m, "ks_1samp", "asymp-deep", normal(154, 60, 4.0), "norm", method="asymp", tags=("tail",))
    add(m, "ks_1samp", "approx-deep", normal(154, 60, 4.0), "norm", method="approx", tags=("tail",))
    add(m, "ks_1samp", "uniform", _g(157).uniform(size=50) ** 1.3, "uniform")
    add(m, "ks_1samp", "expon", _g(158).exponential(size=200), "expon")
    add(m, "ks_1samp", "batch", normal(159, (4, 30), 0.2), "norm", axis=-1, tags=("batch",))
    add(m, "cramervonmises", "n2", np.array([-0.4, 0.9]), "norm", tags=("small",))
    add(m, "cramervonmises", "n3", np.array([0.1, 0.5, 2.0]), "norm", tags=("small",))
    add(m, "cramervonmises", "tail", normal(160, 100, 0.7), "norm", tags=("tail",))
    add(m, "cramervonmises", "large", normal(161, 2000, 0.03), "norm", tags=("large",))
    add(m, "cramervonmises", "uniform", _g(162).uniform(size=40) ** 1.2, "uniform")
    add(m, "cramervonmises", "batch", normal(163, (4, 30), 0.2), "norm", axis=-1, tags=("batch",))


def obs_h2():
    return np.array([16.0, 18, 16, 14, 12, 12])


# ---- H3: rank tests -----------------------------------------------------------------------


def _h3():
    m = "H3"
    add(m, "mannwhitneyu", "exact-5-7", normal(90, 5), normal(91, 7, 1.0), tags=("small", "exact"))
    add(m, "mannwhitneyu", "exact-8-8", normal(92, 8), normal(93, 8, 0.5), tags=("small", "exact"))
    add(m, "mannwhitneyu", "exact-less", normal(92, 8), normal(93, 8, 0.5), alternative="less", tags=("exact",))
    add(m, "mannwhitneyu", "exact-forced", normal(94, 15), normal(95, 20, 0.6), method="exact", tags=("exact",))
    add(m, "mannwhitneyu", "ties", rounded(96, 20), rounded(97, 25, loc=1.0), tags=("ties",))
    add(m, "mannwhitneyu", "no-continuity", rounded(96, 20), rounded(97, 25, loc=1.0), use_continuity=False)
    add(m, "mannwhitneyu", "large", normal(98, 500), normal(99, 600, 0.1), tags=("large",))
    add(m, "mannwhitneyu", "separated", normal(100, 200), normal(101, 200, 100.0), tags=("tail",))
    add(m, "mannwhitneyu", "batch", normal(102, (5, 12)), normal(103, (5, 10), 0.5), axis=-1, tags=("batch",))
    add(m, "ranksums", "small", normal(104, 10), normal(105, 12, 0.7))
    add(m, "ranksums", "large", normal(106, 800), normal(107, 700, 0.1), tags=("large",))
    d = normal(108, 10, 0.5)
    add(m, "wilcoxon", "exact-10", d, tags=("small", "exact"))
    add(m, "wilcoxon", "exact-40", normal(109, 40, 0.3), tags=("exact",))
    add(m, "wilcoxon", "exact-greater", d, alternative="greater", tags=("exact",))
    add(m, "wilcoxon", "asymptotic-60", normal(110, 60, 0.2), tags=("asymptotic",))
    add(m, "wilcoxon", "asymptotic-forced", normal(110, 60, 0.2), method="asymptotic")
    z = rounded(111, 30, step=1.0, loc=0.6, scale=2.0)
    add(m, "wilcoxon", "zeros-wilcox", z, tags=("ties",))
    add(m, "wilcoxon", "zeros-pratt", z, zero_method="pratt", tags=("ties",))
    add(m, "wilcoxon", "zeros-zsplit", z, zero_method="zsplit", tags=("ties",))
    add(m, "wilcoxon", "correction", normal(112, 70, 0.2), correction=True, method="asymptotic")
    add(m, "wilcoxon", "paired", normal(113, 15, 0.4), normal(114, 15))
    add(m, "wilcoxon", "tail", normal(115, 300, 3.0), method="asymptotic", tags=("tail",))
    add(m, "kruskal", "small", normal(116, 6), normal(117, 7, 1.0), normal(118, 5, -0.5))
    add(m, "kruskal", "ties", rounded(119, 15), rounded(120, 18, loc=1), rounded(121, 12), tags=("ties",))
    add(m, "kruskal", "large", normal(122, 500), normal(123, 600, 0.1), tags=("large",))
    add(m, "friedmanchisquare", "small", normal(124, 10), normal(125, 10, 0.5), normal(126, 10, 0.2))
    add(m, "friedmanchisquare", "ties", rounded(127, 12), rounded(128, 12, loc=1), rounded(129, 12), tags=("ties",))
    add(m, "brunnermunzel", "small", normal(130, 10), normal(131, 12, 0.8))
    add(m, "brunnermunzel", "normal", normal(130, 10), normal(131, 12, 0.8), distribution="normal")
    add(m, "brunnermunzel", "large", normal(132, 400), normal(133, 500, 0.1), tags=("large",))
    add(m, "ansari", "exact", normal(134, 10), normal(135, 12, 0, 2.5), tags=("small", "exact"))
    add(m, "ansari", "ties", rounded(136, 30), rounded(137, 35, scale=6), tags=("ties",))
    add(m, "ansari", "large", normal(138, 100), normal(139, 120, 0, 1.3), tags=("large",))
    add(m, "mood", "small", normal(140, 10), normal(141, 12, 0, 2.5))
    add(m, "mood", "large", normal(142, 300), normal(143, 400, 0, 1.2), tags=("large",))
    add(m, "median_test", "three", normal(144, 15), normal(145, 18, 0.7), normal(146, 12, -0.4))
    add(m, "median_test", "ties-above", rounded(147, 20), rounded(148, 22, loc=1), ties="above")
    add(m, "page_trend_test", "exact", normal(149, (5, 3)) + np.arange(3) * 0.5, tags=("exact",))
    add(m, "page_trend_test", "asymptotic", normal(150, (20, 4)) + np.arange(4) * 0.3)
    # added with H3: the exact distributions with ties (conditional on the midranks, H45), the
    # 'auto' thresholds, the deep exact tails, the z of the normal approximation, degenerate input
    tx, ty = rounded(3001, 12), rounded(3002, 15, loc=1.0)
    add(m, "mannwhitneyu", "ties-exact", tx, ty, method="exact", tags=("ties", "exact"))
    add(m, "mannwhitneyu", "ties-exact-greater", ty, tx, method="exact", alternative="greater", tags=("ties", "exact"))
    add(m, "mannwhitneyu", "ties-small-auto", rounded(3003, 6), rounded(3004, 7, loc=1.0), tags=("ties", "small"))
    add(m, "mannwhitneyu", "asymptotic-forced", normal(3005, 30), normal(3006, 25, 0.4), method="asymptotic")
    add(m, "mannwhitneyu", "exact-8-300", normal(3007, 8), normal(3008, 300, 0.5), tags=("exact",))
    add(
        m, "mannwhitneyu", "exact-deep", normal(3009, 30), normal(3010, 30, 6.0), method="exact", tags=("tail", "exact")
    )
    add(m, "mannwhitneyu", "constant", np.full(6, 2.0), np.full(7, 2.0), tags=("degenerate",))
    add(m, "ranksums", "tail", normal(3011, 300), normal(3012, 300, 3.0), alternative="less", tags=("tail",))
    add(m, "ranksums", "ties", rounded(3013, 40), rounded(3014, 35, loc=0.5), tags=("ties",))
    half = np.round(normal(3015, 25, 0.4, 2.0)) / 2 + 0.25  # ties, no zeros
    add(m, "wilcoxon", "ties-exact", half, method="exact", tags=("ties", "exact"))
    zz = rounded(3016, 22, step=1.0, loc=0.7, scale=2.0)
    add(m, "wilcoxon", "zeros-exact-pratt", zz, zero_method="pratt", method="exact", tags=("ties", "exact"))
    add(m, "wilcoxon", "zeros-exact-zsplit", zz, zero_method="zsplit", method="exact", tags=("ties", "exact"))
    add(m, "wilcoxon", "zeros-auto-small", zz[:12], tags=("ties", "small", "exact"))
    add(m, "wilcoxon", "ties-auto-small", half[:11], alternative="greater", tags=("ties", "small", "exact"))
    add(m, "wilcoxon", "exact-deep", normal(3017, 60, 3.0), method="exact", tags=("tail", "exact"))
    add(m, "wilcoxon", "exact-200", normal(3018, 200, 0.15), method="exact", tags=("large", "exact"))
    add(m, "wilcoxon", "less", normal(3019, 35, -0.4), alternative="less")
    add(m, "wilcoxon", "constant-zeros", np.zeros(10), tags=("degenerate",))
    mixed = np.stack([normal(3020, 20, 0.3), rounded(3021, 20, loc=0.5), normal(3022, 20, -0.1)])
    add(m, "wilcoxon", "batch-mixed", mixed, axis=-1, tags=("batch",))
    add(m, "kruskal", "tail", normal(3023, 50), normal(3024, 50, 3.0), normal(3025, 50, 6.0), tags=("tail",))
    add(m, "kruskal", "five", *[normal(3026 + i, 8 + i, 0.3 * i) for i in range(5)])
    add(m, "kruskal", "constant", np.full(4, 1.5), np.full(5, 1.5), tags=("degenerate",))
    add(m, "kruskal", "batch", normal(3031, (4, 10)), normal(3032, (4, 12), 0.5), axis=-1, tags=("batch",))
    add(m, "friedmanchisquare", "four-small", *[normal(3033 + i, 5, 0.4 * i) for i in range(4)], tags=("small",))
    add(m, "friedmanchisquare", "large", normal(3037, 500), normal(3038, 500, 0.1), normal(3039, 500), tags=("large",))
    add(m, "friedmanchisquare", "tail", *[normal(3040 + i, 40, 1.5 * i) for i in range(4)], tags=("tail",))
    add(m, "brunnermunzel", "ties", rounded(3044, 25), rounded(3045, 30, loc=1.0), tags=("ties",))
    add(m, "brunnermunzel", "less", normal(3046, 15), normal(3047, 18, 0.6), alternative="less")
    add(m, "brunnermunzel", "greater-normal", normal(3046, 15), normal(3047, 18, 0.6), alternative="greater",
        distribution="normal")  # fmt: skip
    add(m, "brunnermunzel", "separated", normal(3048, 8), normal(3049, 9, 50.0), tags=("degenerate",))
    add(m, "ansari", "exact-54", normal(3050, 54), normal(3051, 54, 0, 1.4), tags=("exact",))
    add(m, "ansari", "exact-odd", normal(3052, 20), normal(3053, 25, 0, 2.0), tags=("exact",))
    add(m, "ansari", "exact-greater", normal(3052, 20), normal(3053, 25, 0, 2.0), alternative="greater")
    add(
        m, "ansari", "ties-exact", rounded(3054, 15), rounded(3055, 18, scale=6), method="exact", tags=("ties", "exact")
    )
    add(m, "ansari", "less-asymptotic", normal(3056, 40), normal(3057, 50, 0, 0.7), alternative="less",
        method="asymptotic")  # fmt: skip
    add(m, "mood", "ties", rounded(3058, 30), rounded(3059, 35, scale=6), tags=("ties",))
    add(m, "mood", "tail", normal(3060, 300), normal(3061, 300, 0, 4.0), tags=("tail",))
    add(m, "mood", "less", normal(3062, 25), normal(3063, 20, 0, 0.5), alternative="less")
    add(m, "median_test", "two", normal(3064, 20), normal(3065, 25, 0.8))
    add(m, "median_test", "two-no-correction", normal(3064, 20), normal(3065, 25, 0.8), correction=False)
    add(m, "median_test", "log-likelihood", normal(144, 15), normal(145, 18, 0.7), normal(146, 12, -0.4),
        lambda_="log-likelihood")  # fmt: skip
    add(m, "median_test", "ties-ignore", rounded(147, 20), rounded(148, 22, loc=1), ties="ignore")
    add(m, "median_test", "ties-below", rounded(147, 20), rounded(148, 22, loc=1), rounded(3066, 15, loc=-1))
    add(m, "median_test", "all-tied", np.full(3, 1.0), np.full(4, 1.0), tags=("degenerate",))
    add(m, "page_trend_test", "exact-12x8", normal(3067, (12, 8)) + np.arange(8) * 0.15, tags=("exact",))
    add(m, "page_trend_test", "exact-20x3", normal(3068, (20, 3)) + np.arange(3) * 0.2, tags=("exact",))
    add(m, "page_trend_test", "ties", np.round(normal(3069, (6, 5)) + np.arange(5) * 0.4), tags=("ties", "exact"))
    add(m, "page_trend_test", "asymptotic-forced", normal(149, (5, 3)) + np.arange(3) * 0.5, method="asymptotic")
    add(m, "page_trend_test", "predicted", normal(3070, (7, 4)) + np.array([0.6, 0.0, 0.3, 0.9]),
        predicted_ranks=np.array([3, 1, 2, 4]))  # fmt: skip
    gr = _g(3071)
    ranks = np.array([gr.permutation(4) + 1.0 for _ in range(9)])
    add(m, "page_trend_test", "ranked", ranks, ranked=True)
    add(m, "page_trend_test", "tail", normal(3072, (30, 6)) + np.arange(6) * 1.0, tags=("tail",))


# ---- H4: goodness of fit between samples --------------------------------------------------


def _h4():
    m = "H4"
    add(m, "ks_2samp", "exact-8-10", normal(160, 8), normal(161, 10, 0.8), tags=("small", "exact"))
    add(m, "ks_2samp", "exact-100-120", normal(162, 100), normal(163, 120, 0.3), tags=("exact",))
    add(m, "ks_2samp", "exact-less", normal(162, 100), normal(163, 120, 0.3), alternative="less", tags=("exact",))
    add(m, "ks_2samp", "exact-greater", normal(162, 100), normal(163, 120, 0.3), alternative="greater")
    add(m, "ks_2samp", "ties", rounded(164, 40), rounded(165, 50, loc=1), tags=("ties",))
    add(m, "ks_2samp", "tail", normal(166, 300), normal(167, 300, 1.5), tags=("tail",))
    add(m, "ks_2samp", "asymp", normal(168, 3000), normal(169, 2500, 0.05), method="asymp", tags=("large",))
    add(m, "ks_2samp", "batch", normal(170, (4, 15)), normal(171, (4, 12), 0.5), axis=-1, tags=("batch",))
    add(m, "cramervonmises_2samp", "exact", normal(172, 8), normal(173, 10, 0.7), tags=("exact",))
    add(m, "cramervonmises_2samp", "asymptotic", normal(174, 30), normal(175, 40, 0.4))
    s2 = [normal(176, 30), normal(177, 40, 0.5)]
    for v in ("midrank", "right", "continuous"):
        add(m, "anderson_ksamp", v, s2, variant=v)
    add(m, "anderson_ksamp", "three", [normal(178, 20), normal(179, 25, 0.3), rounded(180, 30)], variant="midrank")
    add(m, "epps_singleton_2samp", "default", normal(181, 30), normal(182, 40, 0.4, 1.5))
    # added with H4: ties (the exact distributions conditional on them, H55), the 'auto'
    # thresholds, equal sizes (Hodges' closed form), deep exact tails, the asymptotic forms, sizes
    # past scipy's exact Cramér-von Mises, offsets, degenerate input and batches
    ex, ey = np.array([1.0, 2, 2, 3, 3, 3]), np.array([2.0, 3, 4, 4, 5])
    add(m, "ks_2samp", "ties-small", ex, ey, tags=("ties", "small", "exact"))
    add(m, "ks_2samp", "ties-small-greater", ex, ey, alternative="greater", tags=("ties", "small", "exact"))
    add(m, "ks_2samp", "ties-small-less", ey, ex, alternative="less", tags=("ties", "small", "exact"))
    add(m, "ks_2samp", "ties-greater", rounded(4001, 40), rounded(4002, 50, loc=1), alternative="greater",
        tags=("ties",))  # fmt: skip
    add(m, "ks_2samp", "ties-few-values", ints(4003, 300, 4), ints(4004, 250, 5), tags=("ties", "large"))
    add(m, "ks_2samp", "ties-large", rounded(4005, 2000, step=0.1), rounded(4006, 1500, step=0.1, loc=0.05),
        tags=("ties", "large"))  # fmt: skip
    add(m, "ks_2samp", "equal-50", normal(4007, 50), normal(4008, 50, 0.5), tags=("exact",))
    add(m, "ks_2samp", "equal-50-greater", normal(4007, 50), normal(4008, 50, 0.5), alternative="greater")
    add(m, "ks_2samp", "coprime-997-1000", normal(4009, 997), normal(4010, 1000, 0.1), tags=("large", "exact"))
    add(m, "ks_2samp", "auto-large", normal(4011, 2000), normal(4012, 1500, 0.08), tags=("large", "exact"))
    add(m, "ks_2samp", "separated", normal(4013, 200), normal(4014, 250, 20.0), tags=("tail", "exact"))
    add(m, "ks_2samp", "separated-greater", normal(4013, 200), normal(4014, 250, 20.0), alternative="greater",
        tags=("tail", "exact"))  # fmt: skip
    add(m, "ks_2samp", "asymp-less", normal(168, 3000), normal(169, 2500, 0.05), method="asymp",
        alternative="less", tags=("large",))  # fmt: skip
    add(m, "ks_2samp", "asymp-small", normal(4015, 20), normal(4016, 30, 0.6), method="asymp")
    add(m, "ks_2samp", "asymp-auto", normal(4017, 12000), normal(4018, 1100, 0.1), tags=("large",))
    add(m, "ks_2samp", "offset", normal(4019, 60, 1e8), normal(4020, 70, 1e8 + 0.4), tags=("offset",))
    add(m, "ks_2samp", "constant", np.full(5, 2.0), np.full(4, 2.0), tags=("degenerate",))
    add(m, "ks_2samp", "empty", np.zeros(0), normal(4021, 5), tags=("degenerate",))
    add(m, "ks_2samp", "batch-ties", rounded(4022, (3, 20)), rounded(4023, (3, 25), loc=0.5), axis=-1,
        tags=("batch", "ties"))  # fmt: skip
    add(m, "cramervonmises_2samp", "exact-20", normal(4024, 20), normal(4025, 20, 0.6), tags=("exact",))
    add(m, "cramervonmises_2samp", "auto-21", normal(4026, 21), normal(4027, 15, 0.5))
    add(m, "cramervonmises_2samp", "exact-forced", normal(4028, 30), normal(4029, 26, 0.4), method="exact",
        tags=("exact",))  # fmt: skip
    add(m, "cramervonmises_2samp", "exact-forced-coprime", normal(4030, 23), normal(4031, 19, 0.7),
        method="exact", tags=("exact",))  # fmt: skip
    add(m, "cramervonmises_2samp", "ties-exact", rounded(4032, 12), rounded(4033, 15, loc=1.0), method="exact",
        tags=("ties", "exact"))  # fmt: skip
    add(m, "cramervonmises_2samp", "ties-auto", ints(4034, 9, 4), ints(4035, 11, 5), tags=("ties", "small"))
    add(m, "cramervonmises_2samp", "separated", normal(4036, 10), normal(4037, 12, 30.0), tags=("tail", "exact"))
    add(m, "cramervonmises_2samp", "tail", normal(4038, 300), normal(4039, 300, 1.0), tags=("tail",))
    add(m, "cramervonmises_2samp", "large", normal(4040, 2000), normal(4041, 3000, 0.05), tags=("large",))
    add(m, "cramervonmises_2samp", "asymptotic-small", normal(172, 8), normal(173, 10, 0.7), method="asymptotic")
    add(m, "cramervonmises_2samp", "one", np.array([0.5]), normal(4042, 6), tags=("degenerate",))
    add(m, "cramervonmises_2samp", "batch", normal(4043, (3, 12)), normal(4044, (3, 9), 0.5), axis=-1,
        tags=("batch",))  # fmt: skip
    t3 = [rounded(4045, 25), rounded(4046, 30, loc=0.5), rounded(4047, 20, loc=-0.5)]
    for v in ("midrank", "right", "continuous"):
        add(m, "anderson_ksamp", f"ties-{v}", t3, variant=v, tags=("ties",))
    add(m, "anderson_ksamp", "small", [normal(4048, 5), normal(4049, 5, 0.8)], variant="midrank", tags=("small",))
    add(m, "anderson_ksamp", "capped-high", [normal(4050, 40), normal(4051, 40, 3.0)], variant="midrank",
        tags=("tail",))  # fmt: skip
    add(m, "anderson_ksamp", "capped-low", [normal(4052, 40), normal(4052, 40) + 1e-9], variant="midrank")
    add(m, "anderson_ksamp", "four-large", [normal(4053 + i, 1000 + 250 * i, 0.02 * i) for i in range(4)],
        variant="midrank", tags=("large",))  # fmt: skip
    add(m, "anderson_ksamp", "offset", [normal(4057, 30, 1e8), normal(4058, 35, 1e8 + 0.6)], variant="midrank",
        tags=("offset",))  # fmt: skip
    add(m, "anderson_ksamp", "constant", [np.full(5, 1.0), np.full(6, 1.0)], variant="midrank",
        tags=("degenerate",))  # fmt: skip
    add(m, "epps_singleton_2samp", "small", normal(4059, 8), normal(4060, 10, 0.9), tags=("small",))
    add(m, "epps_singleton_2samp", "large", normal(4061, 600), normal(4062, 700, 0.1, 1.1), tags=("large",))
    add(m, "epps_singleton_2samp", "three-t", normal(4063, 40), normal(4064, 45, 0.3), t=(0.2, 0.5, 1.0))
    add(m, "epps_singleton_2samp", "tail", normal(4065, 200), normal(4066, 200, 2.0, 0.5), tags=("tail",))
    add(m, "epps_singleton_2samp", "offset", normal(4067, 40, 1e8), normal(4068, 45, 1e8 + 0.5), tags=("offset",))
    add(m, "epps_singleton_2samp", "four", normal(4069, 4), normal(4070, 30), tags=("degenerate",))
    add(m, "epps_singleton_2samp", "rank-deficient", ints(4071, 30, 2), ints(4072, 30, 2), tags=("degenerate",))
    add(m, "epps_singleton_2samp", "batch", normal(4073, (3, 20)), normal(4074, (3, 25), 0.5), axis=-1,
        tags=("batch",))  # fmt: skip


# ---- H5: correlation ----------------------------------------------------------------------


def _h5():
    m = "H5"
    x = normal(190, 10)
    add(m, "pearsonr", "small", x, 0.5 * x + normal(191, 10))
    add(m, "pearsonr", "small-greater", x, 0.5 * x + normal(191, 10), alternative="greater")
    add(m, "pearsonr", "large", normal(192, 1000), normal(193, 1000) + 0.1 * normal(192, 1000), tags=("large",))
    for k in (4, 6, 7):
        add(m, "pearsonr", f"near-1-eps{k}", x, x + 10.0**-k * normal(194, 10), tags=("tail",))
    add(m, "pearsonr", "offset", x + 1e8, x * 2 + normal(195, 10) + 5e7, tags=("offset",))
    add(m, "pearsonr", "batch", normal(196, (5, 20)), normal(197, (5, 20)), axis=-1, tags=("batch",))
    add(m, "spearmanrho", "small", x, x**3 + normal(198, 10))
    add(m, "spearmanrho", "ties", rounded(199, 30), rounded(200, 30), tags=("ties",))
    add(m, "spearmanrho", "large", normal(201, 1000), normal(202, 1000), tags=("large",))
    add(m, "spearmanr", "small", x, x**3 + normal(198, 10))
    add(m, "pointbiserialr", "small", ints(203, 20, 2), normal(204, 20))
    add(m, "kendalltau", "exact-10", x, x + normal(205, 10), tags=("exact",))
    add(m, "kendalltau", "exact-30", normal(206, 30), normal(207, 30), tags=("exact",))
    add(m, "kendalltau", "exact-less", normal(206, 30), normal(207, 30), alternative="less", tags=("exact",))
    add(m, "kendalltau", "ties", rounded(208, 40), rounded(209, 40), tags=("ties",))
    add(m, "kendalltau", "variant-c", rounded(208, 40), rounded(209, 40), variant="c")
    add(m, "kendalltau", "large", normal(210, 1000), normal(211, 1000), tags=("large",))
    add(m, "kendalltau", "batch", normal(212, (4, 12)), normal(213, (4, 12)), axis=-1, tags=("batch",))
    add(m, "weightedtau", "small", normal(214, 20), normal(215, 20))
    add(m, "somersd", "ties", rounded(216, 25), rounded(217, 25))
    add(m, "chatterjeexi", "small", normal(218, 20), normal(219, 20))
    add(m, "chatterjeexi", "large", normal(220, 500), np.sin(normal(220, 500)) + 0.1 * normal(221, 500))


# ---- H6: contingency tables ---------------------------------------------------------------


def _h6():
    m = "H6"
    t22 = np.array([[3.0, 7], [9, 2]])
    add(m, "chi2_contingency", "2x2", t22)
    add(m, "chi2_contingency", "2x2-no-correction", t22, correction=False)
    add(m, "chi2_contingency", "3x4", np.array([[10.0, 12, 8, 5], [7, 15, 9, 11], [4, 6, 14, 13]]))
    add(m, "chi2_contingency", "log-likelihood", np.array([[10.0, 12, 8], [7, 15, 9]]), lambda_="log-likelihood")
    add(m, "chi2_contingency", "large", np.array([[5000.0, 4000], [3000, 6000]]), tags=("tail",))
    for name, t in {
        "small": t22,
        "zero": np.array([[0.0, 5], [6, 2]]),
        "large": np.array([[100.0, 50], [30, 120]]),
        "balanced": np.array([[5.0, 5], [5, 5]]),
    }.items():
        add(m, "fisher_exact", name, t)
    add(m, "fisher_exact", "less", t22, alternative="less")
    add(m, "fisher_exact", "greater", t22, alternative="greater")
    add(m, "fisher_exact", "tail", np.array([[300.0, 10], [10, 300]]), tags=("tail",))
    add(m, "fisher_exact", "3x3", np.array([[3.0, 1, 4], [1, 5, 9], [2, 6, 5]]), tags=("rxc",))
    for t in ("small", "large"):
        tab = t22 if t == "small" else np.array([[20.0, 12], [8, 25]])
        add(m, "barnard_exact", t, tab)
        add(m, "boschloo_exact", t, tab)
    add(m, "barnard_exact", "unpooled", t22, pooled=False)
    add(m, "barnard_exact", "less", t22, alternative="less")
    mc = np.array([[30.0, 12], [4, 25]])
    add(m, "mcnemar", "exact", mc)
    add(m, "mcnemar", "chi2", mc, exact=False)
    add(m, "mcnemar", "chi2-no-correction", mc, exact=False, correction=False)
    add(m, "cochrans_q", "small", ints(230, (20, 3), 2))
    add(m, "odds_ratio", "conditional", t22.astype(np.int64))
    add(m, "odds_ratio", "sample", t22.astype(np.int64), kind="sample")
    add(m, "relative_risk", "small", 9, 20, 4, 21)


# ---- H7: proportions and binomial ---------------------------------------------------------


def _h7():
    m = "H7"
    add(m, "binomtest", "3-10", 3, 10)
    add(m, "binomtest", "7-20-p0.3", 7, 20, p=0.3)
    add(m, "binomtest", "0-10", 0, 10)
    add(m, "binomtest", "10-10", 10, 10)
    add(m, "binomtest", "less", 7, 20, p=0.3, alternative="less")
    add(m, "binomtest", "greater", 7, 20, p=0.3, alternative="greater")
    add(m, "binomtest", "large", 50300, 100000)
    add(m, "binomtest", "tail", 990, 1000, tags=("tail",))
    add(m, "binomtest", "p-small", 5, 1000, p=0.001)
    add(m, "quantile_test", "small", normal(240, 20), q=0.3, p=0.5)
    add(m, "poisson_means_test", "small", 10, 5.0, 20, 6.0)
    add(m, "poisson_means_test", "zero", 0, 1.0, 3, 1.0)
    add(m, "proportions_ztest", "one", 15, 40, value=0.5)
    add(m, "proportions_ztest", "two", np.array([15.0, 25]), np.array([40.0, 45]))
    add(m, "proportions_ztest", "two-smaller", np.array([15.0, 25]), np.array([40.0, 45]), alternative="smaller")
    add(m, "proportions_chisquare", "two", np.array([15.0, 25]), np.array([40.0, 45]))
    for meth in ("normal", "agresti_coull", "beta", "wilson", "jeffreys", "binom_test"):
        add(m, "proportion_confint", meth, 7, 30, method=meth)
        add(m, "proportion_confint", f"{meth}-zero", 0, 30, method=meth)
    for comp, meths in {
        "diff": ("wald", "agresti-caffo", "score"),
        "ratio": ("log", "log-adjusted", "score"),
        "odds-ratio": ("logit", "logit-adjusted", "logit-smoothed", "score"),
    }.items():
        add(m, "test_proportions_2indep", f"{comp}-default", 15, 40, 25, 45, compare=comp)
        for meth in meths:
            add(m, "test_proportions_2indep", f"{comp}-{meth}", 15, 40, 25, 45, compare=comp, method=meth)
    for comp, meths in {
        "diff": ("wald", "agresti-caffo", "newcomb", "score"),
        "ratio": ("log", "log-adjusted", "score"),
        "odds-ratio": ("logit", "logit-adjusted", "score"),
    }.items():
        add(m, "confint_proportions_2indep", f"{comp}-default", 15, 40, 25, 45, compare=comp)
        for meth in meths:
            add(m, "confint_proportions_2indep", f"{comp}-{meth}", 15, 40, 25, 45, compare=comp, method=meth)


# ---- H8: multiple comparisons and corrections ---------------------------------------------

PVALS = np.array(
    [1e-300, 3e-17, 1e-10, 2e-6, 0.0004, 0.001, 0.003, 0.009, 0.012, 0.02, 0.03, 0.041, 0.05, 0.07, 0.2, 0.3,
     0.45, 0.6, 0.85, 1.0]
)  # fmt: skip
MULTITEST = ("bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel", "fdr_bh", "fdr_by")
MULTITEST_TWO_STAGE = ("fdr_tsbh", "fdr_tsbky")


def _h8():
    m = "H8"
    g = [normal(250, 10), normal(251, 12, 0.8), normal(252, 9, 0.3)]
    add(m, "tukey_hsd", "three", *g)
    add(m, "tukey_hsd", "games-howell", normal(253, 10), normal(254, 12, 0.8, 3), normal(255, 9), equal_var=False)
    add(m, "dunnett", "two", normal(256, 10, 0.5), normal(257, 12, 1.0), control=normal(258, 11))
    for meth in MULTITEST + MULTITEST_TWO_STAGE:
        add(m, "multipletests", meth, PVALS, method=meth)
    add(m, "multipletests", "hommel-shuffled", _g(259).permutation(PVALS), method="hommel")
    add(m, "multipletests", "sidak-alpha", PVALS, method="sidak", alpha=0.1)
    for meth in ("bh", "by"):
        add(m, "false_discovery_control", meth, PVALS, method=meth)
    for meth in ("fisher", "pearson", "tippett", "stouffer", "mudholkar_george"):
        add(m, "combine_pvalues", meth, PVALS[:12], method=meth)
    add(m, "combine_pvalues", "fisher-small", np.array([0.2, 0.3, 0.6]))
    add(m, "combine_pvalues", "stouffer-weights", np.array([0.01, 0.2, 0.3]), method="stouffer",
        weights=np.array([1.0, 2.0, 0.5]))  # fmt: skip


for _f in (_h1, _h2, _h3, _h4, _h5, _h6, _h7, _h8):
    _f()
del _f

STATSMODELS = {
    "multipletests",
    "proportions_ztest",
    "proportions_chisquare",
    "proportion_confint",
    "test_proportions_2indep",
    "confint_proportions_2indep",
    "mcnemar",
    "cochrans_q",
}


def by_func(func):
    return [p for p in PROBLEMS.values() if p.func == func]
