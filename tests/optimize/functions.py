"""scipy's standard test functions for sd.optimize (docs/design/optimize.md §7), written for a
namespace M of elementary functions (ad.FLOAT, ad.ARB, ad.AD) like mgh.py:

  - minimize: Rosenbrock (scipy.optimize.rosen, rosen_der, rosen_hess, rosen_hess_prod), the
    maximum-entropy problem of scipy's test_optimize.CheckOptimize (Berger et al. 1996), and the
    bound-constrained problem of L-BFGS-B 3.0's driver1 (n = 25);
  - root_scalar and find_root: the 'original' functions f1-f5 of scipy/optimize/_tstutils.py,
    the 154 cases of Alefeld, Potra and Shi (TOMS 748, 1995) and the 45 cases of Chandrupatla
    (1997), with scipy's brackets, starting points and roots;
  - minimize_scalar and find_minimum: scipy's (x - 1.5)^2 - 0.8 and the 55 cases of
    Chandrupatla's minimization paper (1998) with its bracketing procedure;
  - lsq_linear and nnls: the dense problems of scipy's test_lsq_linear and test_nnls and three
    deterministic ones.

The functions are this library's own transcriptions; tests/optimize/test_problems.py checks
them against scipy's and every derivative against Arb. Three second derivatives of scipy's
_tstutils are wrong and are corrected here: aps02_fpp (its sign), aps09_fpp (-12 n (1 - n x)^2
for -12 n^2 (1 - n x)^2) and aps10_fpp (-n x for -n in the exponential's factor).
"""

import math

from .ad import FLOAT


# ---- multivariate


def rosen(x, M=FLOAT):
    """sum 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 (scipy.optimize.rosen)."""
    s = 0.0
    for i in range(len(x) - 1):
        a, b = x[i + 1] - x[i] * x[i], 1.0 - x[i]
        s = s + 100.0 * (a * a) + b * b
    return s


def rosen_der(x, M=FLOAT):
    n = len(x)
    g = [0.0] * n
    for i in range(1, n - 1):
        g[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1]) - 400.0 * (x[i + 1] - x[i] * x[i]) * x[i] - 2.0 * (1.0 - x[i])
    g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0])
    g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2])
    return g


def rosen_hess(x, M=FLOAT):
    n = len(x)
    h = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        h[i][i + 1] = h[i + 1][i] = -400.0 * x[i]
    h[0][0] = 1200.0 * (x[0] * x[0]) - 400.0 * x[1] + 2.0
    for i in range(1, n - 1):
        h[i][i] = 202.0 + 1200.0 * (x[i] * x[i]) - 400.0 * x[i + 1]
    h[n - 1][n - 1] = 200.0
    return h


def rosen_hess_prod(x, p, M=FLOAT):
    h = rosen_hess(x, M)
    return [sum(h[i][j] * p[j] for j in range(len(x))) for i in range(len(x))]


ROSEN_X0 = {2: [-1.2, 1.0], 5: [1.3, 0.7, 0.8, 1.9, 1.2]}  # scipy's tests start from these

# The maximum-entropy problem (scipy test_optimize.CheckOptimize): f(x) = log sum exp(F x) - K x
MAXENT_F = ((1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
MAXENT_K = (1.0, 0.3, 0.5)
MAXENT_X0 = [0.0, 0.0, 0.0]
MAXENT_SOLUTION = [0.0, -0.524869316, 0.487525860]  # scipy's; the problem is invariant in x[0]


def maxent(x, M=FLOAT):
    s = 0.0
    for row in MAXENT_F:
        d = 0.0
        for a, v in zip(row, x):
            d = d + a * v
        s = s + M.exp(d)
    kx = 0.0
    for k, v in zip(MAXENT_K, x):
        kx = kx + k * v
    return M.log(s) - kx


def maxent_der(x, M=FLOAT):
    es = []
    for row in MAXENT_F:
        d = 0.0
        for a, v in zip(row, x):
            d = d + a * v
        es.append(M.exp(d))
    z = 0.0
    for e in es:
        z = z + e
    return [sum(MAXENT_F[i][j] * es[i] for i in range(5)) / z - MAXENT_K[j] for j in range(3)]


# L-BFGS-B 3.0, driver1.f: f = 0.25 (x1 - 1)^2 + 4 sum (x_i - x_{i-1}^2)^2, n = 25,
# 1 <= x_i <= 100 for odd i and -100 <= x_i <= 100 for even i (1-based), x0 = 3
LBFGSB_DRIVER1_N = 25
LBFGSB_DRIVER1_X0 = [3.0] * 25
LBFGSB_DRIVER1_BOUNDS = [((1.0, 100.0) if i % 2 == 0 else (-100.0, 100.0)) for i in range(25)]


def lbfgsb_driver1(x, M=FLOAT):
    # driver1.f's operations, squares as products (gfortran's x**2)
    f = 0.25 * (x[0] - 1.0) * (x[0] - 1.0)
    for i in range(1, len(x)):
        d = x[i] - x[i - 1] * x[i - 1]
        f = f + d * d
    return 4.0 * f


def lbfgsb_driver1_der(x, M=FLOAT):
    n = len(x)
    g = [0.0] * n
    t1 = x[1] - x[0] * x[0]
    g[0] = 2.0 * (x[0] - 1.0) - 16.0 * x[0] * t1
    for i in range(1, n - 1):
        t2 = t1
        t1 = x[i + 1] - x[i] * x[i]
        g[i] = 8.0 * t2 - 16.0 * x[i] * t1
    g[n - 1] = 8.0 * t1
    return g


# ---- scalar roots: the 'original' functions of scipy's _tstutils


def orig1(x, M=FLOAT):
    return x * (x - 1.0)


def orig1_d(x, M=FLOAT):
    return 2.0 * x - 1.0


def orig1_dd(x, M=FLOAT):
    return 2.0


def orig2(x, M=FLOAT):
    return x * x - 1.0


def orig2_d(x, M=FLOAT):
    return 2.0 * x


def orig2_dd(x, M=FLOAT):
    return 2.0


def orig3(x, M=FLOAT):
    return x * (x - 1.0) * (x - 2.0) * (x - 3.0)


def orig3_d(x, M=FLOAT):
    return 4.0 * x**3 - 18.0 * x**2 + 22.0 * x - 6.0


def orig3_dd(x, M=FLOAT):
    return 12.0 * x**2 - 36.0 * x + 22.0


def orig4(x, M=FLOAT):
    """Piecewise linear, discontinuous at the root 1."""
    if x > 1:
        return 1.0 + 0.1 * x
    if x < 1:
        return -1.0 + 0.1 * x
    return 0.0


def orig5(x, M=FLOAT):
    """1 / (1 - x) with the pole at 1 replaced by 0."""
    if x != 1:
        return 1.0 / (1.0 - x)
    return 0.0


class RootCase:
    """A scalar root problem: id, f(x, M), fprime, fprime2 (or None), bracket, x0, root."""

    __slots__ = ("id", "f", "fprime", "fprime2", "bracket", "x0", "root", "smooth", "nfev_paper")

    def __init__(self, id, f, fprime, fprime2, bracket, x0, root, smooth=True, nfev_paper=None):
        self.id, self.f, self.fprime, self.fprime2 = id, f, fprime, fprime2
        self.bracket, self.x0, self.root, self.smooth = tuple(map(float, bracket)), x0, root, smooth
        self.nfev_paper = nfev_paper  # Chandrupatla's count of function values

    def __repr__(self):
        return self.id


_SQRT3 = math.sqrt(3.0)
ORIGINAL = [
    RootCase("original.01", orig1, orig1_d, orig1_dd, (0.5, _SQRT3), 0.6, 1.0),
    RootCase("original.02", orig2, orig2_d, orig2_dd, (0.5, _SQRT3), 0.6, 1.0),
    RootCase("original.03", orig3, orig3_d, orig3_dd, (0.5, _SQRT3), 0.6, 1.0),
    RootCase("original.04", orig4, None, None, (0.5, _SQRT3), 0.6, 1.0, smooth=False),
    RootCase("original.05", orig5, None, None, (0.5, _SQRT3), 0.6, 1.0, smooth=False),
]


# ---- Alefeld, Potra and Shi (1995): fifteen families, with first and second derivatives


def _aps01(x, M):
    return M.sin(x) - x / 2.0


def _aps01d(x, M):
    return M.cos(x) - 0.5


def _aps01dd(x, M):
    return -M.sin(x)


def _aps02(k):
    def f(x, M, k=k):
        s = 0.0
        for i in range(1, 21):
            s = s + (2 * i - 5) ** 2 / (x - i * i) ** k
        return s

    return f


_APS02 = (_aps02(3), _aps02(4), _aps02(5))


def _aps02f(x, M):
    return -2.0 * _APS02[0](x, M)


def _aps02d(x, M):
    return 6.0 * _APS02[1](x, M)


def _aps02dd(x, M):
    # scipy's _tstutils.aps02_fpp has +24 (the sign is wrong)
    return -24.0 * _APS02[2](x, M)


def _p(f):
    """Bind a family's parameters: f(x, M, *params) -> f(x, M)."""
    return lambda *params: lambda x, M=FLOAT: f(x, M, *params)


aps03 = _p(lambda x, M, a, b: a * x * M.exp(b * x))
aps03d = _p(lambda x, M, a, b: a * (b * x + 1.0) * M.exp(b * x))
aps03dd = _p(lambda x, M, a, b: a * (b * (b * x + 1.0) + b) * M.exp(b * x))
aps04 = _p(lambda x, M, n, a: x**n - a)
aps04d = _p(lambda x, M, n, a: n * x ** (n - 1))
aps04dd = _p(lambda x, M, n, a: n * (n - 1) * x ** (n - 2))
aps06 = _p(lambda x, M, n: 2.0 * x * M.exp(-n) - 2.0 * M.exp(-n * x) + 1.0)
aps06d = _p(lambda x, M, n: 2.0 * M.exp(-n) + 2.0 * n * M.exp(-n * x))
aps06dd = _p(lambda x, M, n: -2.0 * n * n * M.exp(-n * x))
aps07 = _p(lambda x, M, n: (1.0 + (1 - n) ** 2) * x - (1.0 - n * x) ** 2)
aps07d = _p(lambda x, M, n: (1.0 + (1 - n) ** 2) + 2.0 * n * (1.0 - n * x))
aps07dd = _p(lambda x, M, n: -2.0 * n * n)
aps08 = _p(lambda x, M, n: x * x - (1.0 - x) ** n)
aps08d = _p(lambda x, M, n: 2.0 * x + n * (1.0 - x) ** (n - 1))
aps08dd = _p(lambda x, M, n: 2.0 - n * (n - 1) * (1.0 - x) ** (n - 2))
aps09 = _p(lambda x, M, n: (1.0 + (1 - n) ** 4) * x - (1.0 - n * x) ** 4)
aps09d = _p(lambda x, M, n: (1.0 + (1 - n) ** 4) + 4.0 * n * (1.0 - n * x) ** 3)
aps09dd = _p(lambda x, M, n: -12.0 * n * n * (1.0 - n * x) ** 2)  # scipy's aps09_fpp has one n
aps10 = _p(lambda x, M, n: M.exp(-n * x) * (x - 1.0) + x**n)
aps10d = _p(lambda x, M, n: M.exp(-n * x) * (-n * (x - 1.0) + 1.0) + n * x ** (n - 1))
aps10dd = _p(
    lambda x, M, n: M.exp(-n * x) * (-n * (-n * (x - 1.0) + 1.0) - n) + n * (n - 1) * x ** (n - 2)
)  # scipy: -n x
aps11 = _p(lambda x, M, n: (n * x - 1.0) / ((n - 1) * x))
aps11d = _p(lambda x, M, n: 1.0 / (n - 1) / x**2)
aps11dd = _p(lambda x, M, n: -2.0 / (n - 1) / x**3)
aps12 = _p(lambda x, M, n: x ** (1.0 / n) - float(n) ** (1.0 / n))
aps12d = _p(lambda x, M, n: x ** ((1.0 - n) / n) / n)
aps12dd = _p(lambda x, M, n: x ** ((1.0 - 2 * n) / n) * (1.0 / n) * (1.0 - n) / n)

_MAX_EXPABLE = math.log(1.7976931348623157e308)


def _aps13(x, M=FLOAT):
    """x exp(-1/x^2): every derivative is 0 at the root 0 (0 where 1/x^2 overflows exp)."""
    if x == 0:
        return 0.0
    y = 1.0 / x**2
    if y > _MAX_EXPABLE:
        return 0.0
    return x / M.exp(y)


def _aps13d(x, M=FLOAT):
    if x == 0:
        return 0.0
    y = 1.0 / x**2
    if y > _MAX_EXPABLE:
        return 0.0
    return (1.0 + 2.0 / x**2) / M.exp(y)


def _aps13dd(x, M=FLOAT):
    if x == 0:
        return 0.0
    y = 1.0 / x**2
    if y > _MAX_EXPABLE:
        return 0.0
    return 2.0 * (2.0 - x**2) / x**5 / M.exp(y)


def _aps14(x, M, n):
    if x <= 0:
        return -n / 20.0
    return n / 20.0 * (x / 1.5 + M.sin(x) - 1.0)


def _aps14d(x, M, n):
    if x <= 0:
        return 0.0
    return n / 20.0 * (1.0 / 1.5 + M.cos(x))


def _aps14dd(x, M, n):
    if x <= 0:
        return 0.0
    return -n / 20.0 * M.sin(x)


def _aps15(x, M, n):
    if x < 0:
        return -0.859
    if x > 2 * 1e-3 / (1 + n):
        return math.e - 1.859
    return M.exp((n + 1) * x / 2 * 1000) - 1.859


def _aps15d(x, M, n):
    if not 0 <= x <= 2 * 1e-3 / (1 + n):
        return math.e - 1.859
    return M.exp((n + 1) * x / 2 * 1000) * (n + 1) / 2 * 1000


def _aps15dd(x, M, n):
    if not 0 <= x <= 2 * 1e-3 / (1 + n):
        return math.e - 1.859
    return M.exp((n + 1) * x / 2 * 1000) * (n + 1) / 2 * 1000 * (n + 1) / 2 * 1000


def _aps_cases():
    pi = math.pi
    out = []

    def add(fam, k, f, d, dd, bracket, x0, root, smooth=True):
        out.append(RootCase(f"aps.{fam:02d}.{k:02d}", f, d, dd, bracket, x0, root, smooth))

    add(
        1,
        0,
        lambda x, M=FLOAT: _aps01(x, M),
        lambda x, M=FLOAT: _aps01d(x, M),
        lambda x, M=FLOAT: _aps01dd(x, M),
        (pi / 2, pi),
        3.0,
        1.89549426703398094e00,
    )
    roots02 = (
        3.02291534727305677e00,
        6.68375356080807848e00,
        1.12387016550022114e01,
        1.96760000806234103e01,
        2.98282273265047557e01,
        4.19061161952894139e01,
        5.59535958001430913e01,
        7.19856655865877997e01,
        9.00088685391666701e01,
        1.10026532748330197e02,
    )
    x02 = (2, 5, 10, 17, 26, 37, 50, 65, 82, 101)
    for k in range(10):
        a, b = (k + 1) ** 2, (k + 2) ** 2
        add(
            2,
            k,
            lambda x, M=FLOAT: _aps02f(x, M),
            lambda x, M=FLOAT: _aps02d(x, M),
            lambda x, M=FLOAT: _aps02dd(x, M),
            (a + 1e-9, b - 1e-9),
            float(x02[k]),
            roots02[k],
        )
    for k, (a, b) in enumerate(((-40, -1), (-100, -2), (-200, -3))):
        add(3, k, aps03(a, b), aps03d(a, b), aps03dd(a, b), (-9, 31), -2.0, 0.0)
    roots04 = (
        6.68740304976422006e-01,
        7.64724491331730039e-01,
        8.17765433957942545e-01,
        8.51339922520784609e-01,
        8.74485272221167897e-01,
    )
    k = 0
    for n, r in zip((4, 6, 8, 10, 12), roots04):
        add(4, k, aps04(n, 0.2), aps04d(n, 0.2), aps04dd(n, 0.2), (0, 5), 2.5, r)
        k += 1
    for n in (4, 6, 8, 10, 12):
        add(4, k, aps04(n, 1), aps04d(n, 1), aps04dd(n, 1), (0, 5), 2.5, 1.0)
        k += 1
    for n in (8, 10, 12, 14):
        add(4, k, aps04(n, 1), aps04d(n, 1), aps04dd(n, 1), (-0.95, 4.05), 1.5, 1.0)
        k += 1
    add(
        5,
        0,
        lambda x, M=FLOAT: M.sin(x) - 0.5,
        lambda x, M=FLOAT: M.cos(x),
        lambda x, M=FLOAT: -M.sin(x),
        (0, 1.5),
        1.3,
        pi / 6,
    )
    rows06 = (
        (1, 0.5, 4.22477709641236709e-01),
        (2, 0.5, 3.06699410483203705e-01),
        (3, 0.5, 2.23705457654662959e-01),
        (4, 0.5, 1.71719147519508369e-01),
        (5, 0.4, 1.38257155056824066e-01),
        (20, 0.1, 3.46573590208538521e-02),
        (40, 5e-02, 1.73286795139986315e-02),
        (60, 1.0 / 30, 1.15524530093324210e-02),
        (80, 2.5e-02, 8.66433975699931573e-03),
        (100, 2e-02, 6.93147180559945415e-03),
    )
    for k, (n, x0, r) in enumerate(rows06):
        add(6, k, aps06(n), aps06d(n), aps06dd(n), (0, 1), x0, r)
    for k, (n, r) in enumerate(
        ((5, 3.84025518406218985e-02), (10, 9.90000999800049949e-03), (20, 2.49375003906201174e-03))
    ):
        add(7, k, aps07(n), aps07d(n), aps07dd(n), (0, 1), 0.4, r)
    for k, (n, r) in enumerate(
        (
            (2, 0.5),
            (5, 3.45954815848242059e-01),
            (10, 2.45122333753307220e-01),
            (15, 1.95547623536565629e-01),
            (20, 1.64920957276440960e-01),
        )
    ):
        add(8, k, aps08(n), aps08d(n), aps08dd(n), (0, 1), 0.9, r)
    for k, (n, r) in enumerate(
        (
            (1, 2.75508040999484394e-01),
            (2, 1.37754020499742197e-01),
            (4, 1.03052837781564422e-02),
            (5, 3.61710817890406339e-03),
            (8, 4.10872918496395375e-04),
            (15, 2.59895758929076292e-05),
            (20, 7.66859512218533719e-06),
        )
    ):
        add(9, k, aps09(n), aps09d(n), aps09dd(n), (0, 1), 0.5, r)
    for k, (n, r) in enumerate(
        (
            (1, 4.01058137541547011e-01),
            (5, 5.16153518757933583e-01),
            (10, 5.39522226908415781e-01),
            (15, 5.48182294340655241e-01),
            (20, 5.52704666678487833e-01),
        )
    ):
        add(10, k, aps10(n), aps10d(n), aps10dd(n), (0, 1), 0.9, r)
    for k, n in enumerate((2, 5, 15, 20)):
        add(11, k, aps11(n), aps11d(n), aps11dd(n), (0.01, 1), 1e-02, 1.0 / n)
    for k, n in enumerate((2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33)):
        add(12, k, aps12(n), aps12d(n), aps12dd(n), (1, 100), 1.1, float(n))
    add(13, 0, _aps13, _aps13d, _aps13dd, (-1, 4), 1.5, 0.0)
    for k, n in enumerate(range(1, 41)):
        add(
            14,
            k,
            _p(_aps14)(n),
            _p(_aps14d)(n),
            _p(_aps14dd)(n),
            (-1000, pi / 2),
            1.0,
            6.23806518961612433e-01,
            smooth=False,
        )
    roots15 = (
        5.90513055942197166e-05,
        5.63671553399369967e-05,
        5.39164094555919196e-05,
        5.16698923949422470e-05,
        4.96030966991445609e-05,
        4.76952852876389951e-05,
        4.59287932399486662e-05,
        4.42884791956647841e-05,
        4.27612902578832391e-05,
        4.13359139159538030e-05,
        4.00024973380198076e-05,
        3.87524192962066869e-05,
        3.75781035599579910e-05,
        3.64728652199592355e-05,
        3.54307833565318273e-05,
        3.44465949299614980e-05,
        3.35156058778003705e-05,
        3.26336162494372125e-05,
        3.17968568584260013e-05,
        3.10019354369653455e-05,
        3.02457906702100968e-05,
        1.22779942324615231e-05,
        6.16953939044086617e-06,
        4.11985852982928163e-06,
        3.09246238772721682e-06,
        2.47520442610501789e-06,
        2.06335676785127107e-06,
        1.76901200781542651e-06,
        1.54816156988591016e-06,
        1.37633453660223511e-06,
        1.23883857889971403e-06,
    )
    ns15 = list(range(20, 41)) + list(range(100, 1001, 100))
    for k, (n, r) in enumerate(zip(ns15, roots15)):
        add(15, k, _p(_aps15)(n), _p(_aps15d)(n), _p(_aps15dd)(n), (-1000, 1e-4), -2.0, r, smooth=False)
    return out


APS = _aps_cases()


# ---- Chandrupatla (1997): nine functions, five brackets each, with the paper's counts


def ch1(x, M=FLOAT):
    return x**3 - 2.0 * x - 5.0


def ch2(x, M=FLOAT):
    return 1.0 - 1.0 / x**2


def ch3(x, M=FLOAT):
    return (x - 3.0) ** 3


def ch4(x, M=FLOAT):
    return 6.0 * (x - 2.0) ** 5


def ch5(x, M=FLOAT):
    return x**9


def ch6(x, M=FLOAT):
    return x**19


def ch7(x, M=FLOAT):
    return 0.0 if abs(x) < 3.8e-4 else x * M.exp(-(x ** (-2)))


def ch8(x, M=FLOAT):
    xi = 0.61489
    return -(3062.0 * (1.0 - xi) * M.exp(-x)) / (xi + (1.0 - xi) * M.exp(-x)) - 1013.0 + 1628.0 / x


def ch9(x, M=FLOAT):
    return M.exp(x) - 2.0 - 0.01 / x**2 + 0.000002 / x**3


_CH = [
    (ch1, 2.0945514815423265, [((2, 3), 7), ((1, 10), 11), ((1, 100), 14), ((-1e4, 1e4), 23), ((-1e10, 1e10), 43)]),
    (ch2, 1.0, [((0.5, 1.51), 8), ((1e-4, 1e4), 22), ((1e-6, 1e6), 28), ((1e-10, 1e10), 41), ((1e-12, 1e12), 48)]),
    (ch3, 3.0, [((0, 5), 21), ((-10, 10), 23), ((-1e4, 1e4), 36), ((-1e6, 1e6), 45), ((-1e10, 1e10), 55)]),
    (ch4, 2.0, [((0, 5), 21), ((-10, 10), 23), ((-1e4, 1e4), 33), ((-1e6, 1e6), 43), ((-1e10, 1e10), 54)]),
    (ch5, 0.0, [((-1, 4), 21), ((-2, 5), 22), ((-1, 10), 23), ((-5, 50), 25), ((-10, 100), 26)]),
    (ch6, 0.0, [((-1, 4), 21), ((-2, 5), 22), ((-1, 10), 23), ((-5, 50), 25), ((-10, 100), 26)]),
    (ch7, 0.0, [((-1, 4), 8), ((-2, 5), 8), ((-1, 10), 11), ((-5, 50), 18), ((-10, 100), 19)]),
    (ch8, 1.0375360332870405, [((2e-4, 2), 9), ((2e-4, 3), 10), ((2e-4, 9), 11), ((2e-4, 27), 12), ((2e-4, 81), 14)]),
    (ch9, 0.7032048403631358, [((2e-4, 1), 7), ((2e-4, 3), 8), ((2e-4, 9), 10), ((2e-4, 27), 11), ((2e-4, 81), 13)]),
]
CHANDRUPATLA = [
    RootCase(f"chandrupatla.{i + 1}.{k + 1}", f, None, None, br, None, root, smooth=(f is not ch7), nfev_paper=nf)
    for i, (f, root, rows) in enumerate(_CH)
    for k, (br, nf) in enumerate(rows)
]

ROOT_CASES = ORIGINAL + APS + CHANDRUPATLA
ROOTS_BY_ID = {c.id: c for c in ROOT_CASES}


# ---- scalar minimization


def scipy_scalar(x, M=FLOAT):
    """scipy test_optimize.TestOptimizeScalar: (x - 1.5)^2 - 0.8, minimum at 1.5."""
    return (x - 1.5) ** 2 - 0.8


def chm1(x, M=FLOAT):
    return 100.0 * (1.0 - x**3.0) ** 2 + (1.0 - x**2.0) + 2.0 * (1.0 - x) ** 2.0


def chm2(x, M=FLOAT):
    return 5.0 + (x - 2.0) ** 6


def chm3(x, M=FLOAT):
    return M.exp(x) - 5.0 * x


def chm4(x, M=FLOAT):
    return x**5.0 - 5.0 * x**3.0 - 20.0 * x + 5.0


def chm5(x, M=FLOAT):
    return 8.0 * x**3 - 2.0 * x**2 - 7.0 * x + 3.0


def chandrupatla_bracket(f, x1, x2):
    """The paper's bracketing procedure (scipy test_chandrupatla._bracket_minimum): steps
    growing by the golden ratio from (x1, x2) downhill until f rises."""
    phi = 1.61803398875
    f1, f2 = f(x1), f(x2)
    step = x2 - x1
    if f2 > f1:
        x1, x2, f1, f2, step = x2, x1, f2, f1, -step
    for _ in range(100):
        step *= phi
        x3 = x2 + step
        f3 = f(x3)
        if f3 < f2:
            x1, x2, f1, f2 = x2, x3, f2, f3
        else:
            break
    return x1, x2, x3


class MinCase:
    """A scalar minimization problem: id, f(x, M), bracket (three points), bounds, nit_paper."""

    __slots__ = ("id", "f", "bracket", "bounds", "nit_paper", "xmin")

    def __init__(self, id, f, bracket, bounds=None, nit_paper=None, xmin=None):
        self.id, self.f, self.bracket, self.bounds = id, f, bracket, bounds
        self.nit_paper, self.xmin = nit_paper, xmin

    def __repr__(self):
        return self.id


_CHM = [
    (
        chm1,
        [
            (-1, 11),
            (-2, 13),
            (-4, 13),
            (-8, 15),
            (-16, 16),
            (-32, 19),
            (-64, 20),
            (-128, 21),
            (-256, 21),
            (-512, 19),
            (-1024, 24),
        ],
    ),
    (
        chm2,
        [
            (-1, 8),
            (-2, 6),
            (-4, 6),
            (-8, 7),
            (-16, 8),
            (-32, 8),
            (-64, 9),
            (-128, 11),
            (-256, 13),
            (-512, 12),
            (-1024, 13),
        ],
    ),
    (
        chm3,
        [
            (-1, 11),
            (-2, 11),
            (-4, 11),
            (-8, 10),
            (-16, 14),
            (-32, 12),
            (-64, 15),
            (-128, 18),
            (-256, 18),
            (-512, 19),
            (-1024, 19),
        ],
    ),
    (
        chm4,
        [
            (-0.05, 9),
            (-0.10, 11),
            (-0.15, 11),
            (-0.20, 11),
            (-0.25, 11),
            (-0.30, 9),
            (-0.35, 9),
            (-0.40, 9),
            (-0.45, 10),
            (-0.50, 10),
            (-0.55, 10),
        ],
    ),
    (
        chm5,
        [
            (-0.05, 6),
            (-0.10, 7),
            (-0.15, 8),
            (-0.20, 10),
            (-0.25, 9),
            (-0.30, 8),
            (-0.35, 7),
            (-0.40, 7),
            (-0.45, 9),
            (-0.50, 9),
            (-0.55, 8),
        ],
    ),
]


def _min_cases():
    out = [
        MinCase("scipy.scalar.0", scipy_scalar, (-3.0, -2.0), bounds=(1.0, 5.0), xmin=1.5),
        MinCase("scipy.scalar.1", scipy_scalar, (-15.0, -1.0, 15.0), bounds=(0.0, 1.0), xmin=1.5),
    ]
    for i, (f, rows) in enumerate(_CHM):
        for k, (x1, nit) in enumerate(rows):
            br = chandrupatla_bracket(f, float(x1), float(x1) + 0.2)
            lo, hi = min(br[0], br[2]), max(br[0], br[2])
            out.append(
                MinCase(f"chandrupatla_min.{i + 1}.{k + 1}", f, tuple(sorted(br)), bounds=(lo, hi), nit_paper=nit)
            )
    return out


MIN_CASES = _min_cases()
MINS_BY_ID = {c.id: c for c in MIN_CASES}


# ---- linear least squares (lsq_linear, nnls): (A rows, b, bounds or None)


def _lsq_problems():
    inf = math.inf
    A1 = [[0.171, -0.057], [-0.049, -0.248], [-0.166, 0.054]]
    b1 = [0.074, 1.014, -0.383]
    out = {
        "scipy.dense": (A1, b1, None),
        "scipy.dense.box": (A1, b1, ([-1.0, -10.0], [1.0, 0.0])),
        "scipy.dense.lb0": (A1, b1, ([0.0, -inf], [inf, inf])),
        "scipy.dense.lbm1": (A1, b1, ([-1.0, 0.0], [inf, inf])),
        "scipy.dense.ubm5": (A1, b1, ([-inf, -inf], [inf, -5.0])),
        "scipy.nnls.3x3": ([[1.0, 0.5, -1.0], [1.0, 0.5, 0.0], [-1.0, 0.0, 1.0]], [-1.0, 2.0, 2.0], None),
        "scipy.nnls.arange": (
            [[float(5 * i + j) for j in range(5)] for i in range(5)],
            [float(sum((5 * i + j) * j for j in range(5))) for i in range(5)],
            None,
        ),
    }
    # deterministic dense problems: a smooth tall matrix, a Hilbert block, a wide one
    A = [[math.cos(0.7 * (i + 1) * (j + 1) + 0.3 * i) for j in range(8)] for i in range(30)]
    b = [math.sin(1.3 * i) + 0.1 * i for i in range(30)]
    out["cos.30x8"] = (A, b, None)
    out["cos.30x8.box"] = (A, b, ([-0.5] * 8, [0.5] * 8))
    H = [[1.0 / (i + j + 1) for j in range(6)] for i in range(12)]
    out["hilbert.12x6"] = (H, [1.0] * 12, None)
    out["hilbert.12x6.nonneg"] = (H, [1.0] * 12, ([0.0] * 6, [inf] * 6))
    W = [[math.sin((i + 1) * (j + 2) * 0.37) for j in range(12)] for i in range(8)]
    out["sin.8x12.box"] = (W, [math.cos(i) for i in range(8)], ([-1.0] * 12, [1.0] * 12))
    return out


LSQ_LINEAR = _lsq_problems()
