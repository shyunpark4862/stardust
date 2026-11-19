"""Reference values for the multivariate normal and t cdfs (T9, csrc/stats/mvcdf.hpp).

    .venv/bin/python tools/mvcdf_ref.py              # every set, on 10 processes
    .venv/bin/python tools/mvcdf_ref.py bvn tvn      # only these sets
    .venv/bin/python tools/mvcdf_ref.py --jobs=4 tvn

writes tests/stats/ref_mvcdf/<set>.npz. The inputs are drawn from a seeded generator, so a
rerun gives the same file.

    bvn       Phi2(h, k; r), rigorous (Arb), as a double-double hi + lo
    tvn       Phi3(h; R), rigorous (Arb), as hi + lo
    factor_n  one-factor normals, d = 4..10, rigorous (Arb) to 2^-64
    factor_t  one-factor t's, d = 2..8: a two-level adaptive quadrature (scipy), ~1e-11
    bvt, tvt  the t at d = 2, 3: the scale mixture through the bvn / tvn identities,
              rigorous (Arb) to 2^-64

The definitions are the textbook ones, never our code's arrangement of them:
  * bvn: Plackett's identity started at r = 0 (Drezner and Wesolowsky 1990; Genz 2004),
        Phi2 = Phi(h) Phi(k) + (1 / 2pi) int_0^asin r exp(-(h^2 - 2hk sin t + k^2) / (2 cos^2 t)) dt,
    which cancels for r < 0; the ball arithmetic carries the precision that needs.
  * tvn: Genz 2004 §3's path: keep r12, move r13 and r23 from 0 by the same factor,
        Phi3 = Phi2(h1, h2; r12) Phi(h3) + sum over (i, k) in {(1, 3), (2, 3)} of
               (1 / 2pi) int_0^asin r_ik exp(-q_ik) Phi(u_j) dt,
    u_j the standardized limit of the third variable given X_i = h_i, X_k = h_k. Every tenth
    point is recomputed along a second path (r13 kept) and the two balls must overlap:
    the identity is checked, not assumed.
  * factor_n: correlations r_ij = l_i l_j (|l_i| < 1, any signs), X_i = l_i Z0 + sqrt(1 - l_i^2) Z_i:
        P = int phi(z) prod_i Phi((h_i - l_i z) / sqrt(1 - l_i^2)) dz
    over [-40, 40], the rest bounded by 2 Phi(-40) and added to the ball.
  * factor_t: the same structure divided by sqrt(S / nu): the factor_n integrand at r h,
    averaged over r = sqrt(S / nu) (scipy.integrate.quad, both levels, epsabs 1e-13).

  * bvt, tvt: the textbook scale mixture, X = Z / R with R^2 = chi2(nu) / nu, so
    S = nu R^2 / 2 ~ Gamma(a), a = nu / 2, and
        P = E[Phi_d(R h)],  R = sqrt(S / a),
    taken through the bvn / tvn identities above term by term. Their integrands are
    exp(-R^2 q(t)) and exp(-R^2 q(t)) Phi(R u(t)), whose expectations over S are closed forms
    (the Gamma(a) weight times exp(-S q / a) is (1 + q/a)^-a times the law of S / (1 + q/a)):
        E[exp(-S q / a)]                     = (1 + q/a)^-a,
        E[exp(-S q / a) Phi(sqrt(S / a) u)] = (1 + q/a)^-a T_nu(u / sqrt(1 + q/a)),
    T_nu the Student t cdf (by 2F1: around 0 the series in -v^2 / nu, in the tails the
    incomplete beta form, each continued analytically for acb.integral). What is left is the
    r = 0 start E[prod_i Phi(R h_i)]: one integral over u = log S of entire functions on
    [u_lo, u_hi], the ends bounded with prod Phi <= 1 by P(a, e^u_lo) <= e^(a u_lo) / Gamma(a + 1)
    and Q(a, e^u_hi), each below 2^-80 of P, added to the ball. tvt is computed along two
    paths (r12 kept, r13 kept) whose balls must overlap. Our code integrates the mixture
    itself, in another variable and with double Phi_d (csrc/stats/mvcdf.hpp t_mixture): this
    is an independent evaluation.

A ball that does not settle by 8192 bits, or a point past POINT_SECONDS, is dropped. Arb is C
with no Python-side caches; the pool's workers are replaced every 200 points anyway.
"""

import math
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
from flint import acb, arb, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from arbref import DOUBLE_TINY, MAX_PREC, NotSettled, add_error, intersect, precision, rigorous, to_dd  # noqa: E402

OUT = ROOT / "tests" / "stats" / "ref_mvcdf"
JOBS = 10
POINT_SECONDS = 60
TARGET = 64  # bits: the references are good to ~5e-20 relative, far below the 1e-14 measured


class _Timeout(Exception):
    pass


# ---------------------------------------------------------------- Arb pieces


def _Phi(x):
    """Phi(x) for an arb or acb x: erfc(-x / sqrt 2) / 2 (entire)."""
    return (-x / arb(2).sqrt()).erfc() / 2


_DEADLINE = [math.inf]  # the current point's wall-clock end, checked inside the integrands


def _integral(f, a, b):
    """acb.integral of f(t, analytic) over [a, b] at the current precision. The point's time
    limit is checked here rather than by SIGALRM, whose exception python-flint's callback
    wrapper cannot pass on: past it the integrand returns NaN (the integrator gives up) and
    _Timeout follows."""
    late = []

    def cb(t, analytic):
        if late or time.monotonic() > _DEADLINE[0]:
            late.append(True)
            return acb(math.nan)
        return f(t, analytic)

    v = acb.integral(cb, a, b, rel_tol=arb(2) ** (-ctx.prec), abs_tol=arb(2) ** (-ctx.prec - 1100))
    if late:
        raise _Timeout
    return v


def _bvn(h, k, r):
    h, k, r = arb(h), arb(k), arb(r)
    p0 = _Phi(h) * _Phi(k)
    if r == 0:
        return p0

    def f(t, analytic):
        s, c = t.sin(), t.cos()
        return (-(h * h - 2 * h * k * s + k * k) / (2 * c * c)).exp()

    integral = _integral(f, 0, r.asin())
    return p0 + integral.real / (2 * arb.pi())


def _tvn_term(hi, hk, hj, a, c1, c2):
    """int_0^asin c1 exp(-q(h_i, h_k; s)) Phi(u_j) dt with sigma_ik = s = sin t, sigma_jk = s c2 / c1."""
    if c1 == 0:
        return arb(0)
    hi, hk, hj, a, c1, c2 = (arb(v) for v in (hi, hk, hj, a, c1, c2))

    def f(t, analytic):
        s, c = t.sin(), t.cos()
        b = s * c2 / c1
        oms = c * c
        det = 1 - a * a - b * b - s * s + 2 * a * b * s
        num = hj * oms - a * (hi - s * hk) - b * (hk - s * hi)
        u = num / (oms * det).sqrt(analytic=analytic)
        return (-(hi * hi - 2 * hi * hk * s + hk * hk) / (2 * oms)).exp() * _Phi(u)

    return _integral(f, 0, c1.asin()).real


def _tvn(h, R, keep=(0, 1)):
    i, j = keep
    k = 3 - i - j
    p0 = _bvn(h[i], h[j], R[i][j]) * _Phi(arb(h[k]))
    J = _tvn_term(h[i], h[k], h[j], R[i][j], R[i][k], R[j][k]) + _tvn_term(h[j], h[k], h[i], R[i][j], R[j][k], R[i][k])
    return p0 + J / (2 * arb.pi())


def _factor_n(h, lam):
    h = [arb(v) for v in h]
    lam = [arb(v) for v in lam]
    sd = [(1 - v * v).sqrt() for v in lam]

    def f(z, analytic):
        p = (-z * z / 2).exp() / (2 * arb.pi()).sqrt()
        for hi, li, si in zip(h, lam, sd):
            p = p * _Phi((hi - li * z) / si)
        return p

    L = 40
    v = _integral(f, -L, L).real
    return add_error(v, 2 * _Phi(arb(-L)))


def _settle(fn):
    return rigorous(fn, target=TARGET, floor=DOUBLE_TINY)


# ---------------------------------------------------------------- the sets


def _corr3(rng, kind):
    """A random 3x3 correlation matrix (positive definite)."""
    while True:
        if kind == "random":
            V = rng.normal(size=(3, 3 + rng.integers(0, 4)))
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            R = V @ V.T
        elif kind == "negative3":  # all three negative
            R = np.eye(3)
            for a, b in ((0, 1), (0, 2), (1, 2)):
                R[a, b] = R[b, a] = -rng.uniform(0.01, 0.5)
        elif kind == "negative2":
            R = np.eye(3)
            R[0, 1] = R[1, 0] = rng.uniform(0, 0.95)
            R[0, 2] = R[2, 0] = -rng.uniform(0.01, 0.95)
            R[1, 2] = R[2, 1] = -rng.uniform(0.01, 0.95)
        elif kind == "singular":  # the smallest eigenvalue 10^-3 .. 10^-9
            V = rng.normal(size=(3, 3))
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            R = V @ V.T
            w, Q = np.linalg.eigh(R)
            w[0] = 10.0 ** rng.uniform(-9, -3)
            R = Q @ np.diag(w) @ Q.T
            d = np.sqrt(np.diag(R))
            R = R / d[:, None] / d[None, :]
        np.fill_diagonal(R, 1.0)
        R = (R + R.T) / 2
        if np.linalg.eigvalsh(R)[0] > 1e-12 and np.all(np.abs(R[np.triu_indices(3, 1)]) < 1):
            p = rng.permutation(3)
            return R[np.ix_(p, p)]


def points_bvn():
    rng = np.random.default_rng(20261116)
    H = [-37.0, -25.0, -15.0, -10.0, -7.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    Rs = [
        -0.999999,
        -0.999,
        -0.99,
        -0.95,
        -0.9,
        -0.75,
        -0.5,
        -0.3,
        -0.1,
        -0.01,
        0.01,
        0.1,
        0.3,
        0.5,
        0.75,
        0.9,
        0.95,
        0.99,
        0.999,
        0.999999,
    ]
    pts, region = [], []
    for a in range(len(H)):
        for b in range(a, len(H)):
            for r in Rs:
                pts.append((H[a], H[b], r))
                region.append(0)  # grid
    for _ in range(3000):
        pts.append((rng.uniform(-10, 6), rng.uniform(-10, 6), rng.uniform(-1, 1)))
        region.append(1)  # random
    for _ in range(600):  # near h = -k, r < 0: where Phi2(h, k; -1) is a narrow interval
        h = rng.uniform(-6, 6)
        pts.append((h, -h + rng.choice([-1, 1]) * 10.0 ** rng.uniform(-12, 0), -rng.uniform(0, 1)))
        region.append(2)
    for _ in range(600):  # near-singular
        pts.append((rng.uniform(-8, 4), rng.uniform(-8, 4), rng.choice([-1, 1]) * (1 - 10.0 ** rng.uniform(-12, -2))))
        region.append(3)
    return pts, region


def points_tvn():
    rng = np.random.default_rng(20261117)
    pts, region = [], []
    plan = [
        ("random", 1500, (-7, 5), 0),
        ("random", 400, (-20, -5), 1),
        ("negative3", 400, (-6, 4), 2),
        ("negative2", 400, (-6, 4), 3),
        ("singular", 300, (-6, 4), 4),
    ]
    for kind, n, (lo, hi), reg in plan:
        for _ in range(n):
            R = _corr3(rng, kind)
            h = rng.uniform(lo, hi, size=3)
            pts.append((tuple(h), (R[0, 1], R[0, 2], R[1, 2])))
            region.append(reg)
    return pts, region


def points_factor(dims, seed, t=False):
    rng = np.random.default_rng(seed)
    pts = []
    for d in dims:
        for _ in range(12):
            lam = rng.uniform(-0.95, 0.95, size=d)
            if rng.random() < 0.4:
                lam = np.abs(lam)
            h = rng.uniform(-2.5, 2.5, size=d)
            nu = float(rng.choice([1.0, 2.5, 4.0, 10.0, 30.0])) if t else math.inf
            pts.append((tuple(h), tuple(lam), nu))
    return pts


def points_t(d, n, seed):
    rng = np.random.default_rng(seed)
    nus = [0.5, 1.0, 1.7, 3.0, 4.5, 10.0, 30.0, 200.0, 1e4]
    pts = []
    for q in range(n):
        if d == 2:
            r = rng.uniform(-0.99, 0.99)
            R = (r,)
        else:
            Rm = _corr3(rng, "random" if q % 3 else "negative2")
            R = (Rm[0, 1], Rm[0, 2], Rm[1, 2])
        lo = -30.0 if q % 5 == 0 else -6.0
        h = tuple(rng.uniform(lo, 4.0, size=d))
        pts.append((h, R, nus[q % len(nus)]))
    return pts


def _tcdf(v, a, analytic):
    """T_nu(v), nu = 2a, the Student t cdf, for an acb v near the real axis (analytic: the
    integrator's flag; a box reaching a branch cut gives NaN, and the integrator subdivides).
    |Re v| <= 1/2: 1/2 + v Gamma(a + 1/2) / (sqrt(nu pi) Gamma(a)) 2F1(1/2, a + 1/2; 3/2; -v^2 / nu);
    further out, for Re w < 0 with w = v (or -v, and 1 minus it), I_z(a, 1/2) / 2 with
    z = nu / (nu + w^2): z^a (-w) / sqrt(nu + w^2) Gamma(a + 1/2) / (Gamma(a + 1) sqrt pi)
    2F1(a + 1/2, 1; a + 1; z) / 2, which keeps its relative accuracy in the far tail."""
    v = acb(v)
    nu = 2 * a
    re = v.real
    if re < -0.5 or re > 0.5:
        neg = re < 0
        w = v if neg else -v
        s2 = nu + w * w
        z = nu / s2
        if analytic and not (z.real < 1):
            return acb(math.nan)
        c = (a + arb(0.5)).lgamma() - (a + 1).lgamma() - arb.pi().log() / 2
        tail = (
            (a * (nu.log() - s2.log(analytic=analytic)) + c).exp()
            * (-w)
            / s2.sqrt(analytic=analytic)
            * z.hypgeom_2f1(a + arb(0.5), 1, a + 1)
            / 2
        )
        return tail if neg else 1 - tail
    x = -v * v / nu
    if analytic and not (x.real < 1):
        return acb(math.nan)
    c = ((a + arb(0.5)).lgamma() - a.lgamma()).exp() / (nu * arb.pi()).sqrt()
    return arb(0.5) + v * c * x.hypgeom_2f1(arb(0.5), a + arb(0.5), arb(1.5))


T_EPS = 2.0**-80  # each end of the r = 0 start's integral is below this fraction of P


def _phi_mix(h, a, guess):
    """E[prod_i Phi(sqrt(S / a) h_i)], S ~ Gamma(a): the integral over u = log S, cut at the
    weight's peak u = log a and its scales; the ends left out are bounded (prod Phi <= 1) by
    the Gamma mass beyond them, placed below T_EPS guess (guess: our own value of P)."""
    h = [arb(v) for v in h]
    lga, ra = a.lgamma(), a.sqrt()

    def f(u, analytic):
        s = (u / 2).exp() / ra
        p = (a * u - u.exp() - lga).exp()
        for hi in h:
            p = p * _Phi(s * hi)
        return p

    af = float(a.mid())
    la = math.log(af)
    sig = 1 / math.sqrt(af) if af > 1 else 1.0
    ulo = float(((arb(T_EPS) * guess).log() + (a + 1).lgamma()).mid()) / af
    uhi = math.log(af + 40 * math.sqrt(af) + 750)
    cuts = sorted({c for c in [la + k * sig for k in (-16, -4, -1, 0, 1, 4)] if ulo < c < uhi})
    left = [ulo + 10 * j for j in range(1, int((cuts[0] - ulo) / 10))] if cuts else []
    ends = [ulo] + left + cuts + [uhi]
    total = arb(0)
    for lo, hi in zip(ends[:-1], ends[1:]):
        total += _integral(f, arb(lo), arb(hi)).real
    left_mass = (a * arb(ulo) - (a + 1).lgamma()).exp()  # P(a, x) <= x^a / Gamma(a + 1)
    right_mass = arb(uhi).exp().gamma_upper(a, regularized=1)
    return add_error(total, left_mass + right_mass)


def _t_plackett(hi, hk, a, r, third=None):
    """A Plackett term of _bvn / _tvn mixed over S (r its upper limit sin):
    int_0^asin r (1 + q/a)^-a [T_nu(u / sqrt(1 + q/a))] dt, q = (h_i^2 - 2 h_i h_k sin t + h_k^2)
    / (2 cos^2 t). third None: no T factor (bvt). third = (h_j, None, None): u = h_j (tvn's
    Phi2(h_i, h_k; r) Phi(h_j)). third = (h_j, r_kept, c2): u the u_j of _tvn_term, whose c1 is r."""
    if r == 0:
        return arb(0)
    hi, hk, r = arb(hi), arb(hk), arb(r)
    if third is not None:
        hj, rk, c2 = third
        hj = arb(hj)

    def f(t, analytic):
        s, c = t.sin(), t.cos()
        oms = c * c
        L = (1 + (hi * hi - 2 * hi * hk * s + hk * hk) / (2 * oms) / a).log(analytic=analytic)
        w = (-a * L).exp()
        if third is None:
            return w
        if rk is None:
            u = hj
        else:  # _tvn_term's u_j with sigma_ik = s, sigma_jk = s c2 / r, r_ij = rk
            ra, b = arb(rk), s * arb(c2) / r
            det = 1 - ra * ra - b * b - s * s + 2 * ra * b * s
            u = (hj * oms - ra * (hi - s * hk) - b * (hk - s * hi)) / (oms * det).sqrt(analytic=analytic)
        return w * _tcdf(u * (-L / 2).exp(), a, analytic)

    return _integral(f, 0, r.asin()).real


def _tmix(h, R, nu, guess, keep=(0, 1)):
    """P(X <= h) for the t: the r = 0 start plus the mixed Plackett terms (d = 2: R = (r,);
    d = 3: R = (r12, r13, r23), along the path that keeps r_keep)."""
    a = arb(nu) / 2
    P = _phi_mix(h, a, guess)
    if len(h) == 2:
        J = _t_plackett(h[0], h[1], a, R[0])
    else:
        RR = [[1, R[0], R[1]], [R[0], 1, R[2]], [R[1], R[2], 1]]
        i, j = keep
        k = 3 - i - j
        J = (
            _t_plackett(h[i], h[j], a, RR[i][j], (h[k], None, None))
            + _t_plackett(h[i], h[k], a, RR[i][k], (h[j], RR[i][j], RR[j][k]))
            + _t_plackett(h[j], h[k], a, RR[j][k], (h[i], RR[i][j], RR[i][k]))
        )
    return P + J / (2 * arb.pi())


T_SECONDS = 600


def _w_tmix(p):
    h, R, nu = p
    _DEADLINE[0] = time.monotonic() + T_SECONDS
    try:
        guess = arb(_guess(h, R, nu))
        if len(h) == 2:
            v = _settle(lambda: _tmix(h, R, nu, guess))
        else:  # two paths: disjoint balls are a bug here, and stop the build
            v1 = _settle(lambda: _tmix(h, R, nu, guess, (0, 1)))
            v2 = _settle(lambda: _tmix(h, R, nu, guess, (0, 2)))
            with precision(MAX_PREC):  # the intersection rounds to the working precision
                v = intersect(v1, v2)
        return to_dd(v)
    except NotSettled, _Timeout:
        return (math.nan, math.nan)
    finally:
        _DEADLINE[0] = math.inf


def _guess(h, R, nu):
    """Our own value, only to place the left end of the range (bounded by T_EPS of it)."""
    import stardust as sd
    from stardust._core import stats as C

    d = len(h)
    S = np.eye(d)
    if d == 2:
        S[0, 1] = S[1, 0] = R[0]
    else:
        for q, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            S[i, j] = S[j, i] = R[q]
    v, _ = C.mv_cdf(
        sd.asarray(S[None]),
        sd.asarray(np.array([h])),
        sd.asarray(np.zeros(1, np.int64)),
        sd.asarray(np.array([nu])),
        1e-6,
        0,
    )
    return max(float(np.asarray(v)[0]), 1e-300)


# ---------------------------------------------------------------- workers


def _run(fn):
    _DEADLINE[0] = time.monotonic() + POINT_SECONDS
    try:
        return to_dd(_settle(fn))
    except NotSettled, _Timeout:
        return (math.nan, math.nan)
    finally:
        _DEADLINE[0] = math.inf


def _w_bvn(p):
    h, k, r = p
    return _run(lambda: _bvn(h, k, r))


def _w_tvn(p):
    h, (r12, r13, r23) = p
    R = [[1, r12, r13], [r12, 1, r23], [r13, r23, 1]]
    return _run(lambda: _tvn(h, R))


def _w_tvn_check(p):
    h, (r12, r13, r23) = p
    R = [[1, r12, r13], [r12, 1, r23], [r13, r23, 1]]
    _DEADLINE[0] = time.monotonic() + POINT_SECONDS
    try:
        a = _settle(lambda: _tvn(h, R, (0, 1)))
        b = _settle(lambda: _tvn(h, R, (0, 2)))
    except NotSettled, _Timeout:
        return None  # not checked
    finally:
        _DEADLINE[0] = math.inf
    return bool(a.overlaps(b))


def _w_factor_n(p):
    h, lam, _ = p
    return _run(lambda: _factor_n(h, lam))


def _w_factor_t(p):
    from scipy import integrate, special, stats

    h, lam, nu = p
    h, lam = np.asarray(h), np.asarray(lam)
    sd = np.sqrt(1 - lam * lam)

    def inner(r):
        g = lambda z: np.exp(-z * z / 2) / math.sqrt(2 * math.pi) * np.prod(special.ndtr((h * r - lam * z) / sd))
        return integrate.quad(g, -12, 12, epsabs=1e-14, epsrel=1e-13, limit=400)[0]

    chi = stats.chi(nu)
    dens = lambda r: chi.pdf(r * math.sqrt(nu)) * math.sqrt(nu)
    hi = chi.ppf(1 - 1e-16) / math.sqrt(nu)
    v = integrate.quad(
        lambda r: dens(r) * inner(r), 0, hi, epsabs=1e-13, epsrel=1e-12, limit=400, points=[1.0] if hi > 1 else None
    )[0]
    return (v, 0.0)


def _pool_map(fn, pts, jobs):
    with multiprocessing.get_context("fork").Pool(jobs, maxtasksperchild=200) as pool:
        return pool.map(fn, pts, chunksize=4)


def build(name, jobs):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if name == "bvn":
        pts, region = points_bvn()
        res = _pool_map(_w_bvn, pts, jobs)
        a = np.array(pts)
        out = dict(h=a[:, 0], k=a[:, 1], r=a[:, 2], region=np.array(region))
    elif name == "tvn":
        pts, region = points_tvn()
        checks = _pool_map(_w_tvn_check, pts[::10], jobs)
        bad = [10 * i for i, c in enumerate(checks) if c is False]
        if bad:
            raise SystemExit(f"tvn: the two paths disagree at {bad}")
        print(
            f"tvn: {sum(c is True for c in checks)} points agree along two paths "
            f"({sum(c is None for c in checks)} unchecked: timeout)",
            flush=True,
        )
        res = _pool_map(_w_tvn, pts, jobs)
        out = dict(h=np.array([p[0] for p in pts]), r=np.array([p[1] for p in pts]), region=np.array(region))
    elif name in ("factor_n", "factor_t"):
        t = name == "factor_t"
        pts = points_factor(range(4, 11) if not t else range(2, 9), 7 if not t else 8, t)
        res = _pool_map(_w_factor_t if t else _w_factor_n, pts, jobs)
        L = max(len(p[0]) for p in pts)
        h = np.full((len(pts), L), np.inf)  # padded with +inf: those coordinates drop out
        lam = np.zeros((len(pts), L))
        for i, p in enumerate(pts):
            h[i, : len(p[0])] = p[0]
            lam[i, : len(p[1])] = p[1]
        out = dict(h=h, lam=lam, d=np.array([len(p[0]) for p in pts]), nu=np.array([p[2] for p in pts]))
    elif name in ("bvt", "tvt"):
        d = 2 if name == "bvt" else 3
        pts = points_t(d, 120 if d == 2 else 60, 20 + d)
        res = _pool_map(_w_tmix, pts, jobs)
        out = dict(h=np.array([p[0] for p in pts]), r=np.array([p[1] for p in pts]), nu=np.array([p[2] for p in pts]))
    else:
        raise SystemExit(f"unknown set {name}")
    r = np.array(res)
    out["hi"], out["lo"] = r[:, 0], r[:, 1]
    keep = ~np.isnan(out["hi"])
    out = {k: v[keep] for k, v in out.items()}
    np.savez_compressed(OUT / f"{name}.npz", **out)
    print(f"{name}: {keep.sum()} of {len(keep)} points, {time.time() - t0:.0f} s", flush=True)


def main(argv):
    jobs = JOBS
    names = []
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        else:
            names.append(a)
    for n in names or ["bvn", "tvn", "factor_n", "factor_t", "bvt", "tvt"]:
        build(n, jobs)


if __name__ == "__main__":
    main(sys.argv[1:])
