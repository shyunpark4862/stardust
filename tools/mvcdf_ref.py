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
    bvt, tvt  the t at d = 2, 3: the scale mixture, outer tanh-sinh (mpmath, 30 digits)
              over rigorous inner Phi2 / Phi3 (Arb), ~1e-20

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

  * bvt, tvt: the textbook scale mixture, S = nu R^2 / 2 ~ Gamma(nu / 2):
        P = int_0^inf x^(a-1) e^-x / Gamma(a) Phi_d(sqrt(x / a) h) dx,  a = nu / 2,
    in u = log x by mpmath's tanh-sinh at 30 digits, cut at the weight's peak u = log a and
    its scales, Phi_d at each node from the bvn / tvn balls above (2^-64). The ends left out
    are bounded with Phi_d <= 1: P(a, e^u_lo) <= 1e-22 P, P(S > a + 40 sqrt(a) + 750) <
    1e-300. A point is kept when tanh-sinh's error estimate is below 1e-21 P and two degrees
    of it agree to 1e-20 relative. Not a ball, but an independent evaluation: our code
    integrates in another variable with an analytic left end (csrc/stats/mvcdf.hpp
    t_mixture), and its Phi_d are doubles.

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
from arbref import DOUBLE_TINY, NotSettled, add_error, rigorous, to_dd  # noqa: E402

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
    Rs = [-0.999999, -0.999, -0.99, -0.95, -0.9, -0.75, -0.5, -0.3, -0.1, -0.01,
          0.01, 0.1, 0.3, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.999999]
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
    plan = [("random", 1500, (-7, 5), 0), ("random", 400, (-20, -5), 1), ("negative3", 400, (-6, 4), 2),
            ("negative2", 400, (-6, 4), 3), ("singular", 300, (-6, 4), 4)]
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


def _w_tmix(p):
    import mpmath as mpm

    h, R, nu = p
    d = len(h)
    if d == 2:
        inner = lambda s: _bvn(s * arb(h[0]), s * arb(h[1]), R[0])
    else:
        RR = [[1, R[0], R[1]], [R[0], 1, R[2]], [R[1], R[2], 1]]
        inner = lambda s: _tvn([s * arb(v) for v in h], RR)
    _DEADLINE[0] = time.monotonic() + 600
    try:
        mpm.mp.dps = 30
        a = mpm.mpf(nu) / 2
        lga = mpm.loggamma(a)
        guess = float(_guess(h, R, nu))

        def f(u):
            s = mpm.nstr(mpm.sqrt(mpm.exp(u) / a), 40)  # parsed at each working precision below
            v = _settle(lambda: inner(arb(s)))
            return mpm.exp(a * u - mpm.exp(u) - lga) * mpm.mpf(v.mid().str(30, radius=False))

        la = mpm.log(a)
        sig = 1 / mpm.sqrt(a) if a > 1 else mpm.mpf(1)
        ulo = (mpm.log(mpm.mpf(1e-22) * guess) + mpm.loggamma(a + 1)) / a
        uhi = mpm.log(a + 40 * mpm.sqrt(a) + 750)
        cuts = sorted({float(c) for c in [la + k * sig for k in (-16, -4, -1, 0, 1, 4)] if ulo < c < uhi})
        left = [float(ulo) + 10 * j for j in range(1, int((cuts[0] - float(ulo)) / 10))] if cuts else []
        pts_ = [ulo] + [mpm.mpf(c) for c in left + cuts] + [uhi]
        v1, e1 = mpm.quad(f, pts_, error=True, maxdegree=7)
        v2 = mpm.quad(f, pts_, maxdegree=8)
        if not (e1 <= 1e-21 * abs(v1) and abs(v1 - v2) <= 1e-20 * abs(v1)):
            return (math.nan, math.nan)
        hi_ = float(v1)
        return (hi_, float(v1 - hi_))
    except (NotSettled, _Timeout):
        return (math.nan, math.nan)
    finally:
        _DEADLINE[0] = math.inf


def _guess(h, R, nu):
    """Our own value, only to place the left end of the range (bounded by 1e-22 of it)."""
    import stardust as sd
    from stardust._core import stats as C

    d = len(h)
    S = np.eye(d)
    if d == 2:
        S[0, 1] = S[1, 0] = R[0]
    else:
        for q, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            S[i, j] = S[j, i] = R[q]
    v, _ = C.mv_cdf(sd.asarray(S[None]), sd.asarray(np.array([h])), sd.asarray(np.zeros(1, np.int64)),
                    sd.asarray(np.array([nu])), 1e-6, 0)
    return max(float(np.asarray(v)[0]), 1e-300)


# ---------------------------------------------------------------- workers

def _run(fn):
    _DEADLINE[0] = time.monotonic() + POINT_SECONDS
    try:
        return to_dd(_settle(fn))
    except (NotSettled, _Timeout):
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
    except (NotSettled, _Timeout):
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
    v = integrate.quad(lambda r: dens(r) * inner(r), 0, hi, epsabs=1e-13, epsrel=1e-12, limit=400,
                       points=[1.0] if hi > 1 else None)[0]
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
        print(f"tvn: {sum(c is True for c in checks)} points agree along two paths "
              f"({sum(c is None for c in checks)} unchecked: timeout)", flush=True)
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
